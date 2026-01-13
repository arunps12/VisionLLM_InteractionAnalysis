# File: visionllm_interaction/components/model_trainer.py

import os
import time
from typing import Any, Dict, List, Optional

import cv2
import dagshub
import mlflow
import optuna
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.auto import tqdm

from dotenv import load_dotenv

from visionllm_interaction.logger.logger import get_logger
from visionllm_interaction.exception.custom_exception import CustomException
from visionllm_interaction.entity.config_entity import ModelTrainerConfig
from visionllm_interaction.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from visionllm_interaction.utils.main_utils import read_yaml, write_yaml
from visionllm_interaction.utils.ml_utils import (
    set_seed,
    get_device,
    collate_fn,
    ensure_dir,
    save_checkpoint,
)

logger = get_logger(__name__)

# Load .env once at module load time
load_dotenv(override=False)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class COCODetectionFromManifest(Dataset):
    """
    Minimal COCO detection dataset reading:
      - images_dir
      - COCO instances json file (cleaned)

    Returns:
      image: FloatTensor [3,H,W] in [0,1]
      target: dict with boxes (xyxy), labels, image_id, area, iscrowd
    """

    def __init__(self, images_dir: str, ann_file: str, max_images: Optional[int] = None):
        self.images_dir = images_dir
        self.ann_file = ann_file
        self.max_images = max_images

        coco = self._load_json(self.ann_file)

        self.images = coco.get("images", [])
        self.annotations = coco.get("annotations", [])
        self.categories = coco.get("categories", [])

        if not self.images or not self.annotations:
            raise CustomException(f"COCO JSON seems empty: {ann_file}")

        # Map category_id -> contiguous labels (start at 1; 0 is background)
        if self.categories:
            cat_ids = sorted({c["id"] for c in self.categories})
        else:
            cat_ids = sorted({a["category_id"] for a in self.annotations})

        self.cat_id_to_contig = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}

        self.img_id_to_info = {img["id"]: img for img in self.images}
        self.ann_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.annotations:
            self.ann_by_img.setdefault(ann["image_id"], []).append(ann)

        # Keep only images that have annotations
        self.img_ids = [img["id"] for img in self.images if img["id"] in self.ann_by_img]

        if self.max_images is not None:
            self.img_ids = self.img_ids[: int(self.max_images)]

        if len(self.img_ids) == 0:
            raise CustomException("No images with annotations found after filtering.")

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        import json

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise CustomException(f"Failed to load JSON: {path}", e)

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        image_id = self.img_ids[idx]
        info = self.img_id_to_info[image_id]
        img_path = os.path.join(self.images_dir, info["file_name"])

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        image = torch.from_numpy(image_rgb).permute(2, 0, 1)  # [3,H,W], CPU float32

        anns = self.ann_by_img.get(image_id, [])
        boxes, labels, area, iscrowd = [], [], [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_contig[ann["category_id"]])
            area.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.tensor(area, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }
        return image, target


# ----------------------------------------------------------------------
# Model factory
# ----------------------------------------------------------------------
def build_fasterrcnn_resnet50_fpn(num_classes: int, weights: str = "DEFAULT") -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ----------------------------------------------------------------------
# Training helpers
# ----------------------------------------------------------------------
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every_n_steps: int = 50,
) -> float:
    model.train()
    running = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)
    for step, (images, targets) in enumerate(pbar, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu())
        running += loss_val

        if step % max(int(log_every_n_steps), 1) == 0:
            pbar.set_postfix(loss=loss_val)

    return running / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    
    model.train()
    running = 0.0

    pbar = tqdm(loader, desc="Val", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())

        loss_val = float(loss.detach().cpu())
        running += loss_val
        pbar.set_postfix(loss=loss_val)

    return running / max(len(loader), 1)


# ----------------------------------------------------------------------
# ModelTrainer
# ----------------------------------------------------------------------
class ModelTrainer:
    """
    Trains Faster R-CNN using training manifest produced by DataTransformation:
      artifacts/<timestamp>/data_transformation/training_manifest.yaml

    Supports Optuna HPO for:
      - batch_size (categorical)
      - lr (loguniform)

    Logs to MLflow (DagsHub) if enabled in config/model.yaml and env vars exist.
    """

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.config = model_trainer_config
            self.transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException("Failed to initialize ModelTrainer", e)

    def _load_model_yaml(self) -> Dict[str, Any]:
        if not os.path.exists(self.config.model_config_file_path):
            raise CustomException(f"model.yaml not found: {self.config.model_config_file_path}")
        return read_yaml(self.config.model_config_file_path)

    def _load_training_manifest(self) -> Dict[str, Any]:
        manifest_path = self.transformation_artifact.training_manifest_file_path
        if not os.path.exists(manifest_path):
            raise CustomException(f"Training manifest not found: {manifest_path}")
        return read_yaml(manifest_path)

    def _setup_mlflow(self, model_cfg: Dict[str, Any]):
        mlflow_cfg = model_cfg.get("mlflow", {}) or {}
        if not mlflow_cfg.get("enabled", False):
            logger.info("MLflow disabled in model.yaml")
            return None

        try:
            import dagshub
            import mlflow
        except Exception as e:
            raise CustomException("dagshub or mlflow not installed.", e)

        dagshub.init(
                repo_owner="arunps12",
                repo_name="VisionLLM_InteractionAnalysis",
                mlflow=True,
            )
        exp_name = (
            mlflow_cfg.get("experiment_name")
            or model_cfg.get("experiment", {}).get("name", "default")
        )
        mlflow.set_experiment(exp_name)

        logger.info(f"MLflow enabled via DagsHub. Experiment={exp_name}")
        return mlflow


    @staticmethod
    def _validate_manifest_structure(manifest: Dict[str, Any]) -> None:
        for k in ["paths", "annotations"]:
            if k not in manifest:
                raise CustomException(f"Training manifest missing key: {k}")
        for k in ["train_images", "val_images"]:
            if k not in manifest["paths"]:
                raise CustomException(f"Training manifest.paths missing key: {k}")
        for k in ["train", "val"]:
            if k not in manifest["annotations"]:
                raise CustomException(f"Training manifest.annotations missing key: {k}")

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logger.info("===== Model Trainer Started =====")
            ensure_dir(self.config.model_trainer_dir)

            model_cfg = self._load_model_yaml()
            train_manifest = self._load_training_manifest()
            self._validate_manifest_structure(train_manifest)

            # ---------------------------------------------------------
            # Read YAML config
            # ---------------------------------------------------------
            exp_cfg = model_cfg.get("experiment", {}) or {}
            exp_name = exp_cfg.get("name", "experiment")
            run_name_prefix = exp_cfg.get("run_name_prefix", "run")

            training_cfg = model_cfg.get("training", {}) or {}
            seed = int(training_cfg.get("seed", 42))
            set_seed(seed)

            device_cfg = training_cfg.get("device", "auto")
            device = get_device(device_cfg)
            logger.info(f"Using device: {device}")

            epochs = int(training_cfg.get("epochs", 6))
            num_workers = int(training_cfg.get("num_workers", 4))
            pin_memory = bool(training_cfg.get("pin_memory", True)) and (device.type == "cuda")

            opt_cfg = training_cfg.get("optimizer", {}) or {}
            base_lr = float(opt_cfg.get("lr", 1e-4))
            weight_decay = float(opt_cfg.get("weight_decay", 0.0))

            dl_cfg = training_cfg.get("dataloader", {}) or {}
            base_batch_size = int(dl_cfg.get("batch_size", 4))
            shuffle_train = bool(dl_cfg.get("shuffle_train", True))
            shuffle_val = bool(dl_cfg.get("shuffle_val", False))

            dbg_cfg = training_cfg.get("debug", {}) or {}
            debug_enabled = bool(dbg_cfg.get("enabled", False))
            max_train_images = dbg_cfg.get("max_train_images", None)
            max_val_images = dbg_cfg.get("max_val_images", None)
            if debug_enabled:
                logger.warning(f"DEBUG enabled: max_train_images={max_train_images} max_val_images={max_val_images}")

            log_cfg = model_cfg.get("logging", {}) or {}
            log_every_n_steps = int(log_cfg.get("log_every_n_steps", 50))
            save_every_n_epochs = int(log_cfg.get("save_every_n_epochs", 1))

            out_cfg = model_cfg.get("outputs", {}) or {}
            metric_to_monitor = out_cfg.get("metric_to_monitor", "val_loss")
            save_best_only = bool(out_cfg.get("save_best_only", True))
            checkpoint_name = out_cfg.get("checkpoint_name", "fasterrcnn_best.pt")

            mcfg = model_cfg.get("model", {}) or {}
            weights = mcfg.get("weights", "DEFAULT")
            num_classes = int(mcfg.get("num_classes", 81))

            # Data paths from training manifest
            train_images_dir = train_manifest["paths"]["train_images"]
            val_images_dir = train_manifest["paths"]["val_images"]
            train_ann = train_manifest["annotations"]["train"]
            val_ann = train_manifest["annotations"]["val"]

            if not os.path.isdir(train_images_dir):
                raise CustomException(f"Train images dir missing: {train_images_dir}")
            if not os.path.isdir(val_images_dir):
                raise CustomException(f"Val images dir missing: {val_images_dir}")
            if not os.path.isfile(train_ann):
                raise CustomException(f"Train annotation missing: {train_ann}")
            if not os.path.isfile(val_ann):
                raise CustomException(f"Val annotation missing: {val_ann}")

            # Resolve output paths 
            best_model_path = os.path.join(self.config.model_trainer_dir, checkpoint_name)
            last_model_path = self.config.last_model_file_path  

            # MLflow
            mlflow = self._setup_mlflow(model_cfg)

            # ---------------------------------------------------------
            # Prepare datasets once 
            # ---------------------------------------------------------
            train_ds = COCODetectionFromManifest(
                train_images_dir,
                train_ann,
                max_images=max_train_images if debug_enabled else None,
            )
            val_ds = COCODetectionFromManifest(
                val_images_dir,
                val_ann,
                max_images=max_val_images if debug_enabled else None,
            )

            # ---------------------------------------------------------
            # Train function for one configuration
            # ---------------------------------------------------------
            global_best_val = float("inf")
            global_best_params: Dict[str, Any] = {}
            global_best_trial: Optional[int] = None

            def run_training_once(batch_size: int, lr: float, trial_number: Optional[int]) -> float:
                nonlocal global_best_val, global_best_params, global_best_trial

                # loaders 
                train_loader = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=shuffle_train,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    pin_memory=pin_memory,
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=batch_size,
                    shuffle=shuffle_val,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    pin_memory=pin_memory,
                )

                model = build_fasterrcnn_resnet50_fpn(num_classes=num_classes, weights=weights).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                # MLflow run per trial/config
                mlflow_run = None
                if mlflow is not None:
                    name = f"{run_name_prefix}_bs{batch_size}_lr{lr:.2e}"
                    if trial_number is not None:
                        name = f"{name}_trial{trial_number}"
                    mlflow_run = mlflow.start_run(run_name=name)
                    mlflow.log_params(
                        {
                            "experiment_name": exp_name,
                            "batch_size": batch_size,
                            "lr": lr,
                            "epochs": epochs,
                            "num_classes": num_classes,
                            "weights": weights,
                            "seed": seed,
                            "device": str(device),
                            "weight_decay": weight_decay,
                            "shuffle_train": shuffle_train,
                            "shuffle_val": shuffle_val,
                            "training_manifest": self.transformation_artifact.training_manifest_file_path,
                            "train_images_dir": train_images_dir,
                            "val_images_dir": val_images_dir,
                            "train_ann": train_ann,
                            "val_ann": val_ann,
                        }
                    )

                best_val_this_run = float("inf")

                for epoch in range(1, epochs + 1):
                    logger.info(f"[trial={trial_number}] Epoch {epoch}/{epochs} | bs={batch_size} lr={lr:.2e}")

                    tr = train_one_epoch(
                        model=model,
                        loader=train_loader,
                        optimizer=optimizer,
                        device=device,
                        log_every_n_steps=log_every_n_steps,
                    )
                    va = validate_one_epoch(model=model, loader=val_loader, device=device)

                    logger.info(f"[trial={trial_number}] train_loss={tr:.6f} val_loss={va:.6f}")

                    if mlflow is not None:
                        mlflow.log_metrics({"train_loss": tr, "val_loss": va}, step=epoch)

                    if va < best_val_this_run:
                        best_val_this_run = va

                    # Save checkpoint per epoch interval 
                    if save_every_n_epochs > 0 and (epoch % save_every_n_epochs == 0):
                        trial_tag = f"trial_{trial_number}" if trial_number is not None else "single_run"
                        epoch_ckpt = os.path.join(
                            self.config.model_trainer_dir,
                            f"{trial_tag}_epoch_{epoch}.pt",
                        )
                        save_checkpoint(
                            model,
                            epoch_ckpt,
                            extra={
                                "epoch": epoch,
                                "batch_size": batch_size,
                                "lr": lr,
                                "val_loss": float(va),
                                "seed": seed,
                            },
                        )

                # Trial-specific last checkpoint 
                trial_tag = f"trial_{trial_number}" if trial_number is not None else "single_run"
                trial_last_path = os.path.join(self.config.model_trainer_dir, f"{trial_tag}_last.pt")
                save_checkpoint(
                    model,
                    trial_last_path,
                    extra={
                        "batch_size": batch_size,
                        "lr": lr,
                        "epochs": epochs,
                        "seed": seed,
                        "best_val_loss_this_run": float(best_val_this_run),
                        "trial_number": trial_number,
                    },
                )

                
                if trial_number is None:
                    save_checkpoint(
                        model,
                        last_model_path,
                        extra={
                            "batch_size": batch_size,
                            "lr": lr,
                            "epochs": epochs,
                            "seed": seed,
                            "best_val_loss_this_run": float(best_val_this_run),
                        },
                    )

                # Update global best and save best checkpoint
                if best_val_this_run < global_best_val:
                    global_best_val = best_val_this_run
                    global_best_params = {"batch_size": batch_size, "lr": lr}
                    global_best_trial = trial_number

                    save_checkpoint(
                        model,
                        best_model_path,
                        extra={
                            "batch_size": batch_size,
                            "lr": lr,
                            "epochs": epochs,
                            "seed": seed,
                            "best_val_loss": float(global_best_val),
                            "trial_number": global_best_trial,
                        },
                    )

                if mlflow is not None and mlflow_run is not None:
                    mlflow.log_metric("best_val_loss_this_run", float(best_val_this_run))
                    mlflow.log_param("best_model_path", best_model_path)
                    mlflow.end_run()

                return float(best_val_this_run)

            # ---------------------------------------------------------
            # HPO with Optuna
            # ---------------------------------------------------------
            hpo_cfg = model_cfg.get("hpo", {}) or {}
            hpo_enabled = bool(hpo_cfg.get("enabled", False))
            trials_summary: List[Dict[str, Any]] = []

            if hpo_enabled:
                if (hpo_cfg.get("framework") or "").lower() != "optuna":
                    raise CustomException("Only Optuna HPO is supported in this trainer (hpo.framework must be 'optuna').")

                direction = hpo_cfg.get("direction", "minimize")
                n_trials = int(hpo_cfg.get("n_trials", 10))
                timeout_seconds = hpo_cfg.get("timeout_seconds", None)

                sampler_cfg = (hpo_cfg.get("sampler", {}) or {})
                sampler_name = (sampler_cfg.get("name", "tpe") or "tpe").lower()
                sampler_seed = int(sampler_cfg.get("seed", seed))

                if sampler_name == "tpe":
                    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
                else:
                    sampler = optuna.samplers.TPESampler(seed=sampler_seed)

                study = optuna.create_study(direction=direction, sampler=sampler)

                search_space = hpo_cfg.get("search_space", {}) or {}
                bs_space = search_space.get("batch_size", {}) or {}
                lr_space = search_space.get("lr", {}) or {}

                def objective(trial: optuna.Trial) -> float:
                    # batch size
                    bs_vals = bs_space.get("values", [2, 4])
                    batch_size = trial.suggest_categorical("batch_size", bs_vals)

                    # lr (loguniform)
                    low = float(lr_space.get("low", 1e-5))
                    high = float(lr_space.get("high", 1e-3))
                    lr = trial.suggest_float("lr", low, high, log=True)

                    val_loss = run_training_once(batch_size=int(batch_size), lr=float(lr), trial_number=trial.number)

                    trials_summary.append(
                        {
                            "trial": trial.number,
                            "batch_size": int(batch_size),
                            "lr": float(lr),
                            "best_val_loss": float(val_loss),
                        }
                    )

                    return float(val_loss)

                study.optimize(
                    objective,
                    n_trials=n_trials,
                    timeout=float(timeout_seconds) if timeout_seconds else None,
                )

                best_params = dict(study.best_params)
                best_value = float(study.best_value)
                best_n_trials = n_trials

                logger.info(f"Optuna best params: {best_params}")
                logger.info(f"Optuna best val_loss: {best_value:.6f}")

                
                if save_best_only:
                    # keep last model file representing the best run
                    try:
                        import shutil

                        shutil.copy2(best_model_path, last_model_path)
                    except Exception:
                        pass

            else:
                # Single run with defaults from yaml
                best_value = run_training_once(batch_size=base_batch_size, lr=base_lr, trial_number=None)
                best_params = {"batch_size": base_batch_size, "lr": base_lr}
                best_n_trials = 0

            # ---------------------------------------------------------
            # Write report
            # ---------------------------------------------------------
            report = {
                "status": "SUCCESS",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_yaml": self.config.model_config_file_path,
                "training_manifest": self.transformation_artifact.training_manifest_file_path,
                "device": str(device),
                "experiment": {
                    "name": exp_name,
                    "run_name_prefix": run_name_prefix,
                },
                "model": {
                    "type": (model_cfg.get("model", {}) or {}).get("type", "fasterrcnn_resnet50_fpn"),
                    "weights": weights,
                    "num_classes": num_classes,
                },
                "training": {
                    "epochs": epochs,
                    "num_workers": num_workers,
                    "pin_memory": bool(training_cfg.get("pin_memory", True)),
                    "optimizer": {
                        "name": (opt_cfg.get("name") or "adam"),
                        "lr_default": base_lr,
                        "weight_decay": weight_decay,
                    },
                    "dataloader": {
                        "batch_size_default": base_batch_size,
                        "shuffle_train": shuffle_train,
                        "shuffle_val": shuffle_val,
                    },
                },
                "outputs": {
                    "save_best_only": save_best_only,
                    "metric_to_monitor": metric_to_monitor,
                    "checkpoint_name": checkpoint_name,
                    "best_model_path": best_model_path,
                    "last_model_path": last_model_path,
                    "report_path": self.config.report_file_path,
                },
                "hpo": {
                    "enabled": hpo_enabled,
                    "framework": "optuna" if hpo_enabled else None,
                    "direction": (hpo_cfg.get("direction") if hpo_enabled else None),
                    "objective_metric": (hpo_cfg.get("objective_metric") if hpo_enabled else None),
                    "n_trials": best_n_trials,
                    "best_params": best_params,
                    "best_val_loss": float(best_value),
                    "trials_summary": trials_summary[:200],
                },
            }

            write_yaml(self.config.report_file_path, report)
            logger.info(f"Training report written: {self.config.report_file_path}")
            logger.info("===== Model Trainer Completed Successfully =====")

            return ModelTrainerArtifact(
                model_trainer_dir=self.config.model_trainer_dir,
                best_model_path=best_model_path,
                last_model_path=last_model_path,
                training_report_path=self.config.report_file_path,
                best_metric_name=str(metric_to_monitor),
                best_metric_value=float(best_value),
                hpo_enabled=bool(hpo_enabled),
                best_params=best_params,
                n_trials=best_n_trials,
            )

        except CustomException:
            raise
        except Exception as e:
            raise CustomException("Error in model trainer pipeline", e)
