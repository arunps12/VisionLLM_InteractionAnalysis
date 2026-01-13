import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from visionllm_interaction.logger.logger import get_logger
from visionllm_interaction.exception.custom_exception import CustomException
from visionllm_interaction.entity.config_entity import DataValidationConfig
from visionllm_interaction.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from visionllm_interaction.utils.main_utils import read_yaml, write_yaml

logger = get_logger(__name__)


class DataValidation:
    """
    Validates COCO train+val dataset using:
    - ingestion manifest YAML
    - schema.yaml (existence check for now)
    - val is REQUIRED
    """

    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        try:
            self.config = data_validation_config
            self.ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CustomException("Failed to initialize DataValidation", e)

    # ------------------------------------------------------------------
    # Helpers 
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_mkdir(dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)

    @staticmethod
    def _load_json(file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise CustomException(f"Failed to load JSON: {file_path}", e)

    @staticmethod
    def _list_images(img_dir: str) -> List[str]:
        if not os.path.isdir(img_dir):
            return []
        exts = {".jpg", ".jpeg", ".png"}
        files: List[str] = []
        for p in Path(img_dir).rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(str(p))
        return files

    @staticmethod
    def _validate_required_fields(obj: Dict[str, Any], required: List[str], where: str) -> None:
        missing = [k for k in required if k not in obj]
        if missing:
            raise CustomException(f"Missing keys {missing} in {where}")

    def _validate_manifest(self) -> Dict[str, Any]:
        manifest_path = self.ingestion_artifact.manifest_file_path
        if not os.path.exists(manifest_path):
            raise CustomException(f"Manifest not found: {manifest_path}")

        manifest = read_yaml(manifest_path)

        self._validate_required_fields(
            manifest,
            ["dataset_name", "dataset_format", "paths", "annotations"],
            "manifest",
        )
        self._validate_required_fields(manifest["paths"], ["train_images", "val_images"], "manifest.paths")
        self._validate_required_fields(manifest["annotations"], ["train", "val"], "manifest.annotations")

        return manifest

    def _validate_paths_exist(self, manifest: Dict[str, Any]) -> None:
        train_images = manifest["paths"]["train_images"]
        val_images = manifest["paths"]["val_images"]
        train_ann = manifest["annotations"]["train"]
        val_ann = manifest["annotations"]["val"]

        if not os.path.isdir(train_images):
            raise CustomException(f"Train images dir missing: {train_images}")
        if not os.path.isfile(train_ann):
            raise CustomException(f"Train annotation file missing: {train_ann}")

        # Val 
        if self.config.require_val:
            if not os.path.isdir(val_images):
                raise CustomException(f"Val images dir missing: {val_images}")
            if not os.path.isfile(val_ann):
                raise CustomException(f"Val annotation file missing: {val_ann}")

        ann_dir = os.path.dirname(train_ann)
        if not os.path.isdir(ann_dir):
            raise CustomException(f"Annotations dir missing: {ann_dir}")

    def _validate_coco_json_schema_minimal(self, coco: Dict[str, Any], split_name: str) -> None:
        for k in ["images", "annotations", "categories"]:
            if k not in coco:
                raise CustomException(f"COCO JSON missing top-level key '{k}' in {split_name}")

        if (
            not isinstance(coco["images"], list)
            or not isinstance(coco["annotations"], list)
            or not isinstance(coco["categories"], list)
        ):
            raise CustomException(f"COCO JSON keys must be lists in {split_name}")

        if coco["categories"]:
            self._validate_required_fields(coco["categories"][0], ["id", "name"], f"{split_name}.categories[0]")
        if coco["images"]:
            self._validate_required_fields(
                coco["images"][0], ["id", "file_name", "width", "height"], f"{split_name}.images[0]"
            )
        if coco["annotations"]:
            self._validate_required_fields(
                coco["annotations"][0],
                ["id", "image_id", "category_id", "bbox", "area", "iscrowd"],
                f"{split_name}.annotations[0]",
            )

    def _validate_ids_unique(self, items: List[Dict[str, Any]], key: str, where: str) -> None:
        ids = [it.get(key) for it in items]
        if len(ids) != len(set(ids)):
            raise CustomException(f"Duplicate '{key}' detected in {where}")

    def _validate_cross_refs_and_bbox(
        self,
        coco: Dict[str, Any],
        images_dir: str,
        split_name: str,
    ) -> Dict[str, Any]:
        """
        Strict validation:
        - annotation.image_id exists in images
        - annotation.category_id exists in categories
        - each images.file_name exists on disk
        - bbox is valid and inside image bounds
        Returns stats dict.
        """
        images = coco["images"]
        annotations = coco["annotations"]
        categories = coco["categories"]

        self._validate_ids_unique(images, "id", f"{split_name}.images")
        self._validate_ids_unique(annotations, "id", f"{split_name}.annotations")
        self._validate_ids_unique(categories, "id", f"{split_name}.categories")

        img_by_id = {im["id"]: im for im in images}
        cat_ids = set([c["id"] for c in categories])

        missing_files = 0
        bad_image_refs = 0
        bad_cat_refs = 0
        invalid_bbox = 0

        invalid_bbox_examples: List[Dict[str, Any]] = []
        MAX_EXAMPLES = 10

        # Check JSON images -> disk
        for im in images:
            file_name = im.get("file_name")
            if not file_name:
                raise CustomException(f"Missing file_name in {split_name}.images entry")

            fpath = os.path.join(images_dir, file_name)
            if not os.path.exists(fpath):
                missing_files += 1

            w = im.get("width")
            h = im.get("height")
            if not isinstance(w, (int, float)) or not isinstance(h, (int, float)) or w <= 0 or h <= 0:
                raise CustomException(f"Invalid width/height for image_id={im.get('id')} in {split_name}")

        per_cat = Counter()
        ann_per_image = Counter()

        for ann in annotations:
            image_id = ann.get("image_id")
            category_id = ann.get("category_id")

            if image_id not in img_by_id:
                bad_image_refs += 1
                continue

            if category_id not in cat_ids:
                bad_cat_refs += 1
                continue

            per_cat[category_id] += 1
            ann_per_image[image_id] += 1

            bbox = ann.get("bbox")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(isinstance(x, (int, float)) for x in bbox)
            ):
                invalid_bbox += 1
                if len(invalid_bbox_examples) < MAX_EXAMPLES:
                    invalid_bbox_examples.append(
                        {
                            "ann_id": ann.get("id"),
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": bbox,
                            "image_width": img_by_id.get(image_id, {}).get("width"),
                            "image_height": img_by_id.get(image_id, {}).get("height"),
                            "reason": "bbox malformed",
                        }
                    )
                continue

            x, y, bw, bh = bbox
            if bw <= 0 or bh <= 0 or x < 0 or y < 0:
                invalid_bbox += 1
                if len(invalid_bbox_examples) < MAX_EXAMPLES:
                    invalid_bbox_examples.append(
                        {
                            "ann_id": ann.get("id"),
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": bbox,
                            "image_width": img_by_id.get(image_id, {}).get("width"),
                            "image_height": img_by_id.get(image_id, {}).get("height"),
                            "reason": "bbox non-positive or negative origin",
                        }
                    )
                continue

            im = img_by_id[image_id]
            iw, ih = im["width"], im["height"]
            if (x + bw) > iw or (y + bh) > ih:
                invalid_bbox += 1
                if len(invalid_bbox_examples) < MAX_EXAMPLES:
                    invalid_bbox_examples.append(
                        {
                            "ann_id": ann.get("id"),
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": bbox,
                            "image_width": iw,
                            "image_height": ih,
                            "reason": "bbox out of bounds",
                        }
                    )
                continue

        # Strict failures with actionable details
        if missing_files > 0:
            raise CustomException(f"{split_name}: {missing_files} image files listed in JSON are missing on disk.")
        if bad_image_refs > 0:
            raise CustomException(f"{split_name}: {bad_image_refs} annotations reference missing image_id.")
        if bad_cat_refs > 0:
            raise CustomException(f"{split_name}: {bad_cat_refs} annotations reference missing category_id.")
        if invalid_bbox > 0:
            raise CustomException(
                f"{split_name}: {invalid_bbox} invalid bboxes found. Examples: {invalid_bbox_examples}"
            )

        if len(images) == 0:
            raise CustomException(f"{split_name}: no images found in JSON.")
        if len(annotations) == 0:
            raise CustomException(f"{split_name}: no annotations found in JSON.")
        if len(categories) == 0:
            raise CustomException(f"{split_name}: no categories found in JSON.")

        images_with_anns = sum(1 for _img_id, c in ann_per_image.items() if c > 0)
        if images_with_anns == 0:
            raise CustomException(f"{split_name}: no image has any annotation.")

        return {
            "num_images_json": len(images),
            "num_annotations_json": len(annotations),
            "num_categories": len(categories),
            "images_with_annotations": images_with_anns,
            "per_category_annotation_counts": dict(per_cat),
        }

    def _validate_train_val_category_consistency(
        self,
        train_coco: Dict[str, Any],
        val_coco: Dict[str, Any],
    ) -> None:
        train_cats = {(c["id"], c["name"]) for c in train_coco.get("categories", [])}
        val_cats = {(c["id"], c["name"]) for c in val_coco.get("categories", [])}

        if train_cats != val_cats:
            raise CustomException(
                "Train/Val category mismatch detected. "
                "categories in instances_train2017.json and instances_val2017.json must match exactly."
            )

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def initiate_data_validation(self) -> DataValidationArtifact:
        
        report: Dict[str, Any] = {
            "validation_status": "FAIL",
            "validated": False,
            "manifest_path": getattr(self.ingestion_artifact, "manifest_file_path", None),
            "schema_path": self.config.schema_file_path,
            "error": None,
        }

        try:
            logger.info("===== Data Validation Started (COCO strict train+val) =====")
            self._safe_mkdir(self.config.data_validation_dir)

            # Schema must exist (content enforcement can be added later)
            if not os.path.exists(self.config.schema_file_path):
                raise CustomException(f"Schema file not found: {self.config.schema_file_path}")

            manifest = self._validate_manifest()
            self._validate_paths_exist(manifest)

            train_images = manifest["paths"]["train_images"]
            val_images = manifest["paths"]["val_images"]
            train_ann = manifest["annotations"]["train"]
            val_ann = manifest["annotations"]["val"]

            train_img_files = self._list_images(train_images)
            val_img_files = self._list_images(val_images)

            if len(train_img_files) == 0:
                raise CustomException(f"No train images found on disk in: {train_images}")
            if self.config.require_val and len(val_img_files) == 0:
                raise CustomException(f"No val images found on disk in: {val_images}")

            train_coco = self._load_json(train_ann)
            val_coco = self._load_json(val_ann)

            self._validate_coco_json_schema_minimal(train_coco, "train")
            self._validate_coco_json_schema_minimal(val_coco, "val")

            self._validate_train_val_category_consistency(train_coco, val_coco)

            train_stats = self._validate_cross_refs_and_bbox(train_coco, train_images, "train")
            val_stats = self._validate_cross_refs_and_bbox(val_coco, val_images, "val")

            train_cat_ids = set(train_stats["per_category_annotation_counts"].keys())
            val_cat_ids = set(val_stats["per_category_annotation_counts"].keys())
            if train_cat_ids != val_cat_ids:
                raise CustomException(
                    "Category presence mismatch: some categories have annotations only in train or only in val."
                )

            report.update(
                {
                    "validation_status": "PASS",
                    "validated": True,
                    "paths": {
                        "train_images_dir": train_images,
                        "val_images_dir": val_images,
                        "train_annotation_file": train_ann,
                        "val_annotation_file": val_ann,
                    },
                    "disk_counts": {
                        "train_images_files": len(train_img_files),
                        "val_images_files": len(val_img_files),
                    },
                    "train_stats": train_stats,
                    "val_stats": val_stats,
                }
            )

            logger.info("===== Data Validation Completed Successfully =====")

            return DataValidationArtifact(
                data_validation_dir=self.config.data_validation_dir,
                report_file_path=self.config.report_file_path,
                validated=True,
            )

        except CustomException as ce:
            # Safe error text (don’t depend on CustomException internals)
            report["error"] = getattr(ce, "message", None) or str(ce)
            logger.error(f"Data validation failed: {report['error']}")
            raise

        except Exception as e:
            report["error"] = str(e)
            logger.error("Data validation failed with unexpected exception.")
            raise CustomException("Error in data validation pipeline", e)

        finally:
            # ALWAYS write report (PASS or FAIL)
            try:
                self._safe_mkdir(self.config.data_validation_dir)
                write_yaml(self.config.report_file_path, report)
                logger.info(f"Validation report written to: {self.config.report_file_path}")
            except Exception as e:
                logger.error(f"Failed to write validation report: {e}")
