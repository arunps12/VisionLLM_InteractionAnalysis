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
    - schema.yaml 
    - val is REQUIRED
    - drops failed invalid bboxes up to a limit

    Report path:
      artifacts/<timestamp>/data_validation/data_validation_report.yaml
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

    @staticmethod
    def _recompute_category_counts(annotations: List[Dict[str, Any]]) -> Dict[int, int]:
        per_cat = Counter()
        for ann in annotations:
            cid = ann.get("category_id")
            if cid is not None:
                per_cat[cid] += 1
        return dict(per_cat)

    @staticmethod
    def _count_images_with_annotations(annotations: List[Dict[str, Any]]) -> int:
        ann_per_image = Counter()
        for ann in annotations:
            iid = ann.get("image_id")
            if iid is not None:
                ann_per_image[iid] += 1
        return sum(1 for _, c in ann_per_image.items() if c > 0)

    def _validate_cross_refs_and_bbox(
        self,
        coco: Dict[str, Any],
        images_dir: str,
        split_name: str,
        drop_invalid_bbox: bool,
        max_invalid_bbox_allowed: int,
    ) -> Dict[str, Any]:
        """
        Strict validation:
        - annotation.image_id exists in images
        - annotation.category_id exists in categories
        - each images.file_name exists on disk
        - bbox is valid and inside image bounds

        
        - if invalid bboxes <= max_invalid_bbox_allowed and drop_invalid_bbox=True,
          drop those annotations and continue.

        Returns stats dict (includes invalid bbox counts + examples).
        """
        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])

        self._validate_ids_unique(images, "id", f"{split_name}.images")
        self._validate_ids_unique(annotations, "id", f"{split_name}.annotations")
        self._validate_ids_unique(categories, "id", f"{split_name}.categories")

        img_by_id = {im["id"]: im for im in images}
        cat_ids = set([c["id"] for c in categories])

        missing_files = 0
        bad_image_refs = 0
        bad_cat_refs = 0

        invalid_bbox = 0
        invalid_bbox_ann_ids = set()
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

        # Validate annotations refs + bbox bounds
        for ann in annotations:
            image_id = ann.get("image_id")
            category_id = ann.get("category_id")

            if image_id not in img_by_id:
                bad_image_refs += 1
                continue

            if category_id not in cat_ids:
                bad_cat_refs += 1
                continue

            bbox = ann.get("bbox")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(isinstance(x, (int, float)) for x in bbox)
            ):
                invalid_bbox += 1
                invalid_bbox_ann_ids.add(ann.get("id"))
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
                invalid_bbox_ann_ids.add(ann.get("id"))
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
                invalid_bbox_ann_ids.add(ann.get("id"))
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

        # Hard failures first
        if missing_files > 0:
            raise CustomException(f"{split_name}: {missing_files} image files listed in JSON are missing on disk.")
        if bad_image_refs > 0:
            raise CustomException(f"{split_name}: {bad_image_refs} annotations reference missing image_id.")
        if bad_cat_refs > 0:
            raise CustomException(f"{split_name}: {bad_cat_refs} annotations reference missing category_id.")

        # Invalid bbox policy: drop or fail
        dropped_invalid_bboxes = 0
        if invalid_bbox > 0:
            if drop_invalid_bbox and invalid_bbox <= max_invalid_bbox_allowed:
                original_count = len(annotations)
                annotations = [a for a in annotations if a.get("id") not in invalid_bbox_ann_ids]
                dropped_invalid_bboxes = original_count - len(annotations)
                coco["annotations"] = annotations  # mutate to cleaned set
                logger.warning(
                    f"{split_name}: dropped {dropped_invalid_bboxes} invalid bbox annotations "
                    f"(found={invalid_bbox}, limit={max_invalid_bbox_allowed})."
                )
            else:
                raise CustomException(
                    f"{split_name}: {invalid_bbox} invalid bboxes found (limit={max_invalid_bbox_allowed}, "
                    f"drop_invalid_bbox={drop_invalid_bbox}). Examples: {invalid_bbox_examples}"
                )

        # Usefulness checks AFTER potential dropping
        if len(images) == 0:
            raise CustomException(f"{split_name}: no images found in JSON.")
        if len(coco["annotations"]) == 0:
            raise CustomException(f"{split_name}: no annotations found in JSON (after dropping invalid bboxes).")
        if len(categories) == 0:
            raise CustomException(f"{split_name}: no categories found in JSON.")

        images_with_anns = self._count_images_with_annotations(coco["annotations"])
        if images_with_anns == 0:
            raise CustomException(f"{split_name}: no image has any annotation (after dropping invalid bboxes).")

        per_cat_counts = self._recompute_category_counts(coco["annotations"])

        return {
            "num_images_json": len(images),
            "num_annotations_json": len(coco["annotations"]),
            "num_categories": len(categories),
            "images_with_annotations": images_with_anns,
            "per_category_annotation_counts": per_cat_counts,
            "invalid_bbox_found": invalid_bbox,
            "invalid_bbox_dropped": dropped_invalid_bboxes,
            "invalid_bbox_examples": invalid_bbox_examples,
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
            "policy": {
                "require_val": getattr(self.config, "require_val", True),
                "drop_invalid_bbox": getattr(self.config, "drop_invalid_bbox", True),
                "max_invalid_bbox_allowed": getattr(self.config, "max_invalid_bbox_allowed", 100),
            },
            "error": None,
        }

        try:
            logger.info("===== Data Validation Started (COCO train+val) =====")
            self._safe_mkdir(self.config.data_validation_dir)

           
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

            train_stats = self._validate_cross_refs_and_bbox(
                train_coco,
                train_images,
                "train",
                drop_invalid_bbox=self.config.drop_invalid_bbox,
                max_invalid_bbox_allowed=self.config.max_invalid_bbox_allowed,
            )

            val_stats = self._validate_cross_refs_and_bbox(
                val_coco,
                val_images,
                "val",
                drop_invalid_bbox=self.config.drop_invalid_bbox,
                max_invalid_bbox_allowed=self.config.max_invalid_bbox_allowed,
            )

            # Ensure each category appears in both splits (post-drop)
            train_cat_ids = set(train_stats["per_category_annotation_counts"].keys())
            val_cat_ids = set(val_stats["per_category_annotation_counts"].keys())
            if train_cat_ids != val_cat_ids:
                raise CustomException(
                    "Category presence mismatch: some categories have annotations only in train or only in val "
                    "(after dropping invalid bboxes)."
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
            report["error"] = getattr(ce, "message", None) or str(ce)
            logger.error(f"Data validation failed: {report['error']}")
            raise

        except Exception as e:
            report["error"] = str(e)
            logger.error("Data validation failed with unexpected exception.")
            raise CustomException("Error in data validation pipeline", e)

        finally:
            # write report (PASS or FAIL)
            try:
                self._safe_mkdir(self.config.data_validation_dir)
                write_yaml(self.config.report_file_path, report)
                print(f"[DataValidation] Report written to: {self.config.report_file_path}")
                logger.info(f"Validation report written to: {self.config.report_file_path}")
            except Exception as e:
                print(f"[DataValidation] FAILED to write report: {e}")
                logger.error(f"Failed to write validation report: {e}")
