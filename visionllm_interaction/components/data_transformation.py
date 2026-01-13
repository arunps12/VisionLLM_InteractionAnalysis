import os
from datetime import datetime
from typing import Any, Dict, Set, Tuple

from visionllm_interaction.logger.logger import get_logger
from visionllm_interaction.exception.custom_exception import CustomException
from visionllm_interaction.entity.config_entity import DataTransformationConfig
from visionllm_interaction.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)
from visionllm_interaction.utils.main_utils import (
    read_yaml,
    write_yaml,
    load_json,
    write_json,
)

logger = get_logger(__name__)


class DataTransformation:
    """
    staging/cleaning step.

    Inputs:
      - ingestion manifest (raw paths)
      - data validation report (tells which annotation IDs were invalid)

    Outputs:
      - cleaned train/val annotation jsons in:
          artifacts/<timestamp>/data_transformation/annotations/
      - training manifest pointing to cleaned jsons:
          artifacts/<timestamp>/data_transformation/training_manifest.yaml
    """

    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.config = data_transformation_config
            self.ingestion_artifact = data_ingestion_artifact
            self.validation_artifact = data_validation_artifact
        except Exception as e:
            raise CustomException("Failed to initialize DataTransformation", e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_mkdir(dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)

    @staticmethod
    def _validate_manifest_structure(manifest: Dict[str, Any]) -> None:
        required_top = ["dataset_name", "dataset_format", "paths", "annotations"]
        for k in required_top:
            if k not in manifest:
                raise CustomException(f"Manifest missing key '{k}'")

        for k in ["train_images", "val_images"]:
            if k not in manifest["paths"]:
                raise CustomException(f"Manifest.paths missing key '{k}'")

        for k in ["train", "val"]:
            if k not in manifest["annotations"]:
                raise CustomException(f"Manifest.annotations missing key '{k}'")

    @staticmethod
    def _extract_invalid_ann_ids_from_report(report: Dict[str, Any], split: str) -> Set[int]:
        """
        Extract invalid annotation IDs from validation report.

        Prefer full list:
          - train_stats.invalid_bbox_ann_ids
        Fallback to examples :
          - train_stats.invalid_bbox_examples[].ann_id
        """
        stats_key = f"{split}_stats"
        stats = report.get(stats_key, {}) or {}

        
        full_ids = stats.get("invalid_bbox_ann_ids", None)
        if isinstance(full_ids, list) and full_ids:
            out: Set[int] = set()
            for x in full_ids:
                if isinstance(x, int):
                    out.add(x)
                elif isinstance(x, str) and x.isdigit():
                    out.add(int(x))
            return out

       
        examples = stats.get("invalid_bbox_examples", []) or []
        out: Set[int] = set()
        for ex in examples:
            ann_id = ex.get("ann_id")
            if isinstance(ann_id, int):
                out.add(ann_id)
            elif isinstance(ann_id, str) and ann_id.isdigit():
                out.add(int(ann_id))
        return out

    @staticmethod
    def _drop_annotations_by_id(coco: Dict[str, Any], drop_ids: Set[int]) -> Tuple[Dict[str, Any], int]:
        anns = coco.get("annotations", [])
        if not drop_ids:
            return coco, 0

        before = len(anns)
        coco["annotations"] = [a for a in anns if a.get("id") not in drop_ids]
        dropped = before - len(coco["annotations"])
        return coco, dropped

    def _write_training_manifest(
        self,
        base_manifest: Dict[str, Any],
        cleaned_train_ann: str,
        cleaned_val_ann: str,
    ) -> None:
        created_at = datetime.now().isoformat()

        training_manifest = {
            "dataset_name": base_manifest.get("dataset_name"),
            "dataset_format": base_manifest.get("dataset_format"),
            "created_at": created_at,
            "source_manifest": self.ingestion_artifact.manifest_file_path,
            "transformation": {
                "purpose": "clean_annotations_only",
                "used_validation_report": self.validation_artifact.report_file_path,
            },
            "paths": {
                "train_images": base_manifest["paths"]["train_images"],
                "val_images": base_manifest["paths"]["val_images"],
            },
            "annotations": {
                "train": cleaned_train_ann,
                "val": cleaned_val_ann,
            },
        }

        write_yaml(self.config.training_manifest_file_path, training_manifest)
        logger.info(f"Training manifest written to: {self.config.training_manifest_file_path}")

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("===== Data Transformation Started (clean annotations) =====")
            self._safe_mkdir(self.config.data_transformation_dir)
            self._safe_mkdir(self.config.cleaned_annotation_dir)

            # 1) Read ingestion manifest
            base_manifest = read_yaml(self.ingestion_artifact.manifest_file_path)
            self._validate_manifest_structure(base_manifest)

            raw_train_ann = base_manifest["annotations"]["train"]
            raw_val_ann = base_manifest["annotations"]["val"]

            # 2) Read validation report and extract invalid annotation IDs
            validation_report = read_yaml(self.validation_artifact.report_file_path)

            if not validation_report.get("validated", False):
                raise CustomException(
                    f"Validation report says dataset is not validated: {self.validation_artifact.report_file_path}"
                )

            drop_train_ids = self._extract_invalid_ann_ids_from_report(validation_report, "train")
            drop_val_ids = self._extract_invalid_ann_ids_from_report(validation_report, "val")

            logger.info(f"Train invalid annotation IDs to drop: {len(drop_train_ids)}")
            logger.info(f"Val invalid annotation IDs to drop: {len(drop_val_ids)}")

            # 3) Load raw COCO JSONs 
            train_coco = load_json(raw_train_ann)
            val_coco = load_json(raw_val_ann)

            # 4) Drop by IDs
            train_coco, dropped_train = self._drop_annotations_by_id(train_coco, drop_train_ids)
            val_coco, dropped_val = self._drop_annotations_by_id(val_coco, drop_val_ids)

            logger.info(f"Dropped invalid train annotations: {dropped_train}")
            logger.info(f"Dropped invalid val annotations: {dropped_val}")

            # 5) Write cleaned JSONs 
            write_json(self.config.cleaned_train_annotation_file, train_coco)
            write_json(self.config.cleaned_val_annotation_file, val_coco)

            logger.info(f"Cleaned train annotations written: {self.config.cleaned_train_annotation_file}")
            logger.info(f"Cleaned val annotations written: {self.config.cleaned_val_annotation_file}")

            # 6) Write training manifest
            self._write_training_manifest(
                base_manifest=base_manifest,
                cleaned_train_ann=self.config.cleaned_train_annotation_file,
                cleaned_val_ann=self.config.cleaned_val_annotation_file,
            )

            logger.info("===== Data Transformation Completed Successfully =====")

            return DataTransformationArtifact(
                data_transformation_dir=self.config.data_transformation_dir,
                cleaned_train_annotation_file=self.config.cleaned_train_annotation_file,
                cleaned_val_annotation_file=self.config.cleaned_val_annotation_file,
                training_manifest_file_path=self.config.training_manifest_file_path,
                dropped_train_annotation_ids=dropped_train,
                dropped_val_annotation_ids=dropped_val,
            )

        except CustomException:
            raise
        except Exception as e:
            raise CustomException("Error in data transformation pipeline", e)
