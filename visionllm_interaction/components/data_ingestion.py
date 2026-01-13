import os
import zipfile
from datetime import datetime
from pathlib import Path

import kagglehub

from visionllm_interaction.logger.logger import get_logger
from visionllm_interaction.exception.custom_exception import CustomException
from visionllm_interaction.entity.config_entity import DataIngestionConfig
from visionllm_interaction.entity.artifact_entity import DataIngestionArtifact
from visionllm_interaction.utils.main_utils import write_yaml


logger = get_logger(__name__)


class DataIngestion:
    """
    Data ingestion for COCO .

    Write artifacts/<timestamp>/data_ingestion/data_manifest.yaml
    Return DataIngestionArtifact
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.config = data_ingestion_config
        except Exception as e:
            raise CustomException("Failed to initialize DataIngestion", e)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _is_zip(path: str) -> bool:
        p = Path(path)
        return p.is_file() and p.suffix.lower() == ".zip"

    @staticmethod
    def _safe_mkdir(dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)

    def _extract_zip(self, zip_path: str, extract_to: str) -> str:
        """Extract zip_path into extract_to and return extract_to."""
        try:
            self._safe_mkdir(extract_to)
            logger.info(f"Extracting zip: {zip_path} -> {extract_to}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_to)
            return extract_to
        except Exception as e:
            raise CustomException("Failed to extract dataset zip", e)

    def _find_coco_root(self, base_dir: str) -> str:
        """
        Locate the COCO root directory under base_dir.
        Returns the path to the detected COCO root.
        """
        try:
            base = Path(base_dir)

            train_dirname = Path(self.config.raw_train_image_dir).name  # "train2017"

            # Direct match
            if (base / train_dirname).exists() and (base / "annotations").exists():
                return str(base)

            # Check one and two levels deep
            candidates = (
                [base]
                + [p for p in base.glob("*") if p.is_dir()]
                + [p for p in base.glob("*/*") if p.is_dir()]
            )

            for c in candidates:
                if (c / train_dirname).exists() and (c / "annotations").exists():
                    return str(c)

            raise CustomException(
                f"Could not locate COCO root under: {base_dir}. "
                f"Expected folders like '{train_dirname}/' and 'annotations/'."
            )
        except CustomException:
            raise
        except Exception as e:
            raise CustomException("Failed while locating COCO root directory", e)

    def _copytree_if_missing(self, src: str, dst: str) -> None:
        """Copy a directory tree if dst doesn't exist."""
        if os.path.exists(dst):
            logger.info(f"Raw path already exists (skip copy): {dst}")
            return
        import shutil

        logger.info(f"Copying: {src} -> {dst}")
        shutil.copytree(src, dst)

    def _copyfile_if_missing(self, src: str, dst: str) -> None:
        """Copy a file if dst doesn't exist."""
        if os.path.exists(dst):
            logger.info(f"Raw file already exists (skip copy): {dst}")
            return
        import shutil

        self._safe_mkdir(os.path.dirname(dst))
        logger.info(f"Copying: {src} -> {dst}")
        shutil.copy2(src, dst)

    def _prepare_raw_data_dir(self, downloaded_path: str) -> None:
        """
        Ensures config.raw_data_dir contains the expected COCO train/val + annotations.
        - If downloaded_path is a zip, extract it into scratch temp
        - Locate COCO root
        - Copy only required dirs/files into raw_data_dir (train2017, val2017 if present, annotations/)
        """
        try:
            # Ensure scratch raw root exists
            self._safe_mkdir(self.config.raw_data_dir)

            # If zip, extract into a temp folder under scratch raw
            working_dir = downloaded_path
            if self._is_zip(downloaded_path):
                tmp_dir = os.path.join(self.config.raw_data_dir, "_tmp_extract")
                working_dir = self._extract_zip(downloaded_path, tmp_dir)

            coco_root = self._find_coco_root(working_dir)
            logger.info(f"Detected COCO root: {coco_root}")

            train_dirname = Path(self.config.raw_train_image_dir).name  # train2017
            val_dirname = Path(self.config.raw_val_image_dir).name      # val2017

            src_train = os.path.join(coco_root, train_dirname)
            src_val = os.path.join(coco_root, val_dirname)
            src_ann_dir = os.path.join(coco_root, "annotations")

            # Destination (on /scratch)
            dst_train = self.config.raw_train_image_dir
            dst_val = self.config.raw_val_image_dir
            dst_ann_dir = self.config.raw_annotation_dir

            # ---- Required: train + annotations ----
            if not os.path.exists(src_train):
                raise CustomException(f"Missing expected folder in dataset: {src_train}")
            if not os.path.exists(src_ann_dir):
                raise CustomException(f"Missing expected folder in dataset: {src_ann_dir}")

            # Copy only train2017
            self._copytree_if_missing(src_train, dst_train)

            # Copy val2017 only if present 
            if os.path.exists(src_val):
                self._copytree_if_missing(src_val, dst_val)
            else:
                logger.warning(f"val images folder not found at: {src_val} (continuing)")

            # Copy annotations directory (contains instances_*.json etc.)
            self._copytree_if_missing(src_ann_dir, dst_ann_dir)

            # ---- Ensure required annotation exists (train) ----
            if not os.path.exists(self.config.raw_train_annotation_file):
                raise CustomException(
                    f"Train annotation file not found after staging: {self.config.raw_train_annotation_file}"
                )

            # Warn if val annotation is missing
            if not os.path.exists(self.config.raw_val_annotation_file):
                logger.warning(
                    f"Val annotation file not found after staging: {self.config.raw_val_annotation_file} (continuing)"
                )

            logger.info("Raw COCO train/val data prepared under: %s", self.config.raw_data_dir)

        except CustomException:
            raise
        except Exception as e:
            raise CustomException("Failed to prepare raw COCO directory", e)

    def _validate_raw_paths(self) -> None:
        """Validate that required raw paths exist."""
        try:
            required_dirs = [
                self.config.raw_data_dir,
                self.config.raw_train_image_dir,
                self.config.raw_annotation_dir,
            ]
            required_files = [
                self.config.raw_train_annotation_file,
            ]

            for d in required_dirs:
                if not os.path.exists(d):
                    raise CustomException(f"Required directory does not exist: {d}")

            for f in required_files:
                if not os.path.exists(f):
                    raise CustomException(f"Required file does not exist: {f}")

            #  warnings
            if not os.path.exists(self.config.raw_val_image_dir):
                logger.warning(f"Validation: val image dir missing: {self.config.raw_val_image_dir}")

            if not os.path.exists(self.config.raw_val_annotation_file):
                logger.warning(f"Validation: val annotation file missing: {self.config.raw_val_annotation_file}")

            logger.info("Raw path validation completed.")

        except CustomException:
            raise
        except Exception as e:
            raise CustomException("Failed to validate raw dataset paths", e)

    def _write_manifest(self) -> None:
        try:
            self._safe_mkdir(self.config.data_ingestion_dir)

            manifest_path = self.config.manifest_file_path
            created_at = datetime.now().isoformat()

            manifest = {
                "dataset_name": self.config.dataset_name,
                "dataset_format": self.config.dataset_format,
                "created_at": created_at,
                "ingestion": {
                "mode": self.config.ingestion_mode
                },
                "paths": {
                    "train_images": self.config.raw_train_image_dir,
                    "val_images": self.config.raw_val_image_dir,
                },
                "annotations": {
                    "train": self.config.raw_train_annotation_file,
                    "val": self.config.raw_val_annotation_file,
                },
            }

            write_yaml(manifest_path, manifest)
            logger.info(f"Wrote data manifest: {manifest_path}")

        except Exception as e:
            raise CustomException("Failed to write data manifest file", e)


    # ----------------------------------------------------------------------
    # Main entry
    # ----------------------------------------------------------------------
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Main ingestion entrypoint."""
        try:
            logger.info("===== Data Ingestion Started (COCO train/val only) =====")

            # Ensure ingestion artifact dir exists
            self._safe_mkdir(self.config.data_ingestion_dir)

            #  Download dataset (kagglehub cache path)
            downloaded_path = kagglehub.dataset_download(self.config.dataset_name)
            logger.info(f"Downloaded dataset to: {downloaded_path}")

            # Prepare raw data directory (on /scratch via constants)
            self._prepare_raw_data_dir(downloaded_path)

            #  Validate raw paths
            self._validate_raw_paths()

            #  Write manifest into artifacts/<timestamp>/data_ingestion/
            self._write_manifest()

            logger.info("===== Data Ingestion Completed Successfully =====")

            return DataIngestionArtifact(
                data_ingestion_dir=self.config.data_ingestion_dir,
                manifest_file_path=self.config.manifest_file_path,
                ingestion_mode=self.config.ingestion_mode,
                dataset_format=self.config.dataset_format,
            )

        except CustomException:
            raise
        except Exception as e:
            raise CustomException("Error in data ingestion pipeline", e)
