import os
from datetime import datetime

from visionllm_interaction.constants.training_pipeline import (
    ARTIFACTS_DIR,
    TRAINING_PIPELINE_NAME,
    DATA_INGESTION_DIR_NAME,
    DATA_INGESTION_MODE,
    RAW_DATA_DIR,
    RAW_COCO_TRAIN_IMAGE_DIR,
    RAW_COCO_VAL_IMAGE_DIR,
    RAW_COCO_ANNOTATION_DIR,
    RAW_COCO_TRAIN_ANN_FILE,
    RAW_COCO_VAL_ANN_FILE,
    DATA_INGESTION_MANIFEST_FILE,
    DATASET_NAME,
)

from visionllm_interaction.constants.training_pipeline import (
    DATA_VALIDATION_DIR_NAME,
    SCHEMA_FILE_PATH,
    DATA_VALIDATION_REPORT_FILE,
    DROP_INVALID_BBOX,
    MAX_INVALID_BBOX_ALLOWED,
)

from visionllm_interaction.constants.training_pipeline import (
    DATA_TRANSFORMATION_DIR_NAME,
    DATA_TRANSFORMATION_ANNOTATION_DIR_NAME,
    CLEANED_TRAIN_ANN_FILE_NAME,
    CLEANED_VAL_ANN_FILE_NAME,
    DATA_TRANSFORMATION_MANIFEST_FILE,
)

from visionllm_interaction.constants.training_pipeline import (
    ARTIFACTS_DIR,
    TRAINING_PIPELINE_NAME,
    MODEL_TRAINER_DIR_NAME,
    MODEL_CONFIG_FILE_PATH,
    MODEL_TRAINER_REPORT_FILE_NAME,
    MODEL_TRAINER_BEST_MODEL_FILE_NAME,
    MODEL_TRAINER_LAST_MODEL_FILE_NAME,
)

class TrainingPipelineConfig:
    """
    Global training pipeline configuration.
    Creates a timestamped artifact directory for each run.
    """

    def __init__(self, timestamp: datetime = datetime.now()):
        timestamp_str = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        self.pipeline_name: str = TRAINING_PIPELINE_NAME
        self.artifact_name: str = ARTIFACTS_DIR
        self.artifact_dir: str = os.path.join(self.artifact_name, timestamp_str)

        self.timestamp: str = timestamp_str


class DataIngestionConfig:
    """
    Configuration for Data Ingestion stage.

    - Points to RAW COCO dataset staged on /scratch (images + annotations)
    - Defines where ingestion artifacts (manifest) are written (inside artifacts/<timestamp>/data_ingestion)
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        
        self.dataset_name: str = DATASET_NAME
        self.ingestion_mode: str = DATA_INGESTION_MODE
        self.dataset_format: str = "raw"  

        # -----------------------------
        # Artifact directory for stage
        # -----------------------------
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            DATA_INGESTION_DIR_NAME,
        )

        # -----------------------------
        # RAW COCO dataset paths (on /scratch)
        # -----------------------------
        self.raw_data_dir: str = RAW_DATA_DIR

        self.raw_train_image_dir: str = RAW_COCO_TRAIN_IMAGE_DIR
        self.raw_val_image_dir: str = RAW_COCO_VAL_IMAGE_DIR

        self.raw_annotation_dir: str = RAW_COCO_ANNOTATION_DIR
        self.raw_train_annotation_file: str = RAW_COCO_TRAIN_ANN_FILE
        self.raw_val_annotation_file: str = RAW_COCO_VAL_ANN_FILE

        # -----------------------------
        # Ingestion artifact output
        # -----------------------------
        self.manifest_file_path: str = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_MANIFEST_FILE,
        )

class DataValidationConfig:
    """
    Configuration for Data Validation stage.

    - Reads the ingestion manifest YAML
    - Validates train+val images and COCO json annotations (strict)
    - Writes a validation report YAML inside artifacts/<timestamp>/data_validation/
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # -----------------------------
        # Artifact directory for stage
        # -----------------------------
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            DATA_VALIDATION_DIR_NAME,
        )

        # -----------------------------
        # Schema file path
        # -----------------------------
        self.schema_file_path: str = SCHEMA_FILE_PATH

        # -----------------------------
        # Validation report output
        # -----------------------------
        self.report_file_path: str = os.path.join(
            self.data_validation_dir,
            DATA_VALIDATION_REPORT_FILE,
        )

        # -----------------------------
        # val is REQUIRED
        # -----------------------------
        self.require_val: bool = True

        # Policy knobs
        self.drop_invalid_bbox: bool = DROP_INVALID_BBOX
        self.max_invalid_bbox_allowed: int = MAX_INVALID_BBOX_ALLOWED


class DataTransformationConfig:
    """
    Configuration for Data Transformation stage.

    This stage:
    - reads ingestion manifest (raw paths)
    - reads validation report to know which annotation IDs to drop
    - writes cleaned COCO JSONs + training manifest
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            DATA_TRANSFORMATION_DIR_NAME,
        )

        self.cleaned_annotation_dir: str = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_ANNOTATION_DIR_NAME,
        )

        self.cleaned_train_annotation_file: str = os.path.join(
            self.cleaned_annotation_dir,
            CLEANED_TRAIN_ANN_FILE_NAME,
        )

        self.cleaned_val_annotation_file: str = os.path.join(
            self.cleaned_annotation_dir,
            CLEANED_VAL_ANN_FILE_NAME,
        )

        self.training_manifest_file_path: str = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_MANIFEST_FILE,
        )


class ModelTrainerConfig:
    """
    Configuration for Model Trainer stage.

    Reads:
      - config/model.yaml
    Writes:
      - artifacts/<timestamp>/model_trainer/
    """

    def __init__(self, training_pipeline_config: "TrainingPipelineConfig"):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            MODEL_TRAINER_DIR_NAME,
        )

        # model.yaml path 
        self.model_config_file_path: str = MODEL_CONFIG_FILE_PATH

        # outputs
        self.report_file_path: str = os.path.join(
            self.model_trainer_dir,
            MODEL_TRAINER_REPORT_FILE_NAME,
        )
        self.best_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            MODEL_TRAINER_BEST_MODEL_FILE_NAME,
        )
        self.last_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            MODEL_TRAINER_LAST_MODEL_FILE_NAME,
        )
