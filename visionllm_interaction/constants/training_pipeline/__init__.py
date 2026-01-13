"""
Constants for VisionLLM Interaction Analysis training pipeline
"""

# ==================================================
# GLOBAL PIPELINE CONSTANTS
# ==================================================

DATASET_NAME: str = "awsaf49/coco-2017-dataset"

ARTIFACTS_DIR: str = "artifacts"
TRAINING_PIPELINE_NAME: str = "visionllm_interaction_pipeline"


# ==================================================
# DATA INGESTION STAGE
# ==================================================

DATA_INGESTION_DIR_NAME: str = "data_ingestion"

# Ingestion strategy (e.g. register | download | local)
DATA_INGESTION_MODE: str = "register"


# ==================================================
# ACTIVE DATA STAGING (SCRATCH)
# ==================================================


SCRATCH_ROOT_DIR: str = "/scratch/users/arunps"

PROJECT_DATA_ROOT: str = (
    f"{SCRATCH_ROOT_DIR}/visionllm_interaction/data"
)


# ==================================================
# RAW COCO DATASET (STAGED ON SCRATCH)
# ==================================================

RAW_DATA_DIR: str = f"{PROJECT_DATA_ROOT}/raw"

# COCO image directories
RAW_COCO_TRAIN_IMAGE_DIR: str = f"{RAW_DATA_DIR}/train2017"
RAW_COCO_VAL_IMAGE_DIR: str = f"{RAW_DATA_DIR}/val2017"

# COCO annotation files
RAW_COCO_ANNOTATION_DIR: str = f"{RAW_DATA_DIR}/annotations"

RAW_COCO_TRAIN_ANN_FILE: str = (
    f"{RAW_COCO_ANNOTATION_DIR}/instances_train2017.json"
)

RAW_COCO_VAL_ANN_FILE: str = (
    f"{RAW_COCO_ANNOTATION_DIR}/instances_val2017.json"
)


# ==================================================
# DATA INGESTION ARTIFACTS
# ==================================================

DATA_INGESTION_ARTIFACT_DIR: str = (
    f"{ARTIFACTS_DIR}/{DATA_INGESTION_DIR_NAME}"
)

# Manifest file created by data ingestion stage
DATA_INGESTION_MANIFEST_FILE: str = "data_manifest.yaml"

# ==================================================
# DATA VALIDATION STAGE
# ==================================================

DATA_VALIDATION_DIR_NAME: str = "data_validation"

# Schema file to validate the dataset
SCHEMA_FILE_PATH: str = "config/schema.yaml"

# Report file created by validation stage
DATA_VALIDATION_REPORT_FILE: str = "data_validation_report.yaml"

