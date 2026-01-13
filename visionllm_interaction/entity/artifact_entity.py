from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class DataIngestionArtifact:
    """
    Artifact produced by the Data Ingestion stage.

    - Points to the ingestion manifest file created
    - Records dataset format and ingestion mode
    """

    data_ingestion_dir: str
    manifest_file_path: str
    ingestion_mode: str
    dataset_format: str  # e.g. "raw"



@dataclass
class DataValidationArtifact:
    """
    Artifact produced by Data Validation stage.

    - report_file_path: YAML report with stats + PASS/FAIL
    - validated: overall boolean status
    """

    data_validation_dir: str
    report_file_path: str
    validated: bool


@dataclass
class DataTransformationArtifact:
    """
    Artifact produced by Data Transformation stage.
    """

    data_transformation_dir: str
    cleaned_train_annotation_file: str
    cleaned_val_annotation_file: str
    training_manifest_file_path: str
    dropped_train_annotation_ids: int
    dropped_val_annotation_ids: int


@dataclass
class ModelTrainerArtifact:
    """
    Artifact produced by Model Trainer stage.
    """

    model_trainer_dir: str
    best_model_path: str
    last_model_path: str
    training_report_path: str

    # key results
    best_metric_name: str
    best_metric_value: float

    # HPO summary 
    hpo_enabled: bool
    best_params: Optional[Dict[str, Any]] = None
    n_trials: Optional[int] = None
