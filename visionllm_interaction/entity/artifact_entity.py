from dataclasses import dataclass


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


from dataclasses import dataclass


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
