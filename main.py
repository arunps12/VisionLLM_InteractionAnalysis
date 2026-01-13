from visionllm_interaction.logger.logger import get_logger
from visionllm_interaction.exception.custom_exception import CustomException
from visionllm_interaction.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
)
from visionllm_interaction.components.data_ingestion import DataIngestion
from visionllm_interaction.components.data_validation import DataValidation
from visionllm_interaction.components.data_transformation import DataTransformation

logger = get_logger(__name__)


def main():
    """
    Entry point for VisionLLM Interaction Analysis pipeline.

    Stages:
      1) Data Ingestion (COCO train/val staging + manifest)
      2) Data Validation (COCO checks + validation report)
      3) Data Transformation (clean annotations + training manifest)
    """
    try:
        logger.info("=== VisionLLM Interaction Analysis: Pipeline Started ===")

        # ------------------------------------------------------------
        # 1) Build training pipeline config (timestamped artifacts)
        # ------------------------------------------------------------
        training_pipeline_config = TrainingPipelineConfig()
        logger.info(f"Run artifacts directory: {training_pipeline_config.artifact_dir}")

        # ------------------------------------------------------------
        # 2) Data Ingestion
        # ------------------------------------------------------------
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)

        logger.info("Starting data ingestion stage...")
        logger.info(f"Dataset: {data_ingestion_config.dataset_name}")
        logger.info(f"Ingestion mode: {data_ingestion_config.ingestion_mode}")
        logger.info(f"Dataset format: {data_ingestion_config.dataset_format}")

        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        logger.info("Data ingestion stage completed successfully.")
        logger.info(f"Manifest written to: {data_ingestion_artifact.manifest_file_path}")
        logger.info(f"DataIngestionArtifact: {data_ingestion_artifact}")

        # ------------------------------------------------------------
        # 3) Data Validation
        # ------------------------------------------------------------
        data_validation_config = DataValidationConfig(training_pipeline_config)

        logger.info("Starting data validation stage...")
        logger.info(f"Schema file: {data_validation_config.schema_file_path}")
        logger.info(f"Require val: {data_validation_config.require_val}")

        data_validation = DataValidation(
            data_validation_config=data_validation_config,
            data_ingestion_artifact=data_ingestion_artifact,
        )
        data_validation_artifact = data_validation.initiate_data_validation()

        logger.info("Data validation stage completed successfully.")
        logger.info(f"Validation report written to: {data_validation_artifact.report_file_path}")
        logger.info(f"DataValidationArtifact: {data_validation_artifact}")

        # ------------------------------------------------------------
        # 4) Data Transformation (clean annotations -> training manifest)
        # ------------------------------------------------------------
        data_transformation_config = DataTransformationConfig(training_pipeline_config)

        logger.info("Starting data transformation stage...")
        logger.info(f"Transformation dir: {data_transformation_config.data_transformation_dir}")

        data_transformation = DataTransformation(
            data_transformation_config=data_transformation_config,
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_artifact=data_validation_artifact,
        )
        data_transformation_artifact = data_transformation.initiate_data_transformation()

        logger.info("Data transformation stage completed successfully.")
        logger.info(f"Cleaned train ann: {data_transformation_artifact.cleaned_train_annotation_file}")
        logger.info(f"Cleaned val ann: {data_transformation_artifact.cleaned_val_annotation_file}")
        logger.info(f"Training manifest written to: {data_transformation_artifact.training_manifest_file_path}")
        logger.info(f"DataTransformationArtifact: {data_transformation_artifact}")

        logger.info("=== VisionLLM Interaction Analysis: Pipeline Finished ===")

    except CustomException as ce:
        logger.error(f"Pipeline failed with CustomException: {ce}")
        raise

    except Exception as e:
        logger.error("Pipeline failed with an unexpected exception.")
        raise CustomException("Unexpected error in main pipeline execution", e)


if __name__ == "__main__":
    main()
