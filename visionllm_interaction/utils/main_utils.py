import os
from typing import Any, Dict

import yaml

from visionllm_interaction.logger.logger import get_logger
from visionllm_interaction.exception.custom_exception import CustomException

logger = get_logger(__name__)


def read_yaml(file_path: str) -> Dict[str, Any]:
    """
    Read a YAML file and return its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed YAML content.

    Raises:
        CustomException: If file does not exist or parsing fails.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)

        if content is None:
            raise ValueError(f"YAML file is empty: {file_path}")

        logger.info(f"YAML file read successfully: {file_path}")
        return content

    except Exception as e:
        raise CustomException(f"Failed to read YAML file: {file_path}", e)


def write_yaml(file_path: str, data: Dict[str, Any]) -> None:
    """
    Write a dictionary to a YAML file.

    Args:
        file_path (str): Path where YAML will be written.
        data (Dict[str, Any]): Data to write.

    Raises:
        CustomException: If writing fails.
    """
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as yaml_file:
            yaml.safe_dump(
                data,
                yaml_file,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )

        logger.info(f"YAML file written successfully: {file_path}")

    except Exception as e:
        raise CustomException(f"Failed to write YAML file: {file_path}", e)
