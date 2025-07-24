 
from pathlib import Path
from box import ConfigBox
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.constants import *
from cnnClassifier.entity.config_entity import (
    EvaluationConfig, DataIngestionConfig, PrepareBaseModelConfig, 
    PrepareCallbacksConfig, TrainingConfig
)
import os


class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):

        # Always resolve to the project root (parent of "src")
        self.ROOT_DIR = Path(__file__).resolve().parent.parent.parent

        # Build paths for YAML configs
        config_path = (self.ROOT_DIR / config_filepath).resolve()
        params_path = (self.ROOT_DIR / params_filepath).resolve()

        # Load YAML files
        self.config = ConfigBox(read_yaml(config_path))
        self.params = ConfigBox(read_yaml(params_path))

        # Ensure artifacts folder exists
        create_directories([self.config.artifacts_root])

    def get_validation_config(self) -> EvaluationConfig:
        return EvaluationConfig(
            path_of_model=self.ROOT_DIR / "artifacts/training/model.h5",
            training_data=self.ROOT_DIR / "artifacts/data_ingestion/Chicken_disease_images",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([self.ROOT_DIR / config.root_dir])

        return DataIngestionConfig(
            root_dir=self.ROOT_DIR / config.root_dir,
            source_URL=config.source_URL,
            local_data_file=self.ROOT_DIR / config.local_data_file,
            unzip_dir=self.ROOT_DIR / config.unzip_dir
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([self.ROOT_DIR / config.root_dir])

        return PrepareBaseModelConfig(
            root_dir=self.ROOT_DIR / config.root_dir,
            base_model_path=self.ROOT_DIR / config.base_model_path,
            updated_base_model_path=self.ROOT_DIR / config.updated_base_model_path,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = self.ROOT_DIR / os.path.dirname(config.checkpoint_model_filepath)

        create_directories([
            model_ckpt_dir,
            self.ROOT_DIR / config.tensorboard_root_log_dir
        ])

        return PrepareCallbacksConfig(
            root_dir=self.ROOT_DIR / config.root_dir,
            tensorboard_root_log_dir=self.ROOT_DIR / config.tensorboard_root_log_dir,
            checkpoint_model_filepath=self.ROOT_DIR / config.checkpoint_model_filepath
        )

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params

        training_data = self.ROOT_DIR / self.config.data_ingestion.unzip_dir / "Chicken_disease_images"
        create_directories([self.ROOT_DIR / training.root_dir])

        return TrainingConfig(
            root_dir=self.ROOT_DIR / training.root_dir,
            trained_model_path=self.ROOT_DIR / training.trained_model_path,
            updated_base_model_path=self.ROOT_DIR / prepare_base_model.updated_base_model_path,
            training_data=training_data,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )
