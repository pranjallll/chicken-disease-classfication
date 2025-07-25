from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_callbacks import PrepareCallback
from cnnClassifier.components.training import Training
from cnnClassifier import logger

import tensorflow as tf
from tensorflow.keras import models as tf_keras_models
import keras.models as standalone_keras_models

# --- Fix for 'fn' argument issue in loss ---
_original_cce = tf.keras.losses.CategoricalCrossentropy

def patched_categorical_crossentropy(*args, **kwargs):
    kwargs.pop('fn', None)  # Strip out invalid arg
    return _original_cce(*args, **kwargs)

tf.keras.losses.CategoricalCrossentropy = patched_categorical_crossentropy
# -------------------------------------------------------

# --- Patch for loading .h5 models safely ---
_original_load_model_tf = tf_keras_models.load_model
_original_load_model_keras = standalone_keras_models.load_model

def safe_load_model(path, *args, **kwargs):
    return _original_load_model_tf(
        path,
        custom_objects={'CategoricalCrossentropy': patched_categorical_crossentropy},
        *args, **kwargs
    )

tf_keras_models.load_model = safe_load_model
standalone_keras_models.load_model = safe_load_model
# -------------------------------------------------------

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        
        # If you already have a model, load it here
        if training_config.trained_model_path.exists():
            logger.info(f"Loading existing model from {training_config.trained_model_path}")
            training.model = tf_keras_models.load_model(training_config.trained_model_path)
        else:
            logger.info("Building a new base model")
            training.get_base_model()
        
        training.train_valid_generator()
        training.train(callback_list=callback_list)

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
