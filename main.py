import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from cnnClassifier import logger 
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_03_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline


# --- Patch for old models with 'fn' in CategoricalCrossentropy ---
import tensorflow as tf
import keras.models as standalone_keras_models
# type: ignore
import tensorflow.keras.models as tf_keras_models

# Store original load_model
_original_load_model_tf = tf_keras_models.load_model
_original_load_model_keras = standalone_keras_models.load_model

def safe_load_model(path, *args, **kwargs):
    """
    Wrapper around keras.models.load_model to handle older models that
    include the deprecated 'fn' argument in CategoricalCrossentropy.
    """
    def patched_categorical_crossentropy(*a, **k):
        k.pop('fn', None)  # Remove old 'fn' argument
        return tf.keras.losses.CategoricalCrossentropy(*a, **k)

    return _original_load_model_tf(path, custom_objects={
        'CategoricalCrossentropy': patched_categorical_crossentropy
    }, *args, **kwargs)

# Monkey-patch globally for both keras and tf.keras
tf_keras_models.load_model = safe_load_model
standalone_keras_models.load_model = safe_load_model
# ---------------------------------------------------------------

# Stage 01: Data Ingestion
STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Stage 02: Prepare Base Model
STAGE_NAME = "Prepare base model"
try: 
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Stage 03: Training
STAGE_NAME = "Training"
try: 
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Evaluation stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evaluation = EvaluationPipeline()
   model_evaluation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logger.exception(e)
        raise e
