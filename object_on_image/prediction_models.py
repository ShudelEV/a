import logging
import pickle

from . import settings

from keras.models import load_model

logger = logging.getLogger(__name__)


logger.info("Loading networks and label binarizers...")

one_obj_model = load_model(settings.ONE_OBJECT_PREDICT_MODEL_PATH)
one_obj_labels = pickle.loads(
    open(settings.ONE_OBJECT_LABEL_BINARIZER_PATH, "rb").read()
)
