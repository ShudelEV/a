import os

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ONE_OBJECT_PREDICT_MODEL_PATH: str = os.getenv(
    "ONE_OBJECT_PREDICT_MODEL_PATH",
    os.path.join(BASE_DIR, "temp", "vggnet.model")
)
ONE_OBJECT_LABEL_BINARIZER_PATH: str = os.getenv(
    "ONE_OBJECT_LABEL_BINARIZER_PATH",
    os.path.join(BASE_DIR, "temp", "vggnet_lb.pickle")
)