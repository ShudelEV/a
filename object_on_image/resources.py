import json
import logging
import mimetypes
from typing import IO, Any

from . import __name__ as project_name
from .controllers import Predictor
from .prediction_models import one_obj_model, one_obj_labels

from tempfile import NamedTemporaryFile


class __Base:

    def __init__(self) -> None:
        self.logger = logging.getLogger(
            f"{project_name}.{self.__class__.__name__.lower()}"
        )


class OneObjectOnImage(__Base):

    _CHUNK_SIZE_BYTES = 4096

    def on_post(self, request, response) -> None:
        predictor = Predictor(one_obj_model, one_obj_labels)
        ext = mimetypes.guess_extension(request.content_type)
        image_file: IO[Any]
        with NamedTemporaryFile(suffix=ext) as image_file:
            while True:
                chunk = request.stream.read(self._CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                image_file.write(chunk)

            label, probability = predictor.predict(image_file.name)

        data = {
            "label": label,
            "probability": probability,
        }
        response.body = json.dumps(data, ensure_ascii=False)


check_one_object = OneObjectOnImage()


__all__ = ("check_one_object",)