import json
import logging
import mimetypes
from typing import IO, Any, Dict

from . import __name__ as project_name
from .controllers import Predictor
from .prediction_models import one_obj_model, one_obj_labels

from tempfile import NamedTemporaryFile


class __Base:
    """Base class for a views."""

    def __init__(self) -> None:
        """Init."""
        self.logger = logging.getLogger(
            f"{project_name}.{self.__class__.__name__.lower()}"
        )


class OneObjectOnImage(__Base):

    _CHUNK_SIZE_BYTES = 4096

    def on_post(self, request, response) -> None:
        """Handle post request."""
        predictor = Predictor(one_obj_model, one_obj_labels)
        # Read image by chunks and save as temporary file
        ext = mimetypes.guess_extension(request.content_type)
        image_file: IO[Any]
        with NamedTemporaryFile(suffix=ext) as image_file:
            while True:
                chunk = request.stream.read(self._CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                image_file.write(chunk)
            # Make prediction
            data: Dict[str, float] = predictor.predict(image_file.name)

        response.body = json.dumps(data, ensure_ascii=False)


check_one_object = OneObjectOnImage()


__all__ = ("check_one_object",)
