from typing import Tuple

import cv2


class Predictor:

    def __init__(self, predict_model, labels) -> None:
        self.predict_model = predict_model
        self.labels = labels

    def predict(
        self, image_path, width=28, height=28, flatten=False
    ) -> Tuple[str, float]:
        """
        Make prediction for an object on the given image.

        :param image_path: path to input image we are going to classify
        :param width: target spatial dimension width
        :param height: target spatial dimension height
        :param flatten: whether or not we should flatten the image
        :return: <label>, <probability value>
        """
        image = cv2.imread(image_path)
        image = cv2.resize(image, (width, height))
        image = image.astype("float") / 255.0
        if flatten:
            image = image.flatten()
            image = image.reshape((1, image.shape[0]))
        else:
            image = image.reshape(
                (1, image.shape[0], image.shape[1], image.shape[2])
            )

        predictions = self.predict_model.predict(image)
        index: int = predictions.argmax(axis=1)[0]
        label: str = self.labels.classes_[index]
        probability: float = predictions[0][index]

        return label, probability