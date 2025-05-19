"""Batch data class."""
import glob
import os
import uuid
from typing import Dict, List, Union, Tuple, Iterable, Any

import cv2
import logging
import numpy as np
from pydantic import BaseModel
from uuid import UUID

from .bounding_box import BoundingBox
from .detected_objects import DetectedObject
from .image import Image

logger = logging.getLogger(__name__)


class Images(BaseModel):
    """List of images class."""
    images: List[Image] = []

    def __len__(self) -> int:
        """Obtain the amount of images.

        Returns:
            Number of images
        """
        return len(self.images)

    @staticmethod
    def read_from_folder(path: str, ext: str = ".jpg") -> List[Image]:
        """Reads images from input folder and initializes the input data.

        Args:
            path: List of images paths
            ext: extension of the image files to be read -> .jpg, .png etc.

        Returns:
            List of image objects
        """
        logger.info("Loading image data from disk...")
        image_paths = glob.glob(f"{path}/*{ext}")
        return [Image(
            id=os.path.basename(path),
            np_data=cv2.imread(path),
        ) for path in image_paths]

    @staticmethod
    def read_from_upload_file(filename: str, image_data: bytes) -> List[Image]:
        """Reads images from UploadFile.

        Args:
            file: an UploadFile object.

        Returns:
            List of image objects with length of 1
        """
        img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return [Image(
            id=filename,
            np_data=img
        )]

    @staticmethod
    def np_init(data: List[np.ndarray], ids: List[Union[str, UUID]] = None) -> List[Image]:
        """Init class from numpy array.

        Args:
            data: List of images as numpy arrays
            ids: Corresponding ids if present, otherwise uuids are generated

        Returns:
            List of image objects
        """
        logger.info("Loading image data from numpy array...")
        return [Image(
            id=ids[idx] if ids is not None else uuid.uuid1(),
            np_data=x,
        ) for idx, x in enumerate(data)]

    def to_onnx_input(self, image_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Get numpy batch data.

        Args:
            image_size: Model input size

        Returns:
            Images in numpy batch format.
        """
        return np.concatenate(self.format_to_onnx(image_size=image_size))

    def format_to_onnx(self, image_size: Tuple[int, int]) -> np.ndarray:
        """Format data to onnx input requirements.

        Args:
            image_size: Model input size

        Returns:
            Formatted data
        """
        np_batch = []

        for image_data in self.letterbox(new_shape=image_size, auto=False):
            img = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img / 255)
            np_batch.append(img)

        return np_batch

    def get_image_data(self) -> List[np.ndarray]:
        """Get image data.

        Returns:
            List of images data.
        """
        return [i.np_data for i in self.images]

    def get_image_ids(self) -> List[str]:
        """Get image id.

        Returns:
            List of images ids.
        """
        return [i.id for i in self.images]

    def get_plots(self) -> List[Any]:
        """Get plot data.

        Returns:
            List of plots.
        """
        return [i.plot_data for i in self.images]

    def letterbox(self, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32) -> List[np.ndarray]:
        """Resize image by applying letterbox for each image.

        Args:
            new_shape: New dimension of images
            color: Letterbox background color
            auto: Minimum rectangle
            scaleup: Only scale down, do not scale up (for better val mAP)
            stride: Stride for scaleup
        """
        np_batch = []
        for image in self.images:
            resized_img = image.letterbox(new_shape=new_shape, color=color, auto=auto, scaleup=scaleup, stride=stride)
            np_batch.append(resized_img)

        return np_batch

    def create_batch(self, batch_size: int = 1) -> Iterable:
        """Create batch of iterable.

        Args:
            batch_size: the batch size

        Returns:
            a batch of size n out of iterable (in order)
        """
        length = len(self.images)
        for ndx in range(0, length, batch_size):
            logger.info("debug")
            yield Images(images=self.images[ndx: min(ndx + batch_size, length)])

    def display(self) -> None:
        """Displays images
        """
        # Show annotations
        for i, image in enumerate(self.get_images()):
            cv2.imshow(f"Image: {i}", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def init_detected_objects(self, predictions: np.ndarray) -> None:
        """Uses the predictions to generate a list of object.
        Args:
            predictions: predictions from the ocr model as np ndarray

        Returns:
            List
        """
        total_objects = []
        batch_ids = np.unique(predictions[:, 0])
        print('********** init_detected_objects **********')
        print(batch_ids)
        for batch_id in batch_ids:
            print(batch_id)
            objects_per_image = []
            batch_id = int(batch_id)
            predictions_for_batch_id = [i for i in predictions if i[0] == batch_id]
            for object_prediction in predictions_for_batch_id:
                bs, x1, y1, x2, y2, cls, conf = object_prediction
                objects_per_image.append(DetectedObject(cls=int(cls), conf=conf, bbox=BoundingBox(position=[x1, y1, x2, y2])))
                self.images[batch_id].objects = objects_per_image
