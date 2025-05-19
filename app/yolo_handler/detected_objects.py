from typing import List, Any, Optional, Tuple
import numpy as np

from .bounding_box import BoundingBox

yolo_classnames = ('anitem', 'anotheritem')


class DetectedObject:
    """Class to store License plate character"""

    def __init__(self, cls: int, conf: float, bbox: BoundingBox, classnames: Tuple[str] = yolo_classnames) -> None:
        self.cls: int = cls
        self.object_class: Optional[str] = classnames[cls]
        self.conf: float = conf
        self.bbox: BoundingBox = bbox

    def get_annotation_for_bbox(self):
        """Returns all attributes required for annotation"""
        return self.bbox.get_coordinates(), self.cls, self.conf

    @staticmethod
    def rescale_prediction(xyxy, dwdh, ratio) -> List[Any]:
        """Rescale prediction back to original image size.

        Args:
            image: image whose characters need to be rescaled

        Returns:
            Rescaled image predictions
        """
        xyxy = np.array(xyxy)
        xyxy -= np.array(dwdh * 2)
        xyxy /= ratio

        return list(xyxy)
