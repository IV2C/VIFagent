from dataclasses import dataclass
import dataclasses
from typing import List

import numpy as np

type Box2D = tuple[float, float, float, float]
type Span = tuple[int, int]


# class BoxDetections(BaseModel):
#     boxes:List[int]

# class BoxDetection(BaseModel):
#     box_2d: Annotated[List[int],Field(min_items=4, max_items=4)]
#     label: str
    

# class Features(BaseModel):
#     image_description:Annotated[str,Field(description="detailed description of the image")]
#     features: Annotated[list[str],Field(description="Name of each individual feature")]
    
@dataclasses.dataclass(frozen=True)
class SegmentationMask:
    # bounding box pixel coordinates (not normalized)
    y0: int  # in [0..height - 1]
    x0: int  # in [0..width - 1]
    y1: int  # in [0..height - 1]
    x1: int  # in [0..width - 1]
    mask: np.array  # [img_height, img_width] with values 0..255
    label: str