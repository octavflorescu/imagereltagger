import numpy as np


class IRT_Image:
    def __init__(self, image: np.array, name: str) -> None:
        self.image = image
        self.name = name

class IRT_Node:
    def __init__(self, name: str, image_name: str, xy=(0,0)) -> None:
        self.name = name
        self.image_name = image_name
        self.x, self.y = xy
