import os
import networkx as nx
import numpy as np
from pathlib import Path
from PIL import Image
import random

import sys
import os
main_proj_path = Path(__file__).parents[1]
sys.path.append(str(main_proj_path))

from src.tagger import ImageRelTagger
from src.models import IRT_Image, IRT_Node

def get_img_at_path(path):
    im= Image.open(path)
    (x, y) = im.size
    I = np.array(im.getdata()).reshape(y, x, 3)
    return I


def gen_2d_coordinates(mx, Mx, my, My):
    return random.randint(mx, Mx), random.randint(my, My)


def display_some_images_w_relations():
    image1 = IRT_Image(image=get_img_at_path(main_proj_path / "resources" / "1.png"),
                       name="im1")
    image2 = IRT_Image(image=get_img_at_path(main_proj_path / "resources" / "2.png"),
                       name="im2")
    image3 = IRT_Image(image=get_img_at_path(main_proj_path / "resources" / "3.png"),
                       name="im3")
    
    images = [image1, image2, image3]
    graph = nx.DiGraph()
    
    for img in images:
        img_shape = img.image.shape[:2]
        nodes = []
        for i in range(random.randint(3, 10)):
            coors = gen_2d_coordinates(0, img_shape[1]-1, 0, img_shape[0]-1)
            nodes.append(IRT_Node(f"{img.name}_{i}", img.name, xy=coors))
        
        # order nodes
        prev_parents = [img.name]
        while nodes:
            successor_nodes = random.sample(nodes, random.randint(1, len(nodes)))
            for node in successor_nodes:
                nodes.remove(node)
            for node in successor_nodes:
                graph.add_edge(prev_parents[random.randint(0, len(prev_parents) - 1)],
                               node)
            prev_parents = successor_nodes
        
    irt = ImageRelTagger(images=images, graph=graph, wh_reserved_per_img=(600,400))
    irt.display()


if __name__ == '__main__':
    display_some_images_w_relations()
