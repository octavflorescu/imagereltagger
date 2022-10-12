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

show_image_nodes = True

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
    nodes_dict = {}
    
    for img in images:
        img_shape = img.image.shape[:2]
        nodes = []
        if show_image_nodes:
            img_node = IRT_Node(img.name, img.name, xy=(0,0))
            nodes.append(img_node)
        for i in range(random.randint(3, 10)):
            coors = gen_2d_coordinates(0, img_shape[1]-1, 0, img_shape[0]-1)
            nodes.append(IRT_Node(f"{img.name}_{i}", img.name, xy=coors))
        
        nodes_dict.update({node.name: node for node in nodes})
        if show_image_nodes: nodes.remove(img_node)
        # order nodes
        parent_nodes = [img]
        while nodes:
            successor_nodes = random.sample(nodes, random.randint(1, len(nodes)))
            for node in successor_nodes:
                nodes.remove(node)
            for node in successor_nodes:
                graph.add_edge(parent_nodes[random.randint(0, len(parent_nodes) - 1)].name,
                               node.name)
            parent_nodes = successor_nodes
        
    irt = ImageRelTagger(images=images, nodes=nodes_dict, graph=graph,
                         wh_reserved_per_img=(600,400),
                         show_image_nodes=show_image_nodes,
                         display_on_h=True)
    irt.display()


if __name__ == '__main__':
    display_some_images_w_relations()
