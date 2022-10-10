from typing import Iterable
import networkx as nx
import numpy as np
import pandas as pd

from bokeh.io import curdoc, show
from bokeh.models import ColumnDataSource, Grid, ImageRGBA as bok_Image, LinearAxis, Plot, Range1d
from bokeh.plotting import figure, Figure

from .models import IRT_Image, IRT_Node


class ImageRelTagger():
    def __init__(self, images: Iterable[IRT_Image], graph: nx.Graph, wh_reserved_per_img=None) -> None:
        """
        Main class. Init, then display, and tag.
        :param images: Usually a list of IRT_images wehre the name is a root in the graph
        :param graph: An nx.Graph where some main nodes identify the names of the (IRT) images.
                      Other graph nodes represent pixels inseide each image.
        :param wh_reserved_per_img: Tuple of how many pixels to reserve per each image.
                                    If None, it will default to the biggest image from the input.
        """
        self.images = images
        self.graph = graph
        if wh_reserved_per_img is None:
            self.img_wh = np.max(np.array([img.image.shape for img in self.images]), axis=0)[:2].astype(int)[::-1]
            self.images_scales = [1.0 for _ in self.images]
        else:
            self.img_wh = wh_reserved_per_img
            self.images_scales = [self.__calculate_scale(img.image) for img in self.images]
        
    def display(self):
        plot = figure(plot_height=self.img_wh[1]*len(self.images), plot_width=self.img_wh[0])
        plot.x_range.range_padding = plot.y_range.range_padding = 0
        # add nodes
        self.__display_images(plot)
        source = self.__extract_graph_data(plot)
        plot.circle(x="x", y="y", size=20, source=source)
        # creates output file
        curdoc().add_root(plot)
        curdoc().theme = 'caliber'
        # showing the plot on output file
        show(plot)
        
    # -- PRIVATE
    
    def __get_all_succesors_of_img(self, img, from_node):
        all_successors_of_img = [node for node in self.graph.successors(from_node) if node.image_name == img.name]
        all_successors_of_successors = [sub_node for node in all_successors_of_img for sub_node in self.__get_all_succesors_of_img(img, node)]
        return all_successors_of_img + all_successors_of_successors
    
    def __extract_graph_data(self, plot):
        """
        Based on the iteration of the provided IRT_Images, add the rest of the graph nodes.
        Each node is in reference of the Image it appears in.
        For display, it will be converted to the plot's ref frame.
        """
        nodes = []
        for indx, img in enumerate(self.images):
            for node in self.__get_all_succesors_of_img(img, img.name):
                nodes.append([node.name, node.x*self.images_scales[indx], self.img_wh[1]*indx + node.y*self.images_scales[indx]])
        source = ColumnDataSource(pd.DataFrame(nodes, columns=["node_name", "x", "y"]))
        return source

    def __display_images(self, plot):
        for indx, img in enumerate(self.images):
            img_hw = img.image.shape[:2]
            rgba_img = self.__convert_rgb_to_rgba(img.image)
            plot.image_rgba(image=[rgba_img],
                            x=0,
                            y=self.img_wh[1]*indx,
                            dw=img_hw[1]*self.images_scales[indx],
                            dh=img_hw[0]*self.images_scales[indx])

    def __convert_rgb_to_rgba(self, rgb_img):
        M, N, _ = rgb_img.shape
        img = np.empty((M, N), dtype = np.uint32)
        view = img.view(dtype = np.uint8).reshape((M, N, 4))
        view[: ,: , 0:3] = rgb_img[: ,: , 0:3] # copy red channel
        view[: ,: , 3] = 255
        img = img[::-1]
        return img
    
    def __calculate_scale(self, rgb_img):
        h, w, _ = rgb_img.shape
        targ_w, targ_h = self.img_wh
        total_scale = 1.0
        while h > targ_h or w > targ_w:
            if w > targ_w:
                local_scale = targ_w / w
                total_scale *= local_scale
                w *= local_scale
                h *= local_scale
            if h > targ_h:
                local_scale = targ_h / h
                total_scale *= local_scale
                w *= local_scale
                h *= local_scale
        return total_scale