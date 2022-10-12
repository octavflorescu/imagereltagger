from typing import Iterable, Dict
import networkx as nx
import numpy as np
import pandas as pd

from bokeh.io import curdoc, show
from bokeh.palettes import Spectral4
from bokeh.models import ColumnDataSource, CustomJS, GraphRenderer, Circle, MultiLine, \
    EdgesAndLinkedNodes, NodesAndLinkedEdges, \
    TapTool, WheelZoomTool, HoverTool, PanTool,\
    Grid, ImageRGBA as bok_Image, LinearAxis, Plot, Range1d
from bokeh.plotting import figure, Figure, from_networkx

from .models import IRT_Image, IRT_Node


class ImageRelTagger():
    def __init__(self, images: Iterable[IRT_Image], nodes: Dict[str, IRT_Node] , graph: nx.Graph,
                 wh_reserved_per_img=None, show_image_nodes=False, display_on_h=False) -> None:
        """
        Main class. Init, then display, and tag.
        :param images: Usually a list of IRT_images wehre the name is a root in the graph
        :param nodes: Used to display the nodes at desired locations
        :param graph: An nx.Graph where some main nodes identify the names of the (IRT) images.
                      Other graph nodes represent pixels inseide each image.
        :param wh_reserved_per_img: Tuple of how many pixels to reserve per each image.
                                    If None, it will default to the biggest image from the input.
        """
        self.images = images
        self.nodes = nodes
        self.graph = graph
        self.show_image_nodes = show_image_nodes
        self.display_on_h = display_on_h
        
        if wh_reserved_per_img is None:
            self.max_img_wh = np.max(np.array([img.image.shape for img in self.images]), axis=0)[:2].astype(int)[::-1]
            self.images_scales = [1.0 for _ in self.images]
        else:
            self.max_img_wh = np.array(wh_reserved_per_img)
            self.images_scales = [self.__calculate_scale(img.image) for img in self.images]
        
    def display(self):
        N = len(self.images)
        plot = figure(plot_height=self.max_img_wh[1]*(1 if self.display_on_h else N),
                      plot_width=self.max_img_wh[0]*(N if self.display_on_h else 1),
                      tools=[HoverTool(tooltips=None), TapTool(), PanTool(), WheelZoomTool(zoom_on_axis=False)])
        plot.toolbar.active_scroll = plot.select_one(WheelZoomTool)
        plot.x_range.range_padding = plot.y_range.range_padding = 0
        
        # display images
        self.__display_images(plot)
        
        # display graph(s) on images
        self.__scale_nodes_data()
        layout_of_nodes = {k: (node.x, node.y) for k,node in self.nodes.items()}
        g = nx.DiGraph(self.graph)
        g.remove_nodes_from([] if self.show_image_nodes else [img.name for img in self.images])
        graph_renderer = from_networkx(g, layout_of_nodes, scale=2, center=(0,0))
        graph_renderer.node_renderer.glyph = Circle(size=20, fill_color=Spectral4[0])
        graph_renderer.node_renderer.selection_glyph = Circle(size=20, fill_color=Spectral4[2])
        graph_renderer.node_renderer.hover_glyph = Circle(size=20, fill_color=Spectral4[1])
        graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)
        graph_renderer.selection_policy = NodesAndLinkedEdges()
        graph_renderer.inspection_policy = EdgesAndLinkedNodes()
        
        plot.renderers.append(graph_renderer)
        
        # creates output file
        curdoc().add_root(plot)
        curdoc().theme = 'caliber'
        # showing the plot on output file
        show(plot)
        
    # -- PRIVATE
    
    def __get_all_succesors_of_img(self, img, from_node):
        all_successors_of_img = [node_name
                                 for node_name in self.graph.successors(from_node)
                                 if self.nodes[node_name].image_name == img.name]
        all_successors_of_successors = [sub_node_name
                                        for node_name in all_successors_of_img
                                        for sub_node_name in self.__get_all_succesors_of_img(img, node_name)]
        return all_successors_of_img + all_successors_of_successors

    def __scale_nodes_data(self):
        """
        Based on the iteration of the provided IRT_Images, add the rest of the graph nodes.
        Each node is in reference of the Image it appears in.
        For display, it will be converted to the plot's ref frame.
        """
        def scale_single_node_coors(node_data, at_img_indx):
            x_shift = self.max_img_wh[0] * at_img_indx if self.display_on_h else 0
            node_data.x = x_shift + node_data.x * self.images_scales[at_img_indx]
            y_shift = 0 if self.display_on_h else self.max_img_wh[1] * at_img_indx
            node_data.y = y_shift + node_data.y * self.images_scales[at_img_indx]
        for indx, img in enumerate(self.images):
            scale_single_node_coors(self.nodes[img.name], indx)
            for node_name in self.__get_all_succesors_of_img(img, img.name):
                scale_single_node_coors(self.nodes[node_name], indx)


    def __display_images(self, plot):
        for indx, img in enumerate(self.images):
            img_hw = img.image.shape[:2]
            rgba_img = self.__convert_rgb_to_rgba(img.image)
            plot.image_rgba(image=[rgba_img],
                            x=self.max_img_wh[0]*indx if self.display_on_h else 0,
                            y=0 if self.display_on_h else self.max_img_wh[1]*indx,
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
        targ_w, targ_h = self.max_img_wh
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
