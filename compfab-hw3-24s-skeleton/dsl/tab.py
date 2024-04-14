import numpy as np
from typing import Optional, Union
from dataclasses import dataclass
from functools import cache
import svgwrite
from svgwrite.shapes import Polygon
from pathlib import Path
from math import pi, tan, sin, cos

# TODO: SVGWRITE USES IMAGES COORDS, I USE TRADITIONAL CARTESIAN COORDINATES

@dataclass
class Tab:
    """
    A structure that represents a tab and a bend with respect to the parent tab.

    Hint: See figure 2 on some guidance to what parameters need to be put here.
    """

    parent: Optional["Tab"]
    children: list["Tab"]
    # TODO 3.2: Add attributes as needed.
    
    def __init__(self, tab_width, tab_length, tab_angle = 0, 
                 bend_angle = None, parent_edge_number = None, parent_offset = None):
         self.tab_width = tab_width
         self.tab_length = tab_length
         self.tab_angle = tab_angle
         
         
         self.bend_angle = bend_angle
         self.parent_edge_number = parent_edge_number
         self.parent_offset = parent_offset;
         
         # Initialize parents and children
         self.parent = None
         self.children = []

    def __hash__(self):
        return id(self)
   
    
    def is_child_in_tab(self, child):
         if child in self.children:
            return True
  
    def remove_child(self, child_to_remove):
        index_to_remove = self.children.index(child_to_remove)
        return self.children.pop(index_to_remove)
       
    def add_child (self, child_to_add):
         # If this new tab is not already a child, add it
         if not(self.is_child_in_tab(child_to_add)):
              self.children.append(child_to_add)
              
         # If parent of the new tab is NOT our current tab, 
         # make the new tab's parent the current tab
         if not child_to_add.parent == self:  # Prevents infinite loop
              child_to_add.set_parent(self)
              
    def set_parent(self, new_parent_tab):
        
        # If this tab already has a parent, remove the old parent (clean the tree)
        if self.parent is not None:
            if self.parent.is_child_in_tab(self):
                self.parent.remove_child(self)

        # Set this new tab as the parent of our current tab
        self.parent = new_parent_tab

        # Add the current tab as a child of the new parent tab
        if isinstance(new_parent_tab, Tab):
            new_parent_tab.add_child(self)

          

    @cache
    def compute_corner_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the four corner points in 2D (2,) based on the attributes.

        Hint: You may want to specify the convention on how you order these points.
        Hint: You can call this function on the parent to help get started.
        """
        # TODO 3.2: Implement this function
        
        # Compute corner points within the local tab coordinates
        # local_corner_points in [X; Y] with [[x1, x2, x3, x4], [y1, y2, y3, y4]]
        # Where points are numbered clockwise starting at the upper left corner        
        # local_corner_points = np.array([[0,
        #                                 self.tab_length * tan(self.tab_angle),
        #                                 self.tab_width + self.tab_length * tan(self.tab_angle),
        #                                 self.tab_width],
        #                                 [0,
        #                                  - self.tab_length,
        #                                  - self.tab_length,
        #                                  0]])
        local_corner_points= np.array([[0, 0],
                                       [self.tab_length * tan(self.tab_angle), self.tab_length],
                                       [self.tab_width + self.tab_length * tan(self.tab_angle), self.tab_length], 
                                       [self.tab_width, 0]]).T
        
        
        if self.parent is not None:
             parent_corner_points = self.parent.compute_corner_points()
             parent_reference_edge_point = parent_corner_points[self.parent_edge_number].reshape([2,1])
             # print(parent_reference_edge_point)
             
             

             # translations = tuple(self.parent_offset \
             #                      * np.array([[sin(self.parent.tab_angle), cos(self.parent.tab_angle)], 
             #                       [1, 0],
             #                       [-sin(self.parent.tab_angle), -cos(self.parent.tab_angle)],
             #                       [-1, 0]]))
                  
             # Define rotation angle and translation corresponding to this edge
             # temp_rotation_angle = rotation_angles[self.parent_edge_number]
             # # temp_translation = translations[self.parent_edge_number].reshape([2,1])
                          
             # rotation_matrix = np.array([[cos(temp_rotation_angle), -sin(temp_rotation_angle)],
             #                            [sin(temp_rotation_angle), cos(temp_rotation_angle)]])
             
             rotation_matrix = self.get_rotation_matrix()

             # Apply rotation to curent corner points
             rotated_corner_points = np.matmul(rotation_matrix, local_corner_points)
             offset_vector = self.parent_offset * np.matmul(rotation_matrix, np.array([1,0])).reshape([2,1])

             # Translate rotated corner points to correct location
             translated_corner_points = parent_reference_edge_point \
                                           + offset_vector \
                                           + rotated_corner_points
             
             # Translate local corner points based on location of tab
              # parent_corner_points = self.parent.compute_corner_points();
             # parent_corner_edge_point = parent_corner_points[]
             
             # translated_corner_points 
             
             # Rotate corner points based on tab 2D orientation
             # initial_tab_point = parent_corner_edge_point + self.parent_offset \
             #      * np.array([[sin(self.parent.tab_angle)], [-cos(self.parent.tab_angle)]])
             
             corner_points = tuple(translated_corner_points[:, point_index] for point_index in range(4))
        else:
             corner_points = tuple(local_corner_points[:, point_index] for point_index in range(4))
   
        
        return corner_points
   
    def get_rotation_angle(self, rotation_angle = 0):
         

          
          # If our current tab is a child:
          if self.parent is not None:
               
               rotation_angles = np.array([0.5*pi - self.parent.tab_angle,
                                           0,
                                           1.5*pi - self.parent.tab_angle,
                                           pi])
               
               # Add the rotation angle for the current tab
               rotation_angle += rotation_angles[self.parent_edge_number]
               
               # Add the rotation angle for the parent tab(s)
               rotation_angle = self.parent.get_rotation_angle(rotation_angle)
          
          return rotation_angle
        
    def get_rotation_matrix(self):
         
         
          temp_rotation_angle = self.get_rotation_angle()

          rotation_matrix = np.array([[cos(temp_rotation_angle), -sin(temp_rotation_angle)],
                                     [sin(temp_rotation_angle), cos(temp_rotation_angle)]])
          
          return rotation_matrix

    def compute_all_corner_points(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Computes all four corner points of all tabs in the current subtree.
        """
        cps = [self.compute_corner_points()]
        for child in self.children:
            cps.extend(child.compute_all_corner_points())
        return cps


def generate_root_tab(tab_width, tab_length, tab_angle = pi/2) -> Tab:
    """
    Generate a new parent tab
    """
    # TODO: 3.2: Update the arguments and implement this function.
    root_tab = Tab(tab_width, tab_length, tab_angle)
    
    return root_tab

def generate_child_tab(parent: Tab, tab_width, tab_length, tab_angle, 
                       bend_angle, parent_edge_number, parent_offset) -> Tab:
    """
    Generate a child tab. Make sure to update the children of parent accordingly.
    """
    # TODO: 3.2: Update the arguments and implement this function.
    
    # Initialize the tab
    child_tab = Tab (tab_width, tab_length, tab_angle, bend_angle, parent_edge_number, parent_offset)
    
    # Update the parent tab with information about this tab
    parent.add_child(child_tab)
    
    # # Update the parent of this tab
    # child_tab.set_parent(parent)

    

    
    return child_tab


def draw_svg(root_tab: Tab, output: Union[str, Path], stroke_width: float = 1):
    cps = root_tab.compute_all_corner_points()
    points = np.array(cps).reshape(-1, 2)
    min_point = points.min(axis=0)  # (2,)
    max_point = points.max(axis=0)  # (2,)
    points -= min_point
    points += 2 * stroke_width
    size = max_point - min_point  # (2,)
    size += 4 * stroke_width
    rects = points.reshape(-1, 4, 2)

    dwg = svgwrite.Drawing(str(output), size=(size[0], size[1]), profile="tiny")

    for rect in rects:
        dwg.add(Polygon(rect, stroke="black", fill="lightgray", stroke_width=stroke_width))

    dwg.save()
