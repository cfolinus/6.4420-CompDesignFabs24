# An example script to use your DSL and compile to an SVG

from tab import Tab, generate_root_tab, generate_child_tab, draw_svg
from math import pi

edge_length = 50
tab_angle = 0
bend_angle = pi/2

root = generate_root_tab(edge_length, edge_length, tab_angle)

child0 = generate_child_tab(root,
                            edge_length,
                            edge_length,
                            tab_angle,
                            bend_angle,
                            0, 0)
child1 = generate_child_tab(root,
                            edge_length,
                            edge_length,
                            tab_angle,
                            bend_angle,
                            1, 0)
child2 = generate_child_tab(root,
                            edge_length,
                            edge_length,
                            tab_angle,
                            bend_angle,
                            2, 0)
child3 = generate_child_tab(root,
                            edge_length,
                            edge_length,
                            tab_angle,
                            bend_angle,
                            3, 0)
child4 = generate_child_tab(child0,
                            edge_length,
                            edge_length,
                            tab_angle,
                            bend_angle,
                            1 , 0)
 
draw_svg(root, "cube.svg") 