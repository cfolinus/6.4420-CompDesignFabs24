# An example script to use your DSL and compile to an SVG

from tab import Tab, generate_root_tab, generate_child_tab, draw_svg
from math import pi, degrees, radians

tab_length = 50
tab_width = 100
tab_angle = pi/6
bend_angle = pi/6

root = generate_root_tab(tab_width, tab_length, tab_angle)

child_tab = generate_child_tab(root, 
                               0.5 * tab_width,
                               0.5 * tab_length, 
                               tab_angle = radians (20), 
                               bend_angle = bend_angle,
                               parent_edge_number = 0,
                               parent_offset = 2)
secondary_child_tab = generate_child_tab(child_tab,
                                         0.25 * tab_width,
                                         0.25 * tab_length, 
                                         tab_angle = radians (10), 
                                         bend_angle = bend_angle,
                                         parent_edge_number = 0,
                                         parent_offset = 2)


a = root.compute_corner_points()

b = child_tab.compute_corner_points()

# child1 = generate_child_tab(root, ...)
# child2 = generate_child_tab(root, ...)
# child3 = generate_child_tab(child1, ...)

draw_svg(root, "example.svg")