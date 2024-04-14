# An example script to use your DSL and compile to an SVG

from tab import Tab, generate_root_tab, generate_child_tab, draw_svg
from math import pi, radians

# edge_length = 50
# tab_angle = 0
# bend_angle = pi/2

tower_height = 100
tower_width = 50
tower_depth = 30

tower_front_gap = 10

support_tab_length = 10
support_tab_angle = radians(15)
support_tab_offset = 5

front_panel_width = 0.5 * (tower_width - tower_front_gap)

# bend_angle = radians(90)

back_panel = generate_root_tab(tab_width = tower_width,
                               tab_length = tower_height,
                               tab_angle = 0)

left_side_panel = generate_child_tab(back_panel,
                                      tab_width = tower_height,
                                      tab_length = tower_depth,
                                      tab_angle = 0, 
                                      bend_angle = pi/2,
                                      parent_edge_number = 0,
                                      parent_offset = 0)
left_front_panel = generate_child_tab(left_side_panel,
                                      tab_width = tower_height,
                                      tab_length = front_panel_width,
                                      tab_angle = 0, 
                                      bend_angle = pi/2,
                                      parent_edge_number = 1,
                                      parent_offset = 0)
left_front_tuck1 = generate_child_tab(left_front_panel,
                                      tab_width = tower_height,
                                      tab_length = tower_front_gap,
                                      tab_angle = 0, 
                                      bend_angle = pi/2,
                                      parent_edge_number = 1,
                                      parent_offset = 0)
left_front_tuck2 = generate_child_tab(left_front_tuck1,
                                      tab_width = tower_height,
                                      tab_length = 0.75 * front_panel_width,
                                      tab_angle = 0, 
                                      bend_angle = pi/2,
                                      parent_edge_number = 1,
                                      parent_offset = 0)

right_side_panel = generate_child_tab(back_panel,
                                     tab_width = tower_height,
                                     tab_length = tower_depth,
                                     tab_angle = 0, 
                                     bend_angle = pi/2,
                                     parent_edge_number = 2,
                                     parent_offset = 0)
right_front_panel = generate_child_tab(right_side_panel,
                                      tab_width = tower_height,
                                      tab_length = front_panel_width,
                                      tab_angle = 0, 
                                      bend_angle = pi/2,
                                      parent_edge_number = 1,
                                      parent_offset = 0)
right_front_tuck1 = generate_child_tab(right_front_panel,
                                      tab_width = tower_height,
                                      tab_length = tower_front_gap,
                                      tab_angle = 0, 
                                      bend_angle = pi/2,
                                      parent_edge_number = 1,
                                      parent_offset = 0)
right_front_tuck2 = generate_child_tab(right_front_tuck1,
                                      tab_width = tower_height,
                                      tab_length = 0.75 * front_panel_width,
                                      tab_angle = 0, 
                                      bend_angle = pi/2,
                                      parent_edge_number = 1,
                                      parent_offset = 0)

back_panel_flange_left = generate_child_tab(back_panel,
                                     tab_width = 0.25 * tower_width,
                                     tab_length = support_tab_length,
                                     tab_angle = -support_tab_angle, 
                                     bend_angle = pi/2,
                                     parent_edge_number = 1,
                                     parent_offset = support_tab_offset)
back_panel_flange_right = generate_child_tab(back_panel,
                                     tab_width = 0.25 * tower_width,
                                     tab_length = support_tab_length,
                                     tab_angle = support_tab_angle, 
                                     bend_angle = pi/2,
                                     parent_edge_number = 1,
                                     parent_offset = tower_width - 0.25 * tower_width - support_tab_offset)
right_side_panel_flange1 = generate_child_tab(right_side_panel,
                                     tab_width = 0.25 * tower_depth,
                                     tab_length = support_tab_length,
                                     tab_angle = -support_tab_angle, 
                                     bend_angle = pi/2,
                                     parent_edge_number = 0,
                                     parent_offset = support_tab_offset)
right_side_panel_flange2 = generate_child_tab(right_side_panel,
                                     tab_width = 0.25 * tower_depth,
                                     tab_length = support_tab_length,
                                     tab_angle = support_tab_angle, 
                                     bend_angle = pi/2,
                                     parent_edge_number = 0 ,
                                     parent_offset = tower_depth - 0.25 * tower_depth - support_tab_offset)
left_side_panel_flange1 = generate_child_tab(left_side_panel,
                                     tab_width = 0.25 * tower_depth,
                                     tab_length = support_tab_length,
                                     tab_angle = -support_tab_angle, 
                                     bend_angle = pi/2,
                                     parent_edge_number = 2,
                                     parent_offset = support_tab_offset)
left_side_panel_flange2 = generate_child_tab(left_side_panel ,
                                     tab_width = 0.25 * tower_depth,
                                     tab_length = support_tab_length,
                                     tab_angle = support_tab_angle, 
                                     bend_angle = pi/2,
                                     parent_edge_number = 2 ,
                                     parent_offset = tower_depth - 0.25 * tower_depth - support_tab_offset)
right_front_flange = generate_child_tab(right_front_panel,
                                        tab_width = 0.5 * front_panel_width,
                                        tab_length = support_tab_length,
                                        tab_angle = 0,
                                        bend_angle = pi/2,
                                        parent_edge_number = 0 ,
                                        parent_offset = support_tab_offset)
left_front_flange = generate_child_tab(left_front_panel,
                                        tab_width = 0.5 * front_panel_width,
                                        tab_length = support_tab_length,
                                        tab_angle = 0,
                                        bend_angle = pi/2,
                                        parent_edge_number = 2 ,
                                        parent_offset = support_tab_offset)
# child0 = generate_child_tab(root,
#                             edge_length,
#                             edge_length,
#                             tab_angle,
#                             bend_angle,
#                             0, 0)
# child1 = generate_child_tab(root,
#                             edge_length,
#                             edge_length,
#                             tab_angle,
#                             bend_angle,
#                             1, 0)
# child2 = generate_child_tab(root,
#                             edge_length,
#                             edge_length,
#                             tab_angle,
#                             bend_angle,
#                             2, 0)
# child3 = generate_child_tab(root,
#                             edge_length,
#                             edge_length,
#                             tab_angle,
#                             bend_angle,
#                             3, 0)
# child4 = generate_child_tab(child0,
#                             edge_length,
#                             edge_length,
#                             tab_angle,
#                             bend_angle,
#                             1 , 0)
 
draw_svg(back_panel, "slide_mount.svg") 