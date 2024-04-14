import numpy as np
import time
import trimesh
from trimesh import Trimesh
from typing import NamedTuple
from pathlib import Path
from typeguard import typechecked
import fire
from scipy.spatial.distance import pdist
from intersection import triangle_plane_intersection, dist_squared
from gcode import convert_to_gcode, write_contours


@typechecked
def slice_to_gcode(stl_in: str, gcode_out: str, dz: float) -> list[list[list[np.ndarray]]]:
    mesh = trimesh.load(stl_in)
    assert isinstance(mesh, Trimesh)
    mesh, bottom, top = transform_to_fit_bed(mesh)
    print(f"Slicing {stl_in} with slice height {dz:.3f}...")
    t_start = time.time()
    edges = slice_mesh(mesh, bottom, top, dz)
    t_slice = time.time()
    contours = create_contours(edges)
    t_contour = time.time()

    print(f"\tSliced into {len(contours)} layers, with {sum([len(l) for l in contours])} total contours.")
    print(f"\t[Slicing: {(t_slice - t_start):.3f}s, Stitching: {(t_contour - t_slice):.3f}s]")
    convert_to_gcode(gcode_out, contours)
    return contours


@typechecked
def transform_to_fit_bed(mesh: Trimesh) -> tuple[Trimesh, float, float]:
    # Compute the bounding box of our mesh
    obj_min = mesh.vertices.min(axis=0)
    obj_max = mesh.vertices.max(axis=0)

    # Hypothetical print bed
    bed_min = np.array([0, 0, 0])
    bed_max = np.array([220, 220, 100])

    # Scale model to fit the bed dimensions, and translate it to fit the center of the bed.
    bed_dim = bed_max - bed_min
    obj_dim = obj_max - obj_min

    scale = min(1.0, bed_dim[0] / obj_dim[0], bed_dim[1] / obj_dim[1], bed_dim[2] / obj_dim[2])
    obj_center = (obj_min + obj_max) / 2  # Get the center.
    obj_center[2] = obj_min[2]  # Drop it like it's hot.

    bed_center = (bed_min + bed_max) / 2
    bed_center[2] = bed_min[2]

    translation = bed_center - obj_center
    scaled_translation = obj_center * (1.0 - scale) + translation
    transform_matrix = trimesh.transformations.scale_and_translate(scale, scaled_translation)
    mesh.apply_transform(transform_matrix)

    bottom = obj_min[2] * scale + scaled_translation[2]
    top = obj_max[2] * scale + scaled_translation[2]
    return (mesh, bottom, top)


class Edge:
    def __init__(self, start: np.ndarray, end: np.ndarray):
        self.start = start  # (3,)
        self.end = end  # (3,)

    def __repr__(self) -> str:
        return f"({format_vertex(self.start)} -> {format_vertex(self.end)})"


def format_vertex(vtx: np.ndarray) -> str:
    return f"{vtx[0]:.1f},{vtx[1]:.1f},{vtx[2]:.1f}"


@typechecked
def slice_mesh(mesh: trimesh.Trimesh, bottom: float, top: float, dz: float) -> list[list[Edge]]:
    """
    Input:
      mesh, the triangle mesh to be sliced
      bottom, the bottom Z coordinate of the mesh.
      top, the top Z coordinate of the mesh.
      dz, the vertical distance between slicing layers.

    Output:
      intersection_edges, a list of edges (in no particular order) for each layer.

    Hint:
      1. Enumerate each plane.
      2. You should not intersect each triangle with each plane. Think about strategies to minimize the number
          of intersections. For a given triangle, how can we know in advance which planes it may intersect?
    """
    total_height = top - bottom
    num_layers = int(np.ceil(total_height / dz))
    slice_plane_heights = [bottom + h * dz for h in range(0, num_layers + 1)]
    plane_normal = np.array([0.0, 0.0, 1.0])

    filtered_triangles = [[] for _ in slice_plane_heights]

    for tri in mesh.triangles:
        # Determine which planes intersect this triangle by Z coordinates
        minZ = min(tri[0][2], tri[1][2], tri[2][2])
        maxZ = max(tri[0][2], tri[1][2], tri[2][2])

        z = minZ
        j = int(np.floor((z - bottom) / dz))
        while (j - 1) * dz < maxZ and j < len(slice_plane_heights):
            filtered_triangles[j].append(tri)
            j += 1

    intersection_edges = []

    for i, height in enumerate(slice_plane_heights):
        plane_origin = np.array([0.0, 0.0, height])

        slice_edges = []
        for tri in filtered_triangles[i]:  # For every triangle this plane intersects
            edges = []

            ix = triangle_plane_intersection(tri, plane_origin, plane_normal)
            if len(ix) >= 2:
                edges.append(Edge(ix[0], ix[1]))
                if len(ix) == 3:
                    edges.append(Edge(ix[1], ix[2]))
                    edges.append(Edge(ix[2], ix[0]))
            slice_edges.extend(edges)

        intersection_edges.append(slice_edges)

    return intersection_edges


def findCycleDfs(current_vertex, parent_vertex, visit_status: list,
			par: list, graph, cycles):

	# If the vertex has been fully visited already
	if visit_status[current_vertex] == 2:
		return
   
     
     
   # If the vertex has already been visited, but not fully (flag = 1),
   # Then we have detected a cycle and want to backtrack to find the full cycle
	if visit_status[current_vertex] == 1:
		other_vertices = []
        
        # Add the parent vertex to the list of other vertices
		temp_vertex = parent_vertex
		other_vertices.append(temp_vertex)

		# As long as our parent vertex is not the current vertex,
        # backtrack through the vertices
		while temp_vertex != current_vertex:
             
            # Update temp_vertex to be its parent
			temp_vertex = par[temp_vertex]
            
            # If temp_vertex comes back as 0, this means 
            
            # Add this vertex to the list of vertices in the cycle
			other_vertices.append(temp_vertex)

		# Add this list of vertices visited as a cycle
		cycles.append(other_vertices)

		return



	# Update the status of this vertex to partially visited
	par[current_vertex] = parent_vertex
	visit_status[current_vertex] = 1

	# Recursively call the DFS on the current graph
	for other_vertex in graph[current_vertex]:

		# if it has not been visited previously
		if other_vertex == par[current_vertex]:
			continue
       
		findCycleDfs(other_vertex, current_vertex, visit_status, par, graph, cycles)

	# Update the status of this vertex to fully visited
	visit_status[current_vertex] = 2



# # add the edges to the graph
# def addEdge(adjacency_list, u, v):
     
# 	is_duplicate_edge = v in adjacency_list[u]
# 	if not(is_duplicate_edge):
# 		adjacency_list[u].append(v)
# 		adjacency_list[v].append(u)
    
# 	return adjacency_list

# # Function to print the cycles
# def printCycles(cycles):
     
# 	# print all the vertex with same cycle
# 	for i in range(0, len(cycles)):

# 		# Print the i-th cycle
# 		print("Cycle Number %d:" % (i+1), end = " ")
# 		for x in cycles[i]:
# 			print(x, end = " ")
# 		print()

# def getInitialConnectedNodeIndex (initial_node_index, adjacency_list):

#      # temp_edges = [adjacency_list[i] for i in range(len(adjacency_list)) 
#      #                  if initial_node_index in adjacency_list[i]]
          
#      # initial_edge = temp_edges[0]
#      # temp_node_index = [initial_edge[i] for i in range(2) if initial_edge[i] != initial_node_index]
#      # connected_node_index = temp_node_index[0]
     
     
#      connected_node_indices = adjacency_list[initial_node_index]
#      connected_node_index = connected_node_indices[0]


#      return connected_node_index


def find_and_add_closest_node(current_nodes, temp_contour, maximum_tol):
     
     # Check that current_nodes has entries -- if it doesn't, exit (no nodes to be added)
     try:
          if (current_nodes.shape[1] == 0):
               exit_flag = 2
               return current_nodes, temp_contour, exit_flag
     except:
          exit_flag = 2
          return current_nodes, temp_contour, exit_flag

     # Find the closest node to the current END node of the contour
     node_displacements = current_nodes - temp_contour[-1]
     node_distances = np.sqrt(np.sum(node_displacements**2, axis = 2))
     min_distance_indices = np.where(node_distances == np.min(node_distances))

     # If there are multiple locations with equal minima, select the first one
     closest_node_index = np.array([min_distance_indices[0][0], 
                                   min_distance_indices[1][0]])

     # Check whether this node is within the tolerance
     is_node_in_tolerance = node_distances[closest_node_index[0],
                                           closest_node_index[1]] <= maximum_tol

     # If node is within tolerance, add it to our contour
     if is_node_in_tolerance:
          
          # Get both points associated with this edge
          temp_closest_node = current_nodes[closest_node_index[0],
                                         closest_node_index[1],
                                         :]
          temp_closest_node_connection = current_nodes[int(not(closest_node_index[0])),
                                          closest_node_index[1],
                                          :]
          
          is_closed_contour = np.all(temp_contour[-1] == temp_closest_node_connection)

                                 
          if is_closed_contour:
               if not(np.all(temp_closest_node == temp_contour[-1])):
                    temp_contour.append(temp_closest_node)
                    
               # Update list of available nodes/edges
               current_nodes = np.delete(current_nodes, closest_node_index[1], axis = 1)

          else:
                            
               # If the new closest node is exactly equal to the current end node, directly add the other point of the edge
               if np.all(temp_closest_node == temp_contour[-1]):
                    temp_contour.append(temp_closest_node_connection)
               
               # Otherwise, add the closest node AND the other point on the edge
               else:
                    temp_contour.append(temp_closest_node)
                    temp_contour.append(temp_closest_node_connection)
                    
               # Update list of available nodes/edges
               current_nodes = np.delete(current_nodes, closest_node_index[1], axis = 1)
               
          exit_flag = 0
          
     # If there were no neighbors in tolerance, return an empty contour
     else:
           exit_flag = 1
               
     return temp_contour, current_nodes, exit_flag
                         
               
               
def find_individual_contour(current_nodes, maximum_tol):
     
     # Initialize contour with first point
     temp_contour = []
     temp_edge_index = 0
     temp_contour.append(current_nodes[0, temp_edge_index, :])
     temp_contour.append(current_nodes[1, temp_edge_index, :])
     
     # Update list of available nodes/edges             
     current_nodes = np.delete(current_nodes, temp_edge_index, axis = 1)
     
     # Initialize convergence criteria
     are_remaining_nodes = True
     is_closed_contour = False
     are_remaining_nodes = True
     num_iterations = 0;
     max_num_iterations = 5000
     is_search_converged = any([(num_iterations >= max_num_iterations),
                              is_closed_contour,
                              not(are_remaining_nodes)])
     exit_flag1 = 0
     exit_flag2 = 0
        
     while not(is_search_converged):
            
            # Search for point to add to beginning of contour
            temp_contour, current_nodes, exit_flag1 = \
                 find_and_add_closest_node(current_nodes, temp_contour, maximum_tol)            
     
            # Update convergence criteria: did we finish the contour by trying to connect a point to the end?
            are_remaining_nodes = check_for_remaining_nodes(current_nodes)
            try:
                 is_closed_contour = np.all(temp_contour[0] == temp_contour[-1])
            except:
                 is_closed_contour = True
                 
            is_beginning_converged = any([(num_iterations >= 5),
                                      is_closed_contour,
                                      not(are_remaining_nodes)])           
                 
            
            
            # Search for point to add to end of contour
            if not(is_beginning_converged):
                 
                 flipped_contour, current_nodes, exit_flag2 = \
                      find_and_add_closest_node(current_nodes, temp_contour[::-1], maximum_tol)
                      
                 temp_contour = flipped_contour[::-1]
                      
            # Update convergence criteria: did we finish the contour by trying to connect a point to the start?
            are_remaining_nodes = check_for_remaining_nodes(current_nodes)
            try:
                 is_closed_contour = np.all(temp_contour[0] == temp_contour[-1])
            except:
                 is_closed_contour = True
                      
            num_iterations += 1
            is_contour_disconnected = exit_flag1 and exit_flag2
               
            is_search_converged = any([(num_iterations >= max_num_iterations),
                                        is_closed_contour,
                                        not(are_remaining_nodes),
                                        is_contour_disconnected])
                 
     return temp_contour, current_nodes

def check_for_remaining_nodes(current_nodes):
     
     try:
          are_remaining_nodes = (current_nodes.shape[1]) > 0
     except:
          are_remaining_nodes = False
          
     return are_remaining_nodes
    
@typechecked
def create_contours(intersection_edges: list[list[Edge]]) -> list[list[list[np.ndarray]]]:
    """
    TODO: 2.1
    Input:
        intersection_edges, the "edge soups" you created in `slice_mesh`.

    Output: 
        contours, a a 3D matrix of 3D vertices.
        contours[i] represents the i'th layer in your slice.
        contours[i][j] represents the j'th individual closed contour in the i'th layer.
        contours[i][j][k] is the k'th vertex of the j'th contour, and is itself a 1 x 3 matrix
        of X,Y,Z vertex coordinates. You should not "close" the contour by adding the start point back
        to the end of the list; this segment will be treated as implicitly present.

    Hints:
     1. Think of how to connect cutting edges. Remember that edges in the input "soup" may be in arbitrary
        orientations.
     2. Some edges may be isolated and cannot form a contour. Detect and remove these.
        Bonus (0 points): think about what causes this to happen.
     3. There can and will be multiple contours in a layer (this is why we have the [j] dimension).
        Think about how to detect and handle this case.
     4. Think about how to optimize this: is there a way we can identify neighboring edges faster than by
        looping over all the other edges to figure out which are connected to it?
    """
    layers: list[list[list[np.ndarray]]] = []
    maximum_tol = 0.0010
    decimal_tol = 4

    for layer_index, layer in enumerate(intersection_edges):

        print(f'Layer: {layer_index}')
        # TODO: Your code here.
        #       Build potentially many contours out of a single layer by connecting edges.
        
        # Only proceed if there is information on this layer
        temp_layer_contours = []
        
        if layer:
             
             # Extract edge start and end nodes for this layer
             num_edges = len(layer)
             is_remaining_edge = np.full((num_edges,), True)
             reordered_edges = np.zeros((num_edges, 6))
             
             # Sort edges so that by [x, y, z] within each edge (ascending)
             for edge_index in range(num_edges):
                  
                  temp_nodes = np.array([layer[edge_index].start, layer[edge_index].end])
                  
                  sort_indices = np.lexsort((temp_nodes[:,2],
                                            temp_nodes[:,1],
                                            temp_nodes[:,0]))
                  
                  # reordered_edge = temp_nodes[sort_indices, :]
                  reordered_edges[edge_index, :] = temp_nodes[sort_indices, :].flatten()                

             all_nodes = np.zeros((2, num_edges, 3))
             all_nodes[0, :, :] = reordered_edges[:, 0:3]
             all_nodes[1, :, :] = reordered_edges[:, 3:6]
             
             current_nodes = all_nodes
             
             # Initialize convergence criteria
             num_iterations = 0
             max_contours_per_layer = 1000
             are_remaining_nodes = True
             is_layer_converged = any([num_iterations >= max_contours_per_layer,
                                       not(are_remaining_nodes)])
             
             while not(is_layer_converged):
                  temp_contour, current_nodes = find_individual_contour(current_nodes, maximum_tol)
                  
                  # After finishing this contour, add it to the list of contours for this layer
                  if len(temp_contour) > 0:
                       if all(temp_contour[0] == temp_contour[-1]):
                        
                            # find_individual_contours can include the endpoint (duplicated)
                            # We will need to remove this to match the gcode exporting
                            temp_layer_contours.append(temp_contour[:-1])
                       else:
                            temp_layer_contours.append(temp_contour)
                    
                  num_iterations += 1
                  are_remaining_nodes = check_for_remaining_nodes(current_nodes)
                  is_layer_converged = any([num_iterations >= max_contours_per_layer,
                                            not(are_remaining_nodes)])
             
             # After finishing this layer, add all contours from this layer
             layers.append(temp_layer_contours)
                 

    return layers







@typechecked
def main(model_name: str, slice_height: float = 0.4):
    stl_file_path = Path(f"data/{model_name}.stl")
    gcode_file = Path(f"output/{model_name}.gcode")
    contour_file = Path(f"output/{model_name}_contour.txt")
    assert stl_file_path.exists(), f"{model_name} does not exist"

    Path("output").mkdir(exist_ok=True)

    contours = slice_to_gcode(str(stl_file_path), str(gcode_file), slice_height)
    write_contours(str(contour_file), contours)


if __name__ == "__main__":
    fire.Fire(main)
