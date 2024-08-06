import numpy as np
from matplotlib import pyplot as plt
from typing import Union, Literal, Optional
from typeguard import typechecked
import fire
from pathlib import Path



def print_points(points):
    print("[" + ", ".join(f"[{pt[0]:.5f},{pt[1]:.5f}]" for pt in points) + "]")
    return None


def pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Compute the Pareto front of a set of 2D points. Minimization is assumed for all properties.

    points: (n, 2)
    returns: (n_pareto, 2) in order of non-decreasing x
    """
    # TODO: 1.1
    assert points.ndim == 2 and points.shape[-1] == 2 and points.shape[0] > 0, \
        'The input array must represent a set of 2D points'

    # Sort the points by both properties
    # --------
    # TODO: Your code here.
    # HINT:
    #   1. The function you will use here is called `np.lexsort`, whose documentation is at
    #      https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html. `np.lexsort`
    #      supports sorting by multiple columns in a user-specified order. You can decide whichever
    #      order you want here.
    #   2. Indexing a NumPy array using the output of `np.lexsort` returns the sorted array.
    # points = points     # <--
    # Sort points by 0th objective, then by 1st objective value
    points = np.array([(points[i, 0], points[i, 1]) for i in np.lexsort((-points[:,1], points[:,0]))])

     # Initialize storage of information about the pareto front
    pareto_indices = [0]        # List of indices to Pareto-optimal points (in the sorted array)
    pareto_x = points[0, 0]     # X value of the last Pareto-optimal point
    pareto_y = points[0, 1]     # Y value of the last Pareto-optimal point

    # Traverse the sorted array to figure out Pareto-optimal points
    for i in range(points.shape[0]):

        # Add this point to the Pareto front if it isn't dominated by the last Pareto-optimal point
        # --------
        # TODO: You code here.        

        if pareto_y > points[i, 1]:       # <--
             # If the x-value of our current point is the same as the last
             # point on the pareto front, remove the last point on the pareto
             # front and replace it with this point
             if pareto_x == points[i, 0]:
                  
                  # Remove the last point from pareto_indices
                  pareto_indices.pop(-1)
                  
                  # Add current point
                  pareto_indices.append(i)
                  
                  # Update the last Pareto-optimal point using this point
                  pareto_x = points[i, 0]
                  pareto_y = points[i, 1]
        
        
             # If the current y-value is higher-performance (lower objective) than 
             # the most recent point on our pareto front, add this point to the pareto front
             else:        
                 pareto_indices.append(i)
     
                 # Update the last Pareto-optimal point using this point
                 # --------
                 # TODO: Your code here.
                 pareto_x = points[i, 0]
                 pareto_y = points[i, 1]

    # Return the Pareto front
    pareto_front = points[pareto_indices]

    return pareto_front

def pareto_brute_force(points: np.ndarray) -> np.ndarray:
    """
    Get the Pareto front by brute-force, O(n^2) algorithm, to serve as a correctness reference.

    points: (n, 2)
    returns: (n_pareto, 2) in order of non-decreasing x
    """
    dom = (points[:, None] >= points[None]).all(axis=2)
    num_points = len(points)
    dom[np.arange(num_points), np.arange(num_points)] = False
    pareto_gt = points[~dom.any(axis=1)]
    return pareto_gt[np.lexsort((pareto_gt[:, 1], pareto_gt[:, 0]))]


@typechecked
def main(
    *,
    n: int = 10,
    filename: Union[str, Path, None] = None,
    test: bool = True,
    verbose: Optional[bool] = None,
):
    if filename is None:
        points = np.random.uniform(0, 1, (n, 2))
    else:
        filename = Path(filename)
        assert filename.exists(), f"{filename} does not exist"
        points = np.load(filename)

    if verbose is None:
        verbose = filename is None

    if verbose:
        print("Input Points:")
        print_points(points)

    pareto_user = pareto_front(points)

    if verbose:
        print("Found Pareto Front:")
        print_points(pareto_user)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.scatter(points[:, 0], points[:, 1])
    ax.scatter(pareto_user[:, 0], pareto_user[:, 1])
    if filename is None:
        ax.set_title(f"Random: {n} points")
        fname = f"random-{n}.png"
    else:
        assert isinstance(filename, Path)
        ax.set_title(filename.stem)
        fname = f"file-{filename.stem}.png"
    Path("output").mkdir(parents=True, exist_ok=True)
    fig.savefig(Path("output") / fname)

    if test:
        # Check correctness
        pareto_gt = pareto_brute_force(points)
        if np.allclose(pareto_gt, pareto_user):
            print("Your output matches the model solution.")
        else:
            print("Your output does not matches the model solution.")


# Unit test
if __name__ == "__main__":
    fire.Fire(main)
