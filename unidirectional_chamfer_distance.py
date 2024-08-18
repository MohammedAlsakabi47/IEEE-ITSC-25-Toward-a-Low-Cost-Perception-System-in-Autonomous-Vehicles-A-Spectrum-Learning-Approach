from scipy.spatial import cKDTree
import numpy as np

def unidirectional_chamfer_distance(source, target):
    # Create a KD-tree for the target point cloud
    target_tree = cKDTree(target)
    
    # Find the nearest neighbor in the target for each point in the source
    distances, _ = target_tree.query(source)
    
    # Calculate the mean distance
    chamfer_dist = np.mean(distances)
    
    return chamfer_dist

def test(a,b):
    return a+b