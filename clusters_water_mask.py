
import numpy as np
from scipy.spatial import cKDTree

def interpolate_mask(small_lat, small_lon, large_lat, large_lon, land_water_mask, cluster_mask):
    """
    Creating cluster and water/land masks for data.

    Args:
        small_lat (np.ndarray): A 2D np array with lat for data.
        small_lon (np.ndarray): A 2D np array with lon for data.
        large_lat (np.ndarray): A 2D np array with the latitude of the entire surface.
        large_lon (np.ndarray): A 2D np array with the longitude of the entire surface.
        land_water_mask (np.ndarray): A 2D np array with water/land mask for the entire surface.
        cluster_mask (np.ndarray): A 2D np array with cluster mask for the entire surface.

    Returns:
        small_land_water_mask (np.ndarray): A 2D np array with water/land mask for the data.
        small_cluster_mask (np.ndarray): A 2D np array with cluster mask for the data.
    """
    
    large_lat_flat = large_lat.flatten()
    large_lon_flat = large_lon.flatten()
    large_coords = np.vstack((large_lat_flat, large_lon_flat)).T

    tree = cKDTree(large_coords)
    
    small_lat_flat = small_lat.flatten()
    small_lon_flat = small_lon.flatten()
    small_coords = np.vstack((small_lat_flat, small_lon_flat)).T
    
    dist, idx = tree.query(small_coords)
    
    land_water_flat = land_water_mask.flatten()
    cluster_flat = cluster_mask.flatten()
    
    small_land_water_mask = land_water_flat[idx].reshape(small_lat.shape)
    small_cluster_mask = cluster_flat[idx].reshape(small_lat.shape)
    
    return small_land_water_mask, small_cluster_mask



if __name__ == "__main__":

    small_lat = np.load('lat_Бофорта.npy')
    small_lon = np.load('lon_Бофорта.npy')
    
    large_lat = np.load('lat_FULL.npy')
    large_lon = np.load('lon_FULL.npy')
    land_water_mask = np.load('land_mask_FULL.npy')
    cluster_mask = np.load('CLUSTERS_FULL.npy')
    

    small_land_water_mask, small_cluster_mask = interpolate_mask(small_lat, small_lon, large_lat, large_lon, land_water_mask, cluster_mask)
