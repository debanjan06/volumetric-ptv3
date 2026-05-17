import numpy as np
import torch

class DALESCoordinateSerializer:
    """
    Quantizes and linearizes unorganized 3D point cloud coordinate matrices
    onto a 1D spatial curve to maximize hardware cache locality (PTv3 Paradigm).
    """
    def __init__(self, quantization_scale=10.0):
        # Scale = 10.0 implies voxel resolution grouping down to 10cm bins
        self.scale = quantization_scale

    def compute_morton_3d_index(self, quantized_xyz):
        """
        Executes bit-interleaving over 3D coordinate columns to generate a 
        unified 1D spatial key array natively.
        """
        # Isolate individual coordinate axes dimensions
        x = quantized_xyz[:, 0]
        y = quantized_xyz[:, 1]
        z = quantized_xyz[:, 2]
        
        morton_index = 0
        # Interleave bits across a grid boundary map up to 1024 (2^10 voxel cells)
        for i in range(10):
            morton_index |= ((x & (1 << i)) << (2 * i)) | \
                           ((y & (1 << i)) << (2 * i + 1)) | \
                           ((z & (1 << i)) << (2 * i + 2))
        return morton_index

    def serialize_point_stream(self, raw_point_cloud):
        """
        Transforms an unstructured [N, 4] raw data matrix into a 
        cache-aligned, sorted sequence block.
        
        Args:
            raw_point_cloud (np.ndarray): Input data array where columns 
                                         represent [X, Y, Z, Intensity]
        """
        print("\n=== Activating Volumetric-PTv3 Spatial Serialization Pass ===")
        xyz = raw_point_cloud[:, :3]
        intensity = raw_point_cloud[:, 3:]
        
        # Step A: Shift coordinates to positive workspace boundaries
        min_bounds = np.min(xyz, axis=0)
        quantized_xyz = np.floor((xyz - min_bounds) * self.scale).astype(np.int64)
        
        # Step B: Compute the interleaved 1D spatial keys
        spatial_keys = self.compute_morton_3d_index(quantized_xyz)
        
        # Step C: Isolate sorting indices to linearize the coordinate layout
        sorting_order = np.argsort(spatial_keys)
        
        # Step D: Construct memory-aligned ordered arrays
        serialized_xyz = xyz[sorting_order]
        serialized_intensity = intensity[sorting_order]
        
        print(f"   -> Input Point Buffer Density : {len(raw_point_cloud)} unorganized points")
        print(f"   -> Linearized Sequence Output : Ordered continuously along 1D curve")
        print("==========================================================")
        
        return (torch.tensor(serialized_xyz, dtype=torch.float32), 
                torch.tensor(serialized_intensity, dtype=torch.float32))

if __name__ == "__main__":
    # Mocking an uncompressed raw DALES tile segment: [X, Y, Z, Sensor Intensity]
    np.random.seed(42)
    mock_point_cloud = np.random.uniform(-50.0, 50.0, (8192, 4))
    
    serializer = DALESCoordinateSerializer(quantization_scale=10.0)
    coords_tensor, features_tensor = serializer.serialize_point_stream(mock_point_cloud)
    
    print("\n[Verification Check]")
    print(f"-> Linearized Coordinates Tensor Shape: {list(coords_tensor.shape)}")
    print(f"-> Serialized Features Tensor Shape   : {list(features_tensor.shape)}")