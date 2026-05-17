import os
import numpy as np
import torch
from torch.utils.data import Dataset

class DALESCoordinateSerializer:
    """
    Quantizes and linearizes unorganized 3D point cloud coordinate matrices
    onto a 1D spatial curve to maximize hardware cache locality (PTv3 Paradigm).
    """
    def __init__(self, quantization_scale=10.0):
        self.scale = quantization_scale

    def compute_morton_3d_index(self, quantized_xyz):
        """
        Executes bit-interleaving over 3D coordinate columns to generate a 
        unified 1D spatial key array natively.
        """
        x = quantized_xyz[:, 0]
        y = quantized_xyz[:, 1]
        z = quantized_xyz[:, 2]
        
        morton_index = 0
        for i in range(10): # Interleave bits up to 1024 voxel boundaries
            morton_index |= ((x & (1 << i)) << (2 * i)) | \
                           ((y & (1 << i)) << (2 * i + 1)) | \
                           ((z & (1 << i)) << (2 * i + 2))
        return morton_index

    def serialize_point_stream(self, xyz, intensity):
        """
        Transforms unorganized spatial vectors into cache-aligned continuous blocks.
        """
        min_bounds = np.min(xyz, axis=0)
        quantized_xyz = np.floor((xyz - min_bounds) * self.scale).astype(np.int64)
        
        spatial_keys = self.compute_morton_3d_index(quantized_xyz)
        sorting_order = np.argsort(spatial_keys)
        
        return sorting_order


class DALESProductionDataset(Dataset):
    """
    High-performance native binary parser streaming 3D PLY tiles directly 
    into memory-aligned PyTorch tensor matrices.
    """
    def __init__(self, data_directory, quantization_scale=10.0, max_points=16384):
        self.data_dir = data_directory
        self.serializer = DALESCoordinateSerializer(quantization_scale)
        self.max_points = max_points
        self.file_list = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.ply')]
        
        if not self.file_list:
            raise RuntimeError(f"No valid .ply target files discovered inside '{data_directory}'")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # Step A: Parse the header length dynamically to find where the binary block begins
        header_offset = 0
        with open(file_path, 'rb') as f:
            for line in f:
                header_offset += len(line)
                if line.decode('ascii', errors='ignore').strip() == "end_header":
                    break
        
        # Step B: Read data utilizing memory-efficient NumPy structural buffers
        # Layout: 3x float32 (x,y,z), 3x int32 (intensity, sem_class, ins_class)
        ply_dt = np.dtype([
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('intensity', 'i4'), ('sem_class', 'i4'), ('ins_class', 'i4')
        ])
        
        raw_data = np.fromfile(file_path, dtype=ply_dt, offset=header_offset)
        
        # Step C: Downsample or slice points to ensure safe uniform batch tracking limits
        if len(raw_data) > self.max_points:
            sampling_indices = np.random.choice(len(raw_data), self.max_points, replace=False)
            data_slice = raw_data[sampling_indices]
        else:
            data_slice = raw_data

        # Step D: Isolate target column sets
        xyz = np.stack([data_slice['x'], data_slice['y'], data_slice['z']], axis=1)
        intensity = data_slice['intensity'].astype(np.float32).reshape(-1, 1)
        labels = data_slice['sem_class'].astype(np.int64)

        # Step E: Compute our PTv3 memory-aligned 1D sequence mapping order
        sorting_order = self.serializer.serialize_point_stream(xyz, intensity)
        
        # Rearrange arrays along our memory-optimized space-filling curve sequence
        coords_sorted = torch.tensor(xyz[sorting_order], dtype=torch.float32)
        features_sorted = torch.tensor(intensity[sorting_order], dtype=torch.float32)
        labels_sorted = torch.tensor(labels[sorting_order], dtype=torch.long)

        return coords_sorted, features_sorted, labels_sorted

if __name__ == "__main__":
    # Test our loader over your actual downloaded training tile file
    train_path = os.path.join("data", "train")
    
    if os.path.exists(train_path):
        try:
            dataset = DALESProductionDataset(data_directory=train_path, max_points=8192)
            coords, features, targets = dataset[0]
            print("\n=== Live DALES Data Ingestion Verification ===")
            print(f"   -> Successfully extracted file from disk cache.")
            print(f"   -> Output Coordinates Tensor Shape : {list(coords.shape)}")
            print(f"   -> Output Features Tensor Shape    : {list(features.shape)}")
            print(f"   -> Output Target Class Tensor Shape: {list(targets.shape)}")
            print("==========================================================")
        except Exception as e:
            print(f"\n[Dataset Verification Failed]: {e}")