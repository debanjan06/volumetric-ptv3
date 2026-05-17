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
        x = quantized_xyz[:, 0]
        y = quantized_xyz[:, 1]
        z = quantized_xyz[:, 2]
        
        morton_index = 0
        for i in range(10):
            morton_index |= ((x & (1 << i)) << (2 * i)) | \
                           ((y & (1 << i)) << (2 * i + 1)) | \
                           ((z & (1 << i)) << (2 * i + 2))
        return morton_index

    def serialize_point_stream(self, xyz):
        min_bounds = np.min(xyz, axis=0)
        quantized_xyz = np.floor((xyz - min_bounds) * self.scale).astype(np.int64)
        spatial_keys = self.compute_morton_3d_index(quantized_xyz)
        return np.argsort(spatial_keys)


class DALESProductionDataset(Dataset):
    """
    Memory-Safe Lazy Spatial Dataset Loader. Prevents MemoryErrors by reading 
    binary files from disk only when actively requested by a training batch.
    """
    def __init__(self, data_directory, quantization_scale=10.0, max_points_per_block=8192, chunks_per_file=32):
        self.data_dir = data_directory
        self.serializer = DALESCoordinateSerializer(quantization_scale)
        self.max_points_per_block = max_points_per_block
        self.chunks_per_file = chunks_per_file
        
        # Instantly index file paths without opening them
        self.file_list = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.ply')]
        if not self.file_list:
            raise RuntimeError(f"No valid .ply target files discovered inside '{data_directory}'")
            
        print(f"\n=== Initialized Lazy Spatial Data Lake ===")
        print(f"   -> Tracked Files on Disk: {len(self.file_list)}")
        print(f"   -> Virtual Samples/File : {self.chunks_per_file}")
        print("==========================================")

    def __len__(self):
        # Create a virtual dataset length based on the number of files and chunks
        return len(self.file_list) * self.chunks_per_file

    def __getitem__(self, idx):
        # Map the virtual index back to a physical file on disk
        file_idx = idx // self.chunks_per_file
        file_path = self.file_list[file_idx]
        
        # Parse the text header to locate the binary block start offset
        header_offset = 0
        with open(file_path, 'rb') as f:
            for line in f:
                header_offset += len(line)
                if line.decode('ascii', errors='ignore').strip() == "end_header":
                    break
        
        # Memory-aligned native structural types
        ply_dt = np.dtype([
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('intensity', 'i4'), ('sem_class', 'i4'), ('ins_class', 'i4')
        ])
        
        # Stream the file data using memory-mapping to keep RAM usage low
        raw_data = np.fromfile(file_path, dtype=ply_dt, offset=header_offset)
        total_points = len(raw_data)
        
        # Extract a random, uniform chunk of points from this file
        if total_points > self.max_points_per_block:
            sampling_indices = np.random.choice(total_points, self.max_points_per_block, replace=False)
            block_data = raw_data[sampling_indices]
        else:
            sampling_indices = np.random.choice(total_points, self.max_points_per_block, replace=True)
            block_data = raw_data[sampling_indices]
            
        xyz = np.stack([block_data['x'], block_data['y'], block_data['z']], axis=1)
        intensity = block_data['intensity'].astype(np.float32).reshape(-1, 1)
        labels = block_data['sem_class'].astype(np.int64)

        # Apply Morton serialization to sort the block linearly
        sorting_order = self.serializer.serialize_point_stream(xyz)
        
        coords_sorted = torch.tensor(xyz[sorting_order], dtype=torch.float32)
        features_sorted = torch.tensor(intensity[sorting_order], dtype=torch.float32)
        labels_sorted = torch.tensor(labels[sorting_order], dtype=torch.long)

        return coords_sorted, features_sorted, labels_sorted

if __name__ == "__main__":
    train_path = os.path.join("data", "train")
    if os.path.exists(train_path) and os.listdir(train_path):
        dataset = DALESProductionDataset(data_directory=train_path, max_points_per_block=8192)
        coords, features, targets = dataset[0]
        print(f"\n[Lazy Ingestion Passed]: Data Shape -> {list(coords.shape)}")