import os
import numpy as np
import torch
from torch.utils.data import Dataset

class DALESCoordinateSerializer:
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
    def __init__(self, data_directory, quantization_scale=10.0, max_points_per_block=8192, chunks_per_file=32):
        self.data_dir = data_directory
        self.serializer = DALESCoordinateSerializer(quantization_scale)
        self.max_points_per_block = max_points_per_block
        self.chunks_per_file = chunks_per_file
        
        self.file_list = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.ply')]
        if not self.file_list:
            raise RuntimeError(f"No valid .ply target files discovered inside '{data_directory}'")
            
        # Target placeholder to store global class frequencies for class balancing
        self.class_counts = np.zeros(16, dtype=np.int64)
        print(f"\n=== Initialized Scaled Lazy Spatial Data Lake ===")
        print(f"   -> Tracked Files on Disk: {len(self.file_list)}")

    def __len__(self):
        return len(self.file_list) * self.chunks_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.chunks_per_file
        file_path = self.file_list[file_idx]
        
        header_offset = 0
        with open(file_path, 'rb') as f:
            for line in f:
                header_offset += len(line)
                if line.decode('ascii', errors='ignore').strip() == "end_header":
                    break
        
        ply_dt = np.dtype([
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('intensity', 'i4'), ('sem_class', 'i4'), ('ins_class', 'i4')
        ])
        
        raw_data = np.fromfile(file_path, dtype=ply_dt, offset=header_offset)
        total_points = len(raw_data)
        
        if total_points > self.max_points_per_block:
            sampling_indices = np.random.choice(total_points, self.max_points_per_block, replace=False)
            block_data = raw_data[sampling_indices]
        else:
            sampling_indices = np.random.choice(total_points, self.max_points_per_block, replace=True)
            block_data = raw_data[sampling_indices]
            
        xyz = np.stack([block_data['x'], block_data['y'], block_data['z']], axis=1)
        
        # FIX: Normalize raw 16-bit sensor intensity metrics cleanly between 0.0 and 1.0
        raw_intensity = block_data['intensity'].astype(np.float32)
        normalized_intensity = raw_intensity / 65535.0
        intensity_features = normalized_intensity.reshape(-1, 1)
        
        labels = block_data['sem_class'].astype(np.int64)

        sorting_order = self.serializer.serialize_point_stream(xyz)
        
        coords_sorted = torch.tensor(xyz[sorting_order], dtype=torch.float32)
        features_sorted = torch.tensor(intensity_features[sorting_order], dtype=torch.float32)
        labels_sorted = torch.tensor(labels[sorting_order], dtype=torch.long)

        return coords_sorted, features_sorted, labels_sorted