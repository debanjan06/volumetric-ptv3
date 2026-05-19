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
        
        self.file_list = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.pt')]
        if not self.file_list:
            raise RuntimeError(f"No valid pre-processed tensor packages discovered inside '{data_directory}'")
            
        print(f"\n=== Initialized RAM-Optimized Lazy Data Lake ===")
        print(f"   -> Tracked Pre-Voxelized Assets: {len(self.file_list)}")

    def __len__(self):
        return len(self.file_list) * self.chunks_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.chunks_per_file
        file_path = self.file_list[file_idx]
        
        # OPTIMIZATION: Load weights with map_location='cpu' to prevent memory spikes
        payload = torch.load(file_path, map_location='cpu')
        
        xyz_all = payload['xyz']
        intensity_all = payload['intensity']
        labels_all = payload['labels']
        
        total_points = len(labels_all)
        
        # Compute random slices rather than copying massive arrays in memory
        if total_points > self.max_points_per_block:
            sampling_indices = np.random.choice(total_points, self.max_points_per_block, replace=False)
        else:
            sampling_indices = np.random.choice(total_points, self.max_points_per_block, replace=True)
            
        # Extract the precise 8192 point subset needed for the active batch
        xyz = xyz_all[sampling_indices]
        intensity_features = intensity_all[sampling_indices]
        labels = labels_all[sampling_indices]
        
        # Clear references immediately to free up RAM overhead
        del payload
        
        xyz_min = np.min(xyz, axis=0)
        xyz_max = np.max(xyz, axis=0)
        xyz_scaled = (xyz - xyz_min) / (xyz_max - xyz_min + 1e-6)
        
        combined_4d_features = np.concatenate([xyz_scaled, intensity_features], axis=-1)
        sorting_order = self.serializer.serialize_point_stream(xyz)
        
        coords_sorted = torch.tensor(xyz[sorting_order], dtype=torch.float32)
        features_sorted = torch.tensor(combined_4d_features[sorting_order], dtype=torch.float32)
        labels_sorted = torch.tensor(labels[sorting_order], dtype=torch.long)

        return coords_sorted, features_sorted, labels_sorted