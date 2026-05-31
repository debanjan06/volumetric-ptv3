import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import cKDTree


class DALESCoordinateSerializer:
    def __init__(self, quantization_scale=10.0):
        self.scale = quantization_scale

    def compute_morton_3d_index(self, quantized_xyz):
        x = quantized_xyz[:, 0]
        y = quantized_xyz[:, 1]
        z = quantized_xyz[:, 2]

        morton_index = 0
        for i in range(10):
            morton_index |= (
                ((x & (1 << i)) << (2 * i))
                | ((y & (1 << i)) << (2 * i + 1))
                | ((z & (1 << i)) << (2 * i + 2))
            )
        return morton_index

    def serialize_point_stream(self, xyz):
        min_bounds = np.min(xyz, axis=0)
        quantized_xyz = np.floor((xyz - min_bounds) * self.scale).astype(np.int64)
        spatial_keys = self.compute_morton_3d_index(quantized_xyz)
        return np.argsort(spatial_keys)


class DALESProductionDataset(Dataset):
    def __init__(
        self,
        data_directory,
        quantization_scale=10.0,
        max_points_per_block=8192,
        chunks_per_file=32,
    ):
        self.data_dir = data_directory
        self.serializer = DALESCoordinateSerializer(quantization_scale)
        self.max_points_per_block = max_points_per_block
        self.chunks_per_file = chunks_per_file

        self.file_list = [
            os.path.join(data_directory, f)
            for f in os.listdir(data_directory)
            if f.endswith(".pt")
        ]
        if not self.file_list:
            raise RuntimeError(
                f"No valid pre-processed tensor packages discovered inside '{data_directory}'"
            )

        print("\n=== Initialized 14D Neighborhood Spatial Data Lake ===")

    def __len__(self):
        return len(self.file_list) * self.chunks_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.chunks_per_file
        file_path = self.file_list[file_idx]

        payload = torch.load(file_path, map_location="cpu", weights_only=False)

        xyz_all = payload["xyz"]
        intensity_all = payload["intensity"]
        context_all = payload["context_features"]
        labels_all = payload["labels"]

        total_points = len(labels_all)
        sampling_indices = np.random.choice(
            total_points,
            self.max_points_per_block,
            replace=(total_points <= self.max_points_per_block),
        )

        xyz = xyz_all[sampling_indices]
        intensity_features = intensity_all[sampling_indices]
        context_features = context_all[sampling_indices]
        labels = labels_all[sampling_indices]

        del payload

        # Local coordinate bounding normalization
        xyz_min = np.min(xyz, axis=0)
        xyz_max = np.max(xyz, axis=0)
        xyz_scaled = (xyz - xyz_min) / (xyz_max - xyz_min + 1e-6)

        # Compile base 7D feature array
        base_7d_features = np.concatenate(
            [xyz_scaled, intensity_features, context_features], axis=-1
        )

        # --- FEATURE ENGINEERING: LOCAL NEIGHBORHOOD AGGREGATION ---
        # Build an optimized spatial tree over the 8192 sampled point block
        tree = cKDTree(xyz)

        # Query the 8 nearest neighbors for every single point (including itself)
        _, neighbor_indices = tree.query(xyz, k=8)

        # Compute the mean feature values across the spatial neighborhoods
        # Shape: [8192, 8, 7] -> Mean over axis 1 -> [8192, 7]
        neighbor_features = base_7d_features[neighbor_indices]
        mean_neighbor_features = np.mean(neighbor_features, axis=1)

        # Concatenate base features with neighbor features to form the 14D representation
        combined_14d_features = np.concatenate(
            [base_7d_features, mean_neighbor_features], axis=-1
        )

        sorting_order = self.serializer.serialize_point_stream(xyz)

        coords_sorted = torch.tensor(xyz[sorting_order], dtype=torch.float32)
        features_sorted = torch.tensor(
            combined_14d_features[sorting_order], dtype=torch.float32
        )
        labels_sorted = torch.tensor(labels[sorting_order], dtype=torch.long)

        return coords_sorted, features_sorted, labels_sorted
