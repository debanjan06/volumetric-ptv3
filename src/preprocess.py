import os
import numpy as np
import torch


def run_offline_voxelization():
    print("=== Launching Globally Standardized Preprocessing Pipeline ===")

    drive_base = "/content/drive/MyDrive/DALES_Processed"
    if not os.path.exists(drive_base):
        drive_base = "/content/drive/My Drive/DALES_Processed"

    partitions = ["train", "test"]
    voxel_size = 0.15  # 15cm grid downsampling bounds

    ply_dt = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("intensity", "i4"),
            ("sem_class", "i4"),
            ("ins_class", "i4"),
        ]
    )

    for partition in partitions:
        src_dir = os.path.join(drive_base, partition)
        dest_dir = os.path.join(drive_base, f"{partition}_voxelized")
        os.makedirs(dest_dir, exist_ok=True)

        if not os.path.exists(src_dir):
            continue

        file_list = [f for f in os.listdir(src_dir) if f.endswith(".ply")]
        print(
            f"\nProcessing '{partition}' partition -> Discovered {len(file_list)} assets:"
        )

        for idx, file_name in enumerate(file_list):
            src_path = os.path.join(src_dir, file_name)
            dest_path = os.path.join(dest_dir, file_name.replace(".ply", ".pt"))

            header_offset = 0
            with open(src_path, "rb") as f:
                for line in f:
                    header_offset += len(line)
                    if line.decode("ascii", errors="ignore").strip() == "end_header":
                        break

            raw_data = np.fromfile(src_path, dtype=ply_dt, offset=header_offset)
            xyz_raw = np.stack([raw_data["x"], raw_data["y"], raw_data["z"]], axis=1)

            # --- 1. VECTORIZED MACRO-SCALE DENSITY (2.0m Grids) ---
            macro_grid_size = 2.0
            macro_coords = np.floor(xyz_raw[:, :2] / macro_grid_size).astype(np.int32)
            macro_keys = macro_coords[:, 0] * 73856093 ^ macro_coords[:, 1] * 19349663

            _, inverse_macro, counts_macro = np.unique(
                macro_keys, return_inverse=True, return_counts=True
            )
            raw_densities = counts_macro[inverse_macro].astype(np.float32)

            # Global Scaling: Standardize using dataset-wide density approximations
            norm_density = (raw_densities - 1500.0) / 800.0

            # --- 2. VECTORIZED MICRO-SCALE RELATIVE HEIGHT & VARIANCE (1.0m Grids) ---
            micro_grid_size = 1.0
            micro_coords = np.floor(xyz_raw[:, :2] / micro_grid_size).astype(np.int32)
            micro_keys = micro_coords[:, 0] * 73856093 ^ micro_coords[:, 1] * 19349663

            sort_idx = np.argsort(micro_keys)
            sorted_keys = micro_keys[sort_idx]
            sorted_z = xyz_raw[sort_idx, 2]

            split_idx = np.where(sorted_keys[:-1] != sorted_keys[1:])[0] + 1
            min_z_per_column = np.minimum.reduceat(sorted_z, np.insert(split_idx, 0, 0))

            _, inverse_micro = np.unique(micro_keys, return_inverse=True)
            point_min_z = min_z_per_column[inverse_micro]
            relative_heights = xyz_raw[:, 2] - point_min_z

            # Global Scaling: Robust Gaussian standardization for elevation
            norm_rel_height = (relative_heights - 5.0) / 10.0

            # Calculate height variance cleanly
            sum_z = np.add.reduceat(sorted_z, np.insert(split_idx, 0, 0))
            sum_z_sq = np.add.reduceat(sorted_z**2, np.insert(split_idx, 0, 0))
            counts_micro = np.diff(np.append(np.insert(split_idx, 0, 0), len(sorted_z)))

            mean_z = sum_z / counts_micro
            var_z = (sum_z_sq / counts_micro) - (mean_z**2)
            var_z = np.clip(var_z, 0.0, None)

            point_var_z = var_z[inverse_micro]

            # Global Scaling: Log-transform to handle variance outliers, then normalize
            norm_height_var = (np.log1p(point_var_z) - 0.5) / 1.2

            # --- 3. APPLY 15CM GRID VOXEL FILTER ---
            voxel_coords = np.floor(xyz_raw / voxel_size).astype(np.int32)
            _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)

            filtered_data = raw_data[unique_indices]
            xyz = np.stack(
                [filtered_data["x"], filtered_data["y"], filtered_data["z"]], axis=1
            )
            intensity = (
                filtered_data["intensity"].astype(np.float32) / 65535.0
            ).reshape(-1, 1)
            labels = filtered_data["sem_class"].astype(np.int64)

            f_rel_height = norm_rel_height[unique_indices].reshape(-1, 1)
            f_height_var = norm_height_var[unique_indices].reshape(-1, 1)
            f_density = norm_density[unique_indices].reshape(-1, 1)

            context_features = np.concatenate(
                [f_rel_height, f_height_var, f_density], axis=-1
            )

            torch.save(
                {
                    "xyz": xyz,
                    "intensity": intensity,
                    "context_features": context_features,
                    "labels": labels,
                },
                dest_path,
            )
            print(
                f"   [{idx + 1}/{len(file_list)}] Standardized & Voxelized: {file_name}"
            )


if __name__ == "__main__":
    run_offline_voxelization()
