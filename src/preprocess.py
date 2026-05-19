import os
import numpy as np
import torch

def run_offline_voxelization():
    print("=== Launching Context-Aware Preprocessing Pipeline ===")
    
    drive_base = r"C:\Users\DEBANJAN SHIL\Documents\volumetric-ptv3\data"
    partitions = ["train", "test"]
    voxel_size = 0.15  # 15cm grid downsampling bounds
    
    ply_dt = np.dtype([
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('intensity', 'i4'), ('sem_class', 'i4'), ('ins_class', 'i4')
    ])
    
    for partition in partitions:
        src_dir = os.path.join(drive_base, partition)
        dest_dir = os.path.join(drive_base, f"{partition}_voxelized")
        os.makedirs(dest_dir, exist_ok=True)
        
        if not os.path.exists(src_dir):
            print(f"   [Warning] Partition path '{src_dir}' missing. Bypassing.")
            continue
            
        file_list = [f for f in os.listdir(src_dir) if f.endswith('.ply')]
        print(f"\nProcessing '{partition}' partition -> Discovered {len(file_list)} assets:")
        
        for idx, file_name in enumerate(file_list):
            src_path = os.path.join(src_dir, file_name)
            dest_path = os.path.join(dest_dir, file_name.replace('.ply', '.pt'))
            
            header_offset = 0
            with open(src_path, 'rb') as f:
                for line in f:
                    header_offset += len(line)
                    if line.decode('ascii', errors='ignore').strip() == "end_header":
                        break
            
            raw_data = np.fromfile(src_path, dtype=ply_dt, offset=header_offset)
            xyz_raw = np.stack([raw_data['x'], raw_data['y'], raw_data['z']], axis=1)
            
            # --- FEATURE ENGINEERING: SPATIAL CONTEXT EXTRACTION ---
            # 1. Compute Macro-Scale Local Density Map (2 meter 2D grids)
            macro_grid_size = 2.0
            macro_coords = np.floor(xyz_raw[:, :2] / macro_grid_size).astype(np.int32)
            macro_keys = macro_coords[:, 0] * 73856093 ^ macro_coords[:, 1] * 19349663
            unique_keys, counts = np.unique(macro_keys, return_counts=True)
            key_to_count = dict(zip(unique_keys, counts))
            raw_densities = np.array([key_to_count[k] for k in macro_keys], dtype=np.float32)
            normalized_densities = (raw_densities - raw_densities.min()) / (raw_densities.max() - raw_densities.min() + 1e-6)
            
            # 2. Compute Micro-Scale Relative Elevation & Height Variance (1 meter 2D grids)
            micro_grid_size = 1.0
            micro_coords = np.floor(xyz_raw[:, :2] / micro_grid_size).astype(np.int32)
            micro_keys = micro_coords[:, 0] * 73856093 ^ micro_coords[:, 1] * 19349663
            
            # Find minimum and variance mapping profiles per spatial column
            unique_micro_keys = np.unique(micro_keys)
            min_z_map = {}
            var_z_map = {}
            
            for k in unique_micro_keys:
                mask = (micro_keys == k)
                z_values = xyz_raw[mask, 2]
                min_z_map[k] = z_values.min()
                var_z_map[k] = z_values.var() if len(z_values) > 1 else 0.0
                
            relative_heights = np.array([xyz_raw[i, 2] - min_z_map[micro_keys[i]] for i in range(len(xyz_raw))], dtype=np.float32)
            height_variances = np.array([var_z_map[micro_keys[i]] for i in range(len(micro_keys))], dtype=np.float32)
            
            # Normalize calculated geometric descriptors
            norm_rel_height = (relative_heights - relative_heights.min()) / (relative_heights.max() - relative_heights.min() + 1e-6)
            norm_height_var = (height_variances - height_variances.min()) / (height_variances.max() - height_variances.min() + 1e-6)
            
            # --- APPLY 15CM GRID VOXEL FILTER ---
            voxel_coords = np.floor(xyz_raw / voxel_size).astype(np.int32)
            _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
            
            filtered_data = raw_data[unique_indices]
            xyz = np.stack([filtered_data['x'], filtered_data['y'], filtered_data['z']], axis=1)
            intensity = (filtered_data['intensity'].astype(np.float32) / 65535.0).reshape(-1, 1)
            labels = filtered_data['sem_class'].astype(np.int64)
            
            # Filter corresponding engineered context attributes to preserve indices
            f_rel_height = norm_rel_height[unique_indices].reshape(-1, 1)
            f_height_var = norm_height_var[unique_indices].reshape(-1, 1)
            f_density = normalized_densities[unique_indices].reshape(-1, 1)
            
            # 3. Compile the 7D Contextual Feature Bundle
            # Shape: [Points, 7] -> [X, Y, Z, Intensity, Rel_Height, Height_Var, Density]
            context_features = np.concatenate([f_rel_height, f_height_var, f_density], axis=-1)
            
            torch.save({
                'xyz': xyz,
                'intensity': intensity,
                'context_features': context_features,
                'labels': labels
            }, dest_path)
            print(f"   [{idx+1}/{len(file_list)}] Voxelized & Context-Encoded: {file_name}")

if __name__ == "__main__":
    run_offline_voxelization()