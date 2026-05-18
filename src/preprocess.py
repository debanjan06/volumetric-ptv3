import os
import numpy as np
import torch

def run_offline_voxelization():
    print("=== Launching Production Preprocessing Voxelization Pipeline ===")
    
    # Define source partitions and destination cache directories
    drive_base = "/content/drive/My Drive/DALES_Processed"
    partitions = ["train", "test"]
    voxel_size = 0.15  # 15cm geometric voxel resolution
    
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
            
            # Skip computation if the file has already been processed and cached
            if os.path.exists(dest_path):
                print(f"   [{idx+1}/{len(file_list)}] Found cached asset: {file_name}")
                continue
                
            # Extract binary array offsets
            header_offset = 0
            with open(src_path, 'rb') as f:
                for line in f:
                    header_offset += len(line)
                    if line.decode('ascii', errors='ignore').strip() == "end_header":
                        break
            
            raw_data = np.fromfile(src_path, dtype=ply_dt, offset=header_offset)
            xyz_raw = np.stack([raw_data['x'], raw_data['y'], raw_data['z']], axis=1)
            
            # Execute heavy CPU 3D Grid Voxel Downsampling once
            voxel_coords = np.floor(xyz_raw / voxel_size).astype(np.int32)
            _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
            filtered_data = raw_data[unique_indices]
            
            # Extract geometries from downsampled array structures
            xyz = np.stack([filtered_data['x'], filtered_data['y'], filtered_data['z']], axis=1)
            intensity = (filtered_data['intensity'].astype(np.float32) / 65535.0).reshape(-1, 1)
            labels = filtered_data['sem_class'].astype(np.int64)
            
            # Save as a fast-loading PyTorch tensor dictionary package
            torch.save({
                'xyz': xyz,
                'intensity': intensity,
                'labels': labels
            }, dest_path)
            print(f"   [{idx+1}/{len(file_list)}] Successfully voxelized and cached: {file_name} -> {len(filtered_data)} points")

if __name__ == "__main__":
    run_offline_voxelization()