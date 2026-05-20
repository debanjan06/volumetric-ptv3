import os

def inspect_ply_header(file_path):
    print(f"\n=== Reading PLY Architectural Header: {os.path.basename(file_path)} ===")
    try:
        with open(file_path, 'rb') as f:
            header_lines = []
            for _ in range(50):  # Read the first 50 lines to capture the full header block
                line = f.readline().decode('ascii', errors='ignore').strip()
                header_lines.append(line)
                if line == "end_header":
                    break
        
        # Display the properties defined inside the file
        for line in header_lines:
            if any(keyword in line for keyword in ["element vertex", "property", "format", "end_header"]):
                print(f"   {line}")
                
    except Exception as e:
        print(f"   [Extraction Error]: {e}")
    print("==========================================================")

if __name__ == "__main__":
    # Create the training path structure locally if it doesn't exist
    train_dir = os.path.join("data", "train")
    
    if os.path.exists(train_dir):
        ply_files = [f for f in os.listdir(train_dir) if f.endswith('.ply')]
        if ply_files:
            target_sample = os.path.join(train_dir, ply_files[0])
            inspect_ply_header(target_sample)
        else:
            print(f"\n[Status] Place your downloaded .ply files inside '{train_dir}/'")
    else:
        print(f"\n[Status] Local directory structure not found. Creating '{train_dir}' layout now.")
        os.makedirs(train_dir, exist_ok=True)
        print(f"-> Please drop a sample training .ply file into: {train_dir}") 
