import os
import torch
import numpy as np
import plotly.graph_objects as pgo
from data_prep import DALESProductionDataset
from train import BimodalQATLinear

def generate_interactive_demo():
    print("=== Launching Volumetric-PTv3 Live Demo Asset Generator ===")
    
    weight_path = "models/volumetric_ptv3_qat_8bit.pth"
    test_dir = "/content/drive/MyDrive/DALES_Processed/test_voxelized"
    if not os.path.exists(test_dir):
        test_dir = "/content/drive/My Drive/DALES_Processed/test_voxelized"
        
    if not os.path.exists(weight_path):
        print("[Error] Production weights missing. Cannot compile demo.")
        return

    # 1. Load a real validation scene sample
    dataset = DALESProductionDataset(data_directory=test_dir, max_points_per_block=8192, chunks_per_file=1)
    coords, features, labels = dataset[0]  # Grab a fresh 14D block
    
    # 2. Reconstruct model and run inference
    model = BimodalQATLinear(in_features=14, out_features=16, bit_width=8)
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        # Shape output: [1, 8192, 14]
        predictions = model(features.unsqueeze(0))
        pred_classes = torch.argmax(predictions.squeeze(0), dim=-1).numpy()
    
    xyz = coords.numpy()
    ground_truth = labels.numpy()
    
    # DALES Standard Simplified Color Palette (Ground, Veg, Buildings)
    color_map = {
        0: '#808080',  # Unknown/Unclassified -> Gray
        1: '#8B4513',  # Ground -> Brown
        2: '#00FF00',  # Vegetation -> Green
        3: '#FFD700',  # Cars -> Gold
        4: '#0000FF',  # Trucks -> Blue
        5: '#FF0000',  # Buildings -> Red
    }
    
    def get_color_list(class_array):
        return [color_map.get(c, '#FFFFFF') for c in class_array]

    # 3. Create interactive 3D subplots using Plotly
    fig = pgo.Figure()
    
    # Left Pane: Ground Truth Layout
    fig.add_trace(pgo.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode='markers',
        marker=dict(size=3, color=get_color_list(ground_truth), opacity=0.8),
        name='Ground Truth Profiles'
    ))
    
    # Save standalone HTML file
    os.makedirs("demo", exist_ok=True)
    output_html = "demo/dales_inference_visualization.html"
    fig.write_html(output_html)
    
    print("\n======================= DEMO SUCCESS =======================")
    print(f"   -> Interactive 3D scene serialized to: {output_html}")
    print("   -> Download this HTML file to your laptop and double-click to open!")
    print("==============================================================")

if __name__ == "__main__":
    generate_interactive_demo()