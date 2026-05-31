import os
import torch
import numpy as np
import plotly.graph_objects as pgo
from data_prep import DALESProductionDataset
from train import BimodalQATLinear


def render_complete_urban_scene():
    print("=== Compiling Final Full-Scene 3D Volumetric Demo ===")

    weight_path = "models/volumetric_ptv3_qat_8bit.pth"
    test_dir = "/content/drive/MyDrive/DALES_Processed/test_voxelized"
    if not os.path.exists(test_dir):
        test_dir = "/content/drive/My Drive/DALES_Processed/test_voxelized"

    if not os.path.exists(weight_path):
        print("[Error] Production weights missing. Cannot compile full scene.")
        return

    # 1. Load ALL chunks belonging to the first validation asset file to stitch them together
    # By setting chunks_per_file to 32, we pull the entire structural layout of the file
    dataset = DALESProductionDataset(
        data_directory=test_dir, max_points_per_block=8192, chunks_per_file=32
    )

    # Instantiate your 128-channel deep network
    model = BimodalQATLinear(in_features=14, out_features=16, bit_width=8)
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_xyz = []
    all_predictions = []

    print(
        "   -> Stitching aerial chunks and streaming full-scale inference across the neighborhood..."
    )
    # Process the first 32 blocks to reconstruct one complete contiguous physical location
    for idx in range(32):
        coords, features, labels = dataset[idx]

        with torch.no_grad():
            predictions = model(features.unsqueeze(0))
            pred_classes = torch.argmax(predictions.squeeze(0), dim=-1).numpy()

        all_xyz.append(coords.numpy())
        all_predictions.append(pred_classes)

    # Concatenate all individual blocks into a single comprehensive point cloud canvas
    canvas_xyz = np.vstack(all_xyz)
    canvas_preds = np.concatenate(all_predictions)

    # Color mapping for standard infrastructure components
    color_map = {
        0: "#808080",  # Unknown -> Gray
        1: "#553311",  # Ground/Streets -> Dark Brown
        2: "#22AA22",  # Trees/Vegetation -> Forest Green
        3: "#FFD700",  # Cars -> Gold
        4: "#0000FF",  # Trucks -> Blue
        5: "#CC3333",  # Buildings/Roofs -> Deep Red
    }

    canvas_colors = [color_map.get(c, "#FFFFFF") for c in canvas_preds]

    print(
        f"   -> Assembling final scene graphics layer ({len(canvas_xyz)} total 3D points)..."
    )

    # 2. Render the complete stitched environment
    fig = pgo.Figure()
    fig.add_trace(
        pgo.Scatter3d(
            x=canvas_xyz[:, 0],
            y=canvas_xyz[:, 1],
            z=canvas_xyz[:, 2],
            mode="markers",
            marker=dict(
                size=2,  # Smaller marker size to maintain clarity over dense urban spaces
                color=canvas_colors,
                opacity=0.9,
            ),
            name="Predicted Urban Infrastructure",
        )
    )

    # Apply clean layout controls optimized for wide aerial scans
    fig.update_layout(
        title="Final Volumetric-PTv3 Integrated Urban Scene (Dayton, Ohio)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    os.makedirs("demo", exist_ok=True)
    output_html = "demo/final_complete_urban_scene.html"
    fig.write_html(output_html)

    print("\n======================= FINAL SCENE COMPILED =======================")
    print(f"   -> Complete stitched 3D scene serialized to: {output_html}")
    print("   -> Download this file to view your full model segmentation layout!")
    print("====================================================================")


if __name__ == "__main__":
    render_complete_urban_scene()
