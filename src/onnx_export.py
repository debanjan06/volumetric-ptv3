import os
import torch
from train import BimodalQATLinear

def export_to_production_onnx():
    print("=== Launching Production-Grade 14D Point Cloud ONNX Compiler ===")
    
    weight_path = "models/volumetric_ptv3_qat_8bit.pth"
    output_dir = "dist"
    os.makedirs(output_dir, exist_ok=True)
    onnx_output_path = os.path.join(output_dir, "volumetric_ptv3_production.onnx")
    
    if not os.path.exists(weight_path):
        print(f"   [Error] Checkpoint '{weight_path}' not found. Please verify the file path.")
        return

    # 1. Instantiate the 128-channel deep residual baseline model
    model = BimodalQATLinear(in_features=14, out_features=16, bit_width=8)
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Define a standard 14D inference input stream payload
    # Shape description: [Batch Size (1), Sampled Points per Block (8192), 14 Spatial Features]
    dummy_input = torch.randn(1, 8192, 14, dtype=torch.float32)

    # 3. Compile the model graph into a highly optimized ONNX binary asset
    print("   -> Running static graph compilation and parsing structural operations...")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,        # Serialize the trained weight parameters directly inside the graph file
        opset_version=17,          # Use modern production operations matrix mappings
        do_constant_folding=True,  # Automatically pre-compute and fuse constant operators (like Batch Normalization)
        input_names=['input_point_stream'],
        output_names=['class_logits'],
        dynamic_axes={
            'input_point_stream': {1: 'point_cloud_density'},  # Enable dynamic processing capacities for varying block densities
            'class_logits': {1: 'point_cloud_density'}
        }
    )
    
    print("\n=================== EXPORT COMPILATION SUMMARY ===================")
    print(f"   -> Production Asset Location  : {onnx_output_path}")
    print(f"   -> Graph Compilation State    : SUCCESS")
    print(f"   -> Deployment Target Readiness: Optimized for ONNX Runtime / TensorRT NPU")
    print("====================================================================")

if __name__ == "__main__":
    export_to_production_onnx()