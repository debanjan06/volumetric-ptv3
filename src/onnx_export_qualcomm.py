import os
import yaml
import torch
from qualcomm_model import VolumetricPTv3QualcommNetwork

def execute_qualcomm_export(config_path: str, output_path: str):
    print(f"[*] Loading execution constraints from unified configuration...")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Extract keys from unified architecture file
    model_cfg = cfg['model']
    hw_cfg = cfg['hardware_compilation']
    shape_cfg = hw_cfg['static_shape']
    
    print(f"[*] Initializing static network tracking graph...")
    model = VolumetricPTv3QualcommNetwork(
        input_dim=model_cfg['input_channels'],
        hidden_dim=model_cfg['hidden_channels'],
        num_classes=model_cfg['num_classes']
    )
    
    # Enforce evaluation constraints to freeze batch statistics
    model.eval()
    
    # Define frozen tensor layout based on hardware target keys
    dummy_tensor = torch.randn(
        shape_cfg['batch_size'],
        shape_cfg['point_count'],
        shape_cfg['feature_dimensions'],
        dtype=torch.float32
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"[*] Compiling to static ONNX graph (Opset {hw_cfg['opset_version']})...")
    torch.onnx.export(
        model,
        dummy_tensor,
        output_path,
        export_params=True,
        opset_version=hw_cfg['opset_version'],
        do_constant_folding=True,
        input_names=["input_point_stream_14d"],
        output_names=["class_scores_16ch"]
    )
    print(f"[+] Qualcomm-compatible asset successfully generated: {output_path}")

if __name__ == "__main__":
    execute_qualcomm_export(
        config_path="config/architecture.yaml",
        output_path="dist/volumetric_ptv3_qualcomm_edge.onnx"
    )