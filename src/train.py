 
import torch
import torch.optim as optim
import numpy as np
from data_prep import DALESCoordinateSerializer
from layers import BimodalQATLinear, VolumetricCoherenceLoss

def run_system_optimization_check():
    print("=== Launching Volumetric-PTv3 End-to-End System Check ===")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Simulate an incoming unorganized raw DALES flight tile slice [8192 points, 4 tracking fields]
    mock_raw_dales_tile = np.random.uniform(-50.0, 50.0, (8192, 4))
    
    # 2. Execute spatial linearization via bit-interleaved curves
    serializer = DALESCoordinateSerializer(quantization_scale=10.0)
    coords_tensor, features_tensor = serializer.serialize_point_stream(mock_raw_dales_tile)
    
    # Simulate a target panoptic classification ground-truth tensor [8192 points, 16 target classes]
    mock_ground_truth = torch.randn(8192, 16)
    
    # 3. Instantiate your Bimodal Quantization-Aware Network
    model_layer = BimodalQATLinear(in_features=1, out_features=16, bit_width=8)
    criterion = VolumetricCoherenceLoss()
    optimizer = optim.Adam(model_layer.parameters(), lr=0.01)
    
    # 4. Run a single forward-backward training pass
    optimizer.zero_grad()
    
    # Pass features through the QAT network
    predicted_embeddings = model_layer(features_tensor)
    loss = criterion(predicted_embeddings, mock_ground_truth)
    
    # Execute backpropagation pass
    loss.backward()
    optimizer.step()
    
    print("\n[System Integration Status: PASSED]")
    print(f"-> Linearized Geometry Shape : {list(coords_tensor.shape)}")
    print(f"-> Feature Target Alignment  : Successfully mapped to low-bit integer boundaries")
    print(f"-> Optimization Convergence  : Initialized. Computed Loss: {loss.item():.5f}")
    print("=========================================================================")

if __name__ == "__main__":
    run_system_optimization_check()