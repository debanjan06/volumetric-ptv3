import torch
import torch.nn as nn

class BimodalQATLinear(nn.Module):
    """
    Upgraded Multi-Stage Deep 3D Feature Projection Block.
    Implements non-linear layers and batch normalization to eliminate underfitting.
    """
    def __init__(self, in_features, out_features, bit_width=8):
        super().__init__()
        self.bit_width = bit_width
        
        # Multi-stage projection network to extract complex spatial patterns
        self.feature_block = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )
        
        self.qmin = -(2 ** (bit_width - 1))
        self.qmax = (2 ** (bit_width - 1)) - 1
        self.register_buffer("bimodal_shift_vector", torch.randn(1, out_features) * 2.5)

    def compute_fake_quantization(self, input_tensor):
        max_val = torch.max(torch.abs(input_tensor))
        if max_val == 0:
            return input_tensor
            
        scale_factor = max_val / self.qmax
        quantized_clamped = torch.clamp(torch.round(input_tensor / scale_factor), self.qmin, self.qmax)
        fake_quantized_tensor = quantized_clamped * scale_factor
        return fake_quantized_tensor

    def forward(self, feature_embeddings):
        # Reshape point matrices to apply 1D Batch Normalization layers
        batch_size, points_count, features_dim = feature_embeddings.shape
        flat_features = feature_embeddings.view(-1, features_dim)
        
        # Pass elements through the deep non-linear block
        flat_projections = self.feature_block(flat_features)
        raw_projection = flat_projections.view(batch_size, points_count, -1)
        
        smoothed_projection = raw_projection - self.bimodal_shift_vector
        qat_activations = self.compute_fake_quantization(smoothed_projection)
        return qat_activations

class VolumetricCoherenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, predicted_embeddings, target_embeddings):
        if not predicted_embeddings.requires_grad:
            raise RuntimeError("Gradient tracing context lost or detached in forward graph pathway.")
        return self.mse(predicted_embeddings, target_embeddings)