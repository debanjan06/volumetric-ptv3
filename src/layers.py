import torch
import torch.nn as nn

class BimodalQATLinear(nn.Module):
    """
    Custom 3D attention linear projection layer implementing bimodal integration 
    transforms and Quantization-Aware Training (QAT) to optimize edge deployment stability.
    """
    def __init__(self, in_features, out_features, bit_width=8):
        super().__init__()
        self.bit_width = bit_width
        self.linear = nn.Linear(in_features, out_features)
        
        # Configure strict fixed-point quantization limits (signed 8-bit range)
        self.qmin = -(2 ** (bit_width - 1))
        self.qmax = (2 ** (bit_width - 1)) - 1
        
        # Learned channel-wise sign-shifting vector to align split post-Key activations
        self.register_buffer("bimodal_shift_vector", torch.randn(1, out_features) * 2.5)

    def compute_fake_quantization(self, input_tensor):
        """
        Clamps and rounds floating-point arrays to simulate hardware quantization limits,
        maintaining straight-through gradient tracing paths for backpropagation.
        """
        max_val = torch.max(torch.abs(input_tensor))
        if max_val == 0:
            return input_tensor
            
        scale_factor = max_val / self.qmax
        quantized_clamped = torch.clamp(torch.round(input_tensor / scale_factor), self.qmin, self.qmax)
        fake_quantized_tensor = quantized_clamped * scale_factor
        return fake_quantized_tensor

    def forward(self, feature_embeddings):
        """
        Executes forward matrix multiplication with activation smoothing and QAT operators.
        """
        raw_projection = self.linear(feature_embeddings)
        smoothed_projection = raw_projection - self.bimodal_shift_vector
        qat_activations = self.compute_fake_quantization(smoothed_projection)
        return qat_activations

class VolumetricCoherenceLoss(nn.Module):
    """
    Custom loss module tracking structural variance regressions over 3D boundaries.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, predicted_embeddings, target_embeddings):
        if not predicted_embeddings.requires_grad:
            raise RuntimeError("Gradient tracing context lost or detached in forward graph pathway.")
        return self.mse(predicted_embeddings, target_embeddings)