import os
import numpy as np
import torch
import torch.nn as nn
from data_prep import DALESProductionDataset

class ResidualQATBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        return self.relu2(out)


class BimodalQATLinear(nn.Module):
    def __init__(self, in_features=14, out_features=16, bit_width=8):
        super().__init__()
        self.bimodal_shift_vector = nn.Parameter(torch.zeros(1, out_features))
        
        hidden_dim = 128
        self.feature_block = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            ResidualQATBlock(hidden_dim),
            ResidualQATBlock(hidden_dim),
            nn.Linear(hidden_dim, out_features)
        )
        
    def forward(self, x):
        orig_shape = x.shape
        if len(orig_shape) == 3:
            x = x.reshape(-1, orig_shape[-1])
            
        out = self.feature_block(x)
        out = out + self.bimodal_shift_vector
        
        if len(orig_shape) == 3:
            out = out.reshape(orig_shape[0], orig_shape[1], -1)
        return out


class DALESCrossValidator:
    def __init__(self, num_classes=16):
        self.num_classes = num_classes

    def execute_validation_pass(self, test_directory, weight_path):
        print("\n=== Launching Volumetric-PTv3 Cross-Validation Pipeline ===")
        
        dataset = DALESProductionDataset(data_directory=test_directory, max_points_per_block=8192, chunks_per_file=32)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        
        print(f"   -> Tracked Pre-Voxelized Assets: {len(dataset.file_list)}")
        print(f"   -> Loading network weights from: {weight_path}")
        
        # Upgraded to match the 14D input and 128-channel residual framework
        model = BimodalQATLinear(in_features=14, out_features=self.num_classes, bit_width=8)
        
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("   -> Ingesting test assets from Drive partition...")
        
        total_intersection = np.zeros(self.num_classes)
        total_union = np.zeros(self.num_classes)
        total_points_evaluated = 0
        
        with torch.no_grad():
            for coords, features, labels in test_loader:
                predictions = model(features)
                predictions_flattened = predictions.view(-1, self.num_classes)
                labels_flattened = torch.clamp(labels, 0, self.num_classes - 1).view(-1)
                
                pred_classes = torch.argmax(predictions_flattened, dim=-1).numpy()
                target_classes = labels_flattened.numpy()
                
                total_points_evaluated += len(target_classes)
                
                for c in range(self.num_classes):
                    pred_mask = (pred_classes == c)
                    target_mask = (target_classes == c)
                    
                    intersection = np.logical_and(pred_mask, target_mask).sum()
                    union = np.logical_or(pred_mask, target_mask).sum()
                    
                    total_intersection[c] += intersection
                    total_union[c] += union
        
        # Calculate mean Intersection over Union safely avoiding division by zero
        iou_per_class = []
        for c in range(self.num_classes):
            if total_union[c] == 0:
                continue
            iou_per_class.append(total_intersection[c] / total_union[c])
            
        generalized_miou = np.mean(iou_per_class) * 100.0
        loss_floor = checkpoint.get('final_loss', 0.0)
        
        print("\n================ CROSS-VALIDATION REPORT ================")
        print(f"   -> Total Evaluated 3D Points   : {total_points_evaluated} points")
        print(f"   -> Out-of-Sample Generalized mIoU: {generalized_miou:.2f}%")
        print(f"   -> Checked Checkpoint Loss Floor: {loss_floor:.5f}")
        print("==========================================================")
        
        return generalized_miou