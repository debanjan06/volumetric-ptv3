import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_prep import DALESProductionDataset
from evaluate import DALESCrossValidator

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


def run_production_training():
    print("=== Launching Volumetric-PTv3 Live Training Pipeline ===")
    torch.manual_seed(42)
    
    # Check for both standard Colab mount paths to maintain environment compatibility
    train_dir = "/content/drive/MyDrive/DALES_Processed/train_voxelized"
    test_dir = "/content/drive/MyDrive/DALES_Processed/test_voxelized"
    
    if not os.path.exists(train_dir):
        train_dir = "/content/drive/My Drive/DALES_Processed/train_voxelized"
        test_dir = "/content/drive/My Drive/DALES_Processed/test_voxelized"
    
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        print(f"   [Error] Training directory '{train_dir}' is empty or missing.")
        return
    
    dataset = DALESProductionDataset(data_directory=train_dir, max_points_per_block=8192, chunks_per_file=32)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Instantiating the upgraded 14-input feature capacity model
    model = BimodalQATLinear(in_features=14, out_features=16, bit_width=8)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    evaluator = DALESCrossValidator(num_classes=16)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    last_computed_loss = None

    model.train()
    for epoch in range(1):
        print(f"\n--- Starting Epoch {epoch + 1} / 1 ---")
        for batch_idx, (coords, features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            features = features.to(device)
            labels = labels.to(device)
            
            predictions = model(features)
            
            predictions_flattened = predictions.view(-1, 16)
            labels_flattened = torch.clamp(labels, 0, 15).view(-1)

            loss = criterion(predictions_flattened, labels_flattened)
            loss.backward()
            optimizer.step()
            
            last_computed_loss = loss.item()
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == 464:
                print(f"   [Batch {batch_idx + 1} / 464] Current Loss Convergence: {last_computed_loss:.5f}")

    print("\n==========================================================")
    print("-> Training Epoch Completed. Serializing Final Checkpoint...")
    
    if last_computed_loss is not None:
        checkpoint_directory = "models"
        os.makedirs(checkpoint_directory, exist_ok=True)
        weight_path = os.path.join(checkpoint_directory, "volumetric_ptv3_qat_8bit.pth")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'final_loss': last_computed_loss
        }, weight_path)
        print(f"-> Production model checkpoint compiled successfully at: {weight_path}")
        
        if os.path.exists(test_dir) and os.listdir(test_dir):
            model = model.cpu()
            _ = evaluator.execute_validation_pass(test_directory=test_dir, weight_path=weight_path)
    else:
        print("-> [Warning] No training batches were executed. Checkpoint bypassed.")

if __name__ == "__main__":
    run_production_training()