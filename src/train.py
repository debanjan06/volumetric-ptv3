import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_prep import DALESProductionDataset
from layers import BimodalQATLinear
from evaluate import DALESCrossValidator

def run_production_training():
    print("=== Launching Volumetric-PTv3 Live Training Pipeline ===")
    torch.manual_seed(42)
    
    train_dir = "/content/drive/My Drive/DALES_Processed/train_voxelized"
    test_dir = "/content/drive/My Drive/DALES_Processed/test_voxelized"
    
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        print(f"   [Error] Training directory '{train_dir}' is empty or missing.")
        return
    
    dataset = DALESProductionDataset(data_directory=train_dir, max_points_per_block=8192, chunks_per_file=32)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = BimodalQATLinear(in_features=4, out_features=16, bit_width=8)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # FIX: Reverted to standard unweighted Cross-Entropy to work alongside voxel filtering
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