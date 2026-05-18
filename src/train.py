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
    
    train_dir = "/content/drive/My Drive/DALES_Processed/train"
    test_dir = "/content/drive/My Drive/DALES_Processed/test"
    
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        print(f"   [Error] Training directory '{train_dir}' is empty or missing.")
        return
    
    dataset = DALESProductionDataset(data_directory=train_dir, max_points_per_block=8192, chunks_per_file=32)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = BimodalQATLinear(in_features=4, out_features=16, bit_width=8)
    
    # 1. Define Inverse Class-Frequency Weights to balance gradients across the 16 classes
    class_weights = torch.tensor([
        0.15,  # Class 0: Ground (Highly Dominant background)
        0.20,  # Class 1: Vegetation (Highly Dominant background)
        0.25,  # Class 2: Buildings (Highly Dominant background)
        1.50,  # Class 3: Wall (Sparse structural feature)
        2.00,  # Class 4: Bridge (Sparse structural feature)
        3.50,  # Class 5: Production Facilities (Rare feature)
        4.00,  # Class 6: Power Lines / Wires (Ultra-Rare geometric profile)
        1.00,  # Class 7: Vehicles (Sparse feature)
        4.50,  # Class 8: High Poles / Supports (Ultra-Rare geometric profile)
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00  # Indices 9-15: Remaining placeholders
    ], dtype=torch.float32)
    
    # 2. Select hardware device engine and transfer elements to active memory registers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    class_weights = class_weights.to(device)
    
    # 3. Initialize Cross-Entropy Loss with integrated categorical class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    evaluator = DALESCrossValidator(num_classes=16)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    last_computed_loss = None

    model.train()
    for epoch in range(1):
        print(f"\n--- Starting Epoch {epoch + 1} / 1 ---")
        for batch_idx, (coords, features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move the streaming feature arrays onto the active GPU hardware device
            features = features.to(device)
            labels = labels.to(device)
            
            # Pass 4D spatial intensity features through your network layers
            predictions = model(features)
            
            # Re-index dimensions to match Cross-Entropy constraints: [Batch * Points, Classes]
            predictions_flattened = predictions.view(-1, 16)
            labels_flattened = torch.clamp(labels, 0, 15).view(-1)

            # Compute backpropagation steps
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
            # Free up model references before kicking off the evaluation sequence
            model = model.cpu()
            _ = evaluator.execute_validation_pass(test_directory=test_dir, weight_path=weight_path)
    else:
        print("-> [Warning] No training batches were executed. Checkpoint bypassed.")

if __name__ == "__main__":
    run_production_training()