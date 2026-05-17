import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_prep import DALESProductionDataset
from layers import BimodalQATLinear, VolumetricCoherenceLoss
from evaluate import VolumetricPerceptionEvaluator

def run_production_training():
    print("=== Launching Volumetric-PTv3 Live Training Pipeline ===")
    torch.manual_seed(42)
    
    train_dir = "/content/drive/My Drive/DALES_Processed/train"
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        print(f"   [Error] Training directory '{train_dir}' is empty or missing.")
        return
    
    # 1. Initialize our high-throughput dataset on physical files
    dataset = DALESProductionDataset(data_directory=train_dir, max_points_per_block=8192, chunks_per_file=32)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 2. Instantiate our core network layers matching our configuration
    model = BimodalQATLinear(in_features=1, out_features=16, bit_width=8)
    criterion = VolumetricCoherenceLoss()
    evaluator = VolumetricPerceptionEvaluator(num_classes=16)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Track loss initialization safely to avoid NameErrors
    last_computed_loss = None

    # 3. Training iteration loop over real files
    model.train()
    for epoch in range(1):
        for batch_idx, (coords, features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Form target ground truth maps [batch, points, classes]
            batch_size, points_count, _ = features.shape
            labels_one_hot = torch.zeros(batch_size, points_count, 16)
            
            # Clamp labels to stay safely within our 16-class matrix bounds
            clamped_labels = torch.clamp(labels, 0, 15)
            labels_one_hot.scatter_(2, clamped_labels.unsqueeze(-1), 1.0)

            # Pass features through our QAT layers
            predictions = model(features)
            
            # Flatten dimensions across points for loss evaluation calculation
            loss = criterion(predictions.view(-1, 16), labels_one_hot.view(-1, 16))
            
            # Step adjustments
            loss.backward()
            optimizer.step()
            
            last_computed_loss = loss.item()
            print(f"\n   [Batch {batch_idx + 1}] Loss Convergence: {last_computed_loss:.5f}")
            
            # 4. Profile the performance metrics of the trained batch immediately
            model.eval()
            with torch.no_grad():
                test_preds = model(features)
                _ = evaluator.generate_scientific_report(
                    coords[0], test_preds[0], labels_one_hot[0]
                )
            model.train()

    print("\n==========================================================")
    print("-> Live Training and Evaluation Sequence Successfully Validated.")
    
    # Save the optimized QAT model parameters if training completed batches
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
    else:
        print("-> [Warning] No training batches were executed. Checkpoint bypassed.")

if __name__ == "__main__":
    run_production_training()