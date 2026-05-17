import os
import torch
from torch.utils.data import DataLoader
from data_prep import DALESProductionDataset
from layers import BimodalQATLinear

class DALESCrossValidator:
    """
    Loads saved 8-bit QAT checkpoints and executes cross-validation pipelines
    over unseen DALES testing PLY slices to certify real-world generalization.
    """
    def __init__(self, num_classes=16):
        self.num_classes = num_classes

    def execute_validation_pass(self, test_directory, weight_path):
        print("\n=== Launching Volumetric-PTv3 Cross-Validation Pipeline ===")
        
        if not os.path.exists(test_directory) or not os.listdir(test_directory):
            print(f"   [Error] Testing directory '{test_directory}' is empty or missing.")
            return

        if not os.path.exists(weight_path):
            print(f"   [Error] Saved model checkpoint path '{weight_path}' not found.")
            return

        # 1. Stream data from the test directory layout
        dataset = DALESProductionDataset(data_directory=test_directory, max_points_per_block=8192, chunks_per_file=32)
        test_loader = DataLoader(dataset, batch_size=2, shuffle=False)

        # 2. Reconstruct the model structure and map saved weights
        model = BimodalQATLinear(in_features=1, out_features=16, bit_width=8)
        checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        total_intersection = torch.zeros(self.num_classes)
        total_union = torch.zeros(self.num_classes)
        evaluated_points = 0

        print(f"   -> Loading network weights from: {weight_path}")
        print(f"   -> Ingesting {len(dataset.file_list)} test assets from Drive partition...")

        # 3. Process the validation loader matrix blocks
        with torch.no_grad():
            for batch_idx, (coords, features, labels) in enumerate(test_loader):
                predictions = model(features)
                
                pred_classes = torch.argmax(predictions, dim=2).view(-1)
                target_classes = labels.view(-1)
                
                evaluated_points += target_classes.numel()

                # Calculate IoU per semantic class across the batch arrays
                for c in range(self.num_classes):
                    pred_mask = (pred_classes == c)
                    target_mask = (target_classes == c)
                    
                    total_intersection[c] += torch.sum(pred_mask & target_mask).item()
                    total_union[c] += torch.sum(pred_mask | target_mask).item()

        # 4. Generate global evaluation metrics
        iou_per_class = []
        for c in range(self.num_classes):
            if total_union[c] == 0:
                # If a class is not present in the test slice, exclude it from penalizing the model
                continue
            iou_per_class.append((total_intersection[c] / total_union[c]).item())

        mean_iou = sum(iou_per_class) / len(iou_per_class) if iou_per_class else 0.0

        print("\n================ CROSS-VALIDATION REPORT ================")
        print(f"   -> Total Evaluated 3D Points   : {evaluated_points} points")
        print(f"   -> Out-of-Sample Generalized mIoU: {mean_iou * 100:.2f}%")
        print(f"   -> Checked Checkpoint Loss Floor: {checkpoint['final_loss']:.5f}")
        print("==========================================================")
        return mean_iou

if __name__ == "__main__":
    # Test path variables matching your Google Drive mounting schema
    test_dir = "/content/drive/My Drive/DALES_Processed/test"
    saved_weights = "models/volumetric_ptv3_qat_8bit.pth"
    
    # Standard dummy parameters to trigger fallback verification loops if local execution is handled
    if not os.path.exists(test_dir):
        test_dir = os.path.join("data", "test")
        
    validator = DALESCrossValidator(num_classes=16)
    _ = validator.execute_validation_pass(test_directory=test_dir, weight_path=saved_weights)