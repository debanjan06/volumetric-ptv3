import os
from huggingface_hub import HfApi


def upload_pipeline_assets_to_hf():
    # Targets your exact repository name locked by the DOI
    repo_id = "Debanjan24/volumetric-ptv3-qat-8bit"

    # Matches the exact local file name we just copied
    local_checkpoint = "volumetric_ptv3_qat_8bit (4).pth"
    local_config = "config/architecture.yaml"

    if not os.path.exists(local_config):
        print("[-] Error: Run this script directly from the repository root directory.")
        return

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("[-] Error: Missing HF_TOKEN environment variable.")
        return

    api = HfApi()
    print(f"[*] Commencing chunked LFS stream to DOI-locked repository: {repo_id}...")

    # Upload the Main Trained Model Weights Checkpoint
    if os.path.exists(local_checkpoint):
        print("[*] Uploading main 128-channel QAT trained weights checkpoint...")
        api.upload_file(
            path_or_fileobj=local_checkpoint,
            path_in_repo="volumetric_ptv3_qat_8bit (4).pth",
            repo_id=repo_id,
            token=hf_token,
        )
        print("[+] Checkpoint upload complete.")
    else:
        print(f"[-] Error: Local checkpoint file '{local_checkpoint}' not found.")


if __name__ == "__main__":
    upload_pipeline_assets_to_hf()
