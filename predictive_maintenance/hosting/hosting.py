from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload to Hugging Face
repo_id = "nairsuj/predictive-maintenance"
repo_type = "space"

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="predictive_maintenance/deployment",     # the local folder containing your files
    repo_id=repo_id,                              # the target repo
    repo_type=repo_type,                          # dataset, model, or space
    path_in_repo="",                              # optional: subfolder path inside the repo
)
