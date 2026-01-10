#!/usr/bin/env python3
"""
Sync model files to HuggingFace.
Run this after pushing code to ensure model files are present.
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_commit, CommitOperationAdd
except ImportError:
    print("Installing huggingface_hub...")
    os.system("pip install huggingface_hub -q")
    from huggingface_hub import HfApi, create_commit, CommitOperationAdd


REPO_ID = "Ranjit0034/finance-entity-extractor"
PROJECT_DIR = Path.home() / "llm-mail-trainer"

# Model files to keep in sync
MODEL_FILES = [
    (PROJECT_DIR / "models/base/phi3-mini/config.json", "config.json"),
    (PROJECT_DIR / "models/adapters/finance-lora-v2/adapter_config.json", "adapter_config.json"),
    (PROJECT_DIR / "models/adapters/finance-lora-v2/adapters.safetensors", "adapters.safetensors"),
]


def check_remote_files(api):
    """Check which model files exist on HuggingFace."""
    try:
        files = api.list_repo_files(REPO_ID)
        return set(files)
    except Exception as e:
        print(f"Error checking remote: {e}")
        return set()


def sync_models():
    """Sync model files to HuggingFace."""
    api = HfApi()
    
    print("🔍 Checking HuggingFace repository...")
    remote_files = check_remote_files(api)
    
    operations = []
    for local_path, repo_path in MODEL_FILES:
        if repo_path not in remote_files:
            if local_path.exists():
                size = local_path.stat().st_size
                print(f"  📤 Will upload: {repo_path} ({size/1024:.1f} KB)")
                operations.append(CommitOperationAdd(
                    path_in_repo=repo_path,
                    path_or_fileobj=str(local_path)
                ))
            else:
                print(f"  ❌ Local file missing: {local_path}")
        else:
            print(f"  ✅ Already exists: {repo_path}")
    
    if operations:
        print(f"\n📤 Uploading {len(operations)} files...")
        try:
            commit = create_commit(
                repo_id=REPO_ID,
                operations=operations,
                commit_message="sync: Restore model files",
                repo_type="model"
            )
            print(f"✅ Uploaded! {commit.commit_url}")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
    else:
        print("\n✅ All model files are present!")


if __name__ == "__main__":
    sync_models()
