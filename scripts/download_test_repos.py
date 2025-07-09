#!/usr/bin/env python3
"""Download test repositories for benchmarking."""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Test repositories to download
TEST_REPOS = [
    ("Django", "https://github.com/django/django.git", "stable/5.0.x"),
    ("Flask", "https://github.com/pallets/flask.git", "3.0.x"),
    ("FastAPI", "https://github.com/tiangolo/fastapi.git", "master"),
    ("Requests", "https://github.com/psf/requests.git", "main"),
]


def download_repo(name: str, url: str, branch: str, target_dir: Path) -> bool:
    """Download a repository using git clone.
    
    Args:
        name: Repository name
        url: Git repository URL
        branch: Branch to checkout
        target_dir: Target directory for the repository
        
    Returns:
        True if successful, False otherwise
    """
    repo_path = target_dir / name.lower()
    
    if repo_path.exists():
        print(f"Repository {name} already exists at {repo_path}")
        return True
    
    print(f"Downloading {name} from {url} (branch: {branch})...")
    
    try:
        # Clone with shallow depth to save space and time
        cmd = [
            "git", "clone",
            "--depth", "1",
            "--branch", branch,
            url,
            str(repo_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error cloning {name}:")
            print(result.stderr)
            return False
            
        print(f"Successfully downloaded {name}")
        return True
        
    except Exception as e:
        print(f"Error downloading {name}: {e}")
        return False


def main():
    """Main function to download all test repositories."""
    # Create test_repos directory
    test_repos_dir = Path(__file__).parent / "test_repos"
    test_repos_dir.mkdir(exist_ok=True)
    
    print(f"Downloading test repositories to {test_repos_dir}")
    print("=" * 60)
    
    success_count = 0
    failed_repos = []
    
    for name, url, branch in TEST_REPOS:
        if download_repo(name, url, branch, test_repos_dir):
            success_count += 1
        else:
            failed_repos.append(name)
    
    print("=" * 60)
    print(f"Downloaded {success_count}/{len(TEST_REPOS)} repositories successfully")
    
    if failed_repos:
        print(f"Failed to download: {', '.join(failed_repos)}")
        sys.exit(1)
    else:
        print("All repositories downloaded successfully!")
        
    # Create a summary file
    summary_path = test_repos_dir / "README.md"
    with open(summary_path, 'w') as f:
        f.write("# Test Repositories\n\n")
        f.write("This directory contains test repositories for benchmarking the AST library.\n\n")
        f.write("## Repositories\n\n")
        for name, url, branch in TEST_REPOS:
            repo_path = test_repos_dir / name.lower()
            if repo_path.exists():
                f.write(f"- **{name}** ({branch} branch): `{repo_path}`\n")
        f.write("\n## Note\n\n")
        f.write("These are shallow clones (depth=1) to save space. ")
        f.write("If you need full history, run `git fetch --unshallow` in the repository.\n")


if __name__ == "__main__":
    main()