#
# MIGRATE
#

# Imports
from huggingface_hub import HfApi, HfFolder, upload_folder
import os
from pathlib import Path
import json


# Generate a README.md dataset card for HuggingFace
def generate_dataset_card(
    output_dir: Path,
    license_name: str = "mit",
    pretty_name: str = "Synthetic Seismic Dataset",
    size_category: str = "100K<n<1M",
    generation_commands: list = None,
    download_example: bool = True
):
    """
    Create a README.md dataset card for HuggingFace.

    :param output_dir: Where to write the README.md
    :param license_name: License type (e.g., 'mit', 'cc-by-4.0')
    :param pretty_name: Human-friendly dataset name
    :param size_category: e.g., '100K<n<1M'
    :param generation_commands: List of commands used to generate the dataset
    :param download_example: Whether to include example download instructions
    """
    # Dataset card header
    yaml_header = f"""---
license: {license_name}
pretty_name: {pretty_name}
task_categories:
  - other
language:
  - multilingual
tags:
  - seismic
  - dispersion-curves
  - synthetic
  - forward-modeling
  - BayHunter
  - inverse-problem
size_categories:
  - {size_category}
---
"""

    # Dataset description
    body = "## Dataset Description\n"
    body += (
        "This dataset contains synthetic seismic models and their corresponding Rayleigh-wave dispersion curves, "
        "generated using forward modeling. It is designed for benchmarking inversion algorithms and training "
        "machine learning models in geophysics.\n\n"
    )

    # Body content
    body += "## Data Structure\n"
    body += "Each sample contains the following fields:\n\n"
    body += "```json\n" + json.dumps({
        "vs": [2.3, 2.7, 3.2],
        "z": [1.2, 2.5, 5.0],
        "vpvs": 1.73,
        "disp_x": [1.0, 2.0, 3.0],
        "disp_y": [3.5, 3.3, 3.1],
        "wave_type": "Rayleigh",
        "velocity_type": "group"
    }, indent=2) + "\n```\n\n"

    # Keep steps for the README
    body += "## Generation Steps\n"
    if generation_commands:
        body += "The dataset was generated using the following commands:\n\n"
        for cmd in generation_commands:
            body += f"```bash\n{cmd}\n```\n"
        # end for
    else:
        body += "_Commands not provided._\n"
    # end if

    # Download instructions
    body += "\n## Download Instructions\n"
    if download_example:
        body += f"You can download the dataset via the 🤗 Hub CLI:\n\n"
        body += f"```bash\nhuggingface-cli download dataset <repo-id> --local-dir ./seismic-dataset\n```\n"
        body += f"\nOr use `datasets` in Python:\n"
        body += f"```python\nfrom datasets import load_dataset\n"
        body += f"ds = load_dataset('<repo-id>', split='train')\n```\n"
    else:
        body += "_Repo ID not provided for download examples._\n"
    # end if

    # Write to README
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(yaml_header + "\n" + body)
    # end with
# end generate_dataset_card


# Upload a dataset directory to HuggingFace Hub
def upload_dataset_to_hf(
    output_dir: str,
    repo_id: str,
    token: str = None,
    repo_type: str = "dataset",
    private: bool = False,
    commit_message: str = "Upload synthetic seismic dataset",
    create_repo: bool = True
):
    """
    Upload a dataset directory to HuggingFace Hub.

    :param output_dir: Path to local dataset directory
    :param repo_id: ID of the repo on the HF Hub (e.g., "nils-schaetti/seismic-dataset")
    :param token: Hugging Face token (or uses default from `huggingface-cli login`)
    :param repo_type: Type of repo ("dataset" by default)
    :param private: Whether the repo is private
    :param commit_message: Commit message for the push
    :param create_repo: Whether to create the repo if it doesn’t exist
    """
    # Get token if not provided
    if token is None:
        token = HfFolder.get_token()
        if token is None:
            raise ValueError("No Hugging Face token found. Please login with `huggingface-cli login` or provide a token.")
        # end if
    # end if

    api = HfApi()

    # Create repo if needed
    if create_repo:
        try:
            api.create_repo(
                repo_id=repo_id,
                token=token,
                repo_type=repo_type,
                private=private,
                exist_ok=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create repo: {e}")
        # end try
    # end if

    # Upload folder
    upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=commit_message
    )

    print(f"Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")
# end upload_dataset_to_hf
