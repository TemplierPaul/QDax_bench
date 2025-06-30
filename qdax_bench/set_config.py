# Create configs directory
import os

def create_configs_directory():
    """
    Create the 'configs' directory if it does not exist.
    """
    configs_dir = "configs"
    if not os.path.exists(configs_dir):
        os.makedirs(configs_dir)
        print(f"Created directory: {configs_dir}")
    else:
        print(f"Directory already exists: {configs_dir}")

    # Main configuration file
    main_config_path = os.path.join(configs_dir, "main.yaml")
    if not os.path.exists(main_config_path):
        cfg = """ # Main configuration file for QDax Bench
defaults:
  - algo: me
  - task: debug
  - hydra: local
  - _self_

# Import configurations from QDax Bench
hydra:
  searchpath: 
    - pkg://qdax_bench.configs

num_loops: 10
seed: 42
plots_dir: plots

corrected_metrics: 
  use: false
  evals: 128

wandb:
  project: QDax_ES
  entity: p-templier-imperial-college-london
  use: False
  tag: null
"""
        with open(main_config_path, 'w') as f:
            f.write(cfg)
        print(f"Created main configuration file: {main_config_path}")
    else:
        print(f"Main configuration file already exists: {main_config_path}")

    # Subfolders
    subfolders = ["task", "algo", "hydra"]
    for subfolder in subfolders:
        subfolder_path = os.path.join(configs_dir, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)
        print(f"Created subfolder: {subfolder_path}")


if __name__ == "__main__":
    create_configs_directory()
    print("Directory setup complete.")