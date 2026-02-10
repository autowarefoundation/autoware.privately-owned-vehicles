#! /usr/bin/env python3
import argparse
from utils.training import load_yaml, set_global_seed

from Models.training.scene3d_lite_trainer import Scene3DLite_trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        default="/home/sergey/DEV/AI/AEI/configs/Scene3D_lite.yaml",
        help="Path to training YAML config"
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    seed = cfg.get("experiment", {}).get("seed", 42)
    set_global_seed(seed)

    trainer = Scene3DLite_trainer(cfg)
    
    trainer.run()

if __name__ == "__main__":
    main()
