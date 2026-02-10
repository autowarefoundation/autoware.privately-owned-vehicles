#! /usr/bin/env python3
import argparse
from utils.training import load_yaml, set_global_seed
from training.depth.deeplabv3plus_trainer import DeepLabV3PlusTrainer
from training.depth.unetplusplus_trainer import UnetPlusPlusTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        default="/home/sergey/DEV/AI/AEI/configs/Scene3D_lite.yaml",
        help="Path to training YAML config"
    )
    parser.add_argument("--model", type=str, default="deeplabv3plus", help="Model to train: | deeplabv3plus | unetplusplus")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    seed = cfg.get("experiment", {}).get("seed", 42)
    set_global_seed(seed)

    if args.model == "deeplabv3plus":
        trainer = DeepLabV3PlusTrainer(cfg)
    elif args.model == "unetplusplus":
        trainer = UnetPlusPlusTrainer(cfg)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    
    trainer.run()

if __name__ == "__main__":
    main()
