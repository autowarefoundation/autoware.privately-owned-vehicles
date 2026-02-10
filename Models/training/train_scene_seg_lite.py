#! /usr/bin/env python3
import argparse
from utils.training import load_yaml, set_global_seed
from Models.training.scene_seg_lite_trainer import SceneSegLiteTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        default="Models/configs/SceneSegLite.yaml",
        help="Path to training YAML config"
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    seed = cfg.get("experiment", {}).get("seed", 42)
    set_global_seed(seed)

    trainer = SceneSegLiteTrainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()
