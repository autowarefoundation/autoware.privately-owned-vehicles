import argparse
from utils.training import load_yaml, set_global_seed
from Models.training.ego_lanes_lite_trainer import EgoLanesLiteTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        default="/home/sergey/DEV/AI/AEI/configs/lanes/egolanes.yaml",
        help="Path to lane detection training YAML config"
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    seed = cfg.get("experiment", {}).get("seed", 42)
    set_global_seed(seed)

    trainer = EgoLanesLiteTrainer(cfg)


    trainer.run()


if __name__ == "__main__":
    main()
