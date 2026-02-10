import argparse
from utils.training import load_yaml, set_global_seed
from training.lane_detection.deeplabv3plus_trainer import DeepLabV3PlusTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        default="/home/sergey/DEV/AI/AEI/configs/lanes/egolanes.yaml",
        help="Path to lane detection training YAML config"
    )
    parser.add_argument("--model", type=str, default="egolanes", help="Model to train: egolanes or deeplabv3plus")

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    seed = cfg.get("experiment", {}).get("seed", 42)
    set_global_seed(seed)

    if args.model == "deeplabv3plus":
        trainer = DeepLabV3PlusTrainer(cfg)
    else:
        raise ValueError(f"Unknown model type: {args.model}")


    trainer.run()


if __name__ == "__main__":
    main()
