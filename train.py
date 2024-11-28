import pathlib
import argparse
import json5
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from util.utils import initialize_config, load_config

CKPT_DIR = "checkpoints"
torch.set_float32_matmul_precision("medium")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate a model using PyTorch Lightning.")
    parser.add_argument(
        "--model-config",
        required=True,
        type=pathlib.Path,
        help="Model configuration file (JSON).",
    )
    parser.add_argument(
        "--dataset-config",
        required=True,
        type=pathlib.Path,
        help="Dataset configuration file (JSON).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        help="Path to a pre-trained model checkpoint file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs to use for training. (Default: 1)",
    )
    parser.add_argument("--seed", default=1234, type=int, help="Random seed. (Default: 1234)")
    parser.add_argument(
        "--gradient-clip-val",
        default=1.0,
        type=float,
        help="Gradient clipping threshold. (Default: 1.0)",
    )
    return parser.parse_args()


def setup_experiment_dirs(
    model_config,
    dataset_config,
    model_config_path,
    dataset_config_path,
    checkpoint_path,
):
    """Set up experiment directories and save configuration."""
    dataset_name = dataset_config_path.stem
    model_name = model_config_path.stem
    exp_dir = pathlib.Path("Experiments") / dataset_name / model_config["exp_dir"] / model_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_config": model_config,
        "dataset_config": dataset_config,
        "model_config_path": str(model_config_path),
        "dataset_config_path": str(dataset_config_path),
        "checkpoint_path": str(checkpoint_path),
        "exp_dir": str(exp_dir),
    }

    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json5.dump(config, f, indent=2, sort_keys=False)

    return exp_dir


def get_trainer(exp_dir, trainer_config, gpus, gradient_clip_val, use_wandb=False, wandb_args={}):
    """Configure and return a PyTorch Lightning Trainer instance."""
    checkpoint_dir = exp_dir / CKPT_DIR
    pesq_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val/BER",
        mode="min",
        filename="best",
        save_top_k=1,
        verbose=True,
        save_weights_only=True,
    )
    val_loss_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        verbose=True,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [pesq_checkpoint, val_loss_checkpoint, lr_monitor]

    if gradient_clip_val == None:
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "ddp"
    
    if use_wandb:
        logger = WandbLogger(save_dir=str(exp_dir), **wandb_args)
    else:
        logger = TensorBoardLogger(save_dir=str(exp_dir), name="tensorboard_logs")

    return Trainer(
        default_root_dir=exp_dir,
        strategy=strategy,
        logger=logger,
        precision=trainer_config.get("precision", "32"),
        accelerator="gpu",
        devices=gpus,
        callbacks=callbacks,
        gradient_clip_val=gradient_clip_val,
        max_epochs=trainer_config["max_epochs"],
        check_val_every_n_epoch=trainer_config["check_val_every_n_epoch"],
        accumulate_grad_batches=trainer_config["accumulate_grad_batches"],
    )


def main():
    """Main function."""
    args = parse_args()
    if args.checkpoint_path and args.resume:
        raise ValueError("Resume conflict with preloaded model. Please use one of them.")

    seed_everything(seed=args.seed)

    model_config = load_config(args.model_config)
    dataset_config = load_config(args.dataset_config)
    exp_dir = setup_experiment_dirs(
        model_config,
        dataset_config,
        args.model_config,
        args.dataset_config,
        args.checkpoint_path,
    )

    train_dataset, val_dataset = (
        initialize_config(dataset_config["train_dataset"]),
        initialize_config(dataset_config["validation_dataset"]),
    )
    lightning_module = initialize_config(model_config["lightning_module"], pass_args=False)(model_config)

    if lightning_module.automatic_optimization == False:
        args.gradient_clip_val = None

    use_wandb = model_config.get("use_wandb", False)
    if use_wandb:
        wandb_args = model_config.get("wandb_args")
        wandb_args["group"] = args.dataset_config.stem
    else:
        wandb_args = {}
     
    trainer = get_trainer(exp_dir, model_config["trainer"], args.gpus, args.gradient_clip_val, use_wandb, wandb_args)
    ckpt_path = (exp_dir / CKPT_DIR / "last.ckpt") if args.resume else args.checkpoint_path

    train_dataloader = lightning_module.get_train_dataloader(train_dataset)
    val_dataloader = lightning_module.get_val_dataloader(val_dataset)

    trainer.fit(
        lightning_module,
        train_dataloader,
        val_dataloader,
        ckpt_path=str(ckpt_path) if ckpt_path else None,
    )


if __name__ == "__main__":
    main()
