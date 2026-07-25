import os
import argparse
from matplotlib import pyplot as plt
import logging
import json
import time
from datetime import datetime
import sys

import torch
import torch.optim as optimizer

from Unet import UNet
from data_load import dataset_prepare_temporal_physics
from utils import *


RATE_LOSS_TYPES = {
    "gradient_mag_rate",
    "gradient_tensor_rate",
    "exch_field_rate",
    "exch_torque_rate",
    "exch_e_density_rate",
    "demag_torque_rate",
    "demag_field_rate",
    "winding_density_rate",
}


def train(epoch, model, optim, train_dataloader):
    """
    Train ONLY on the 64x64 dataset.

    This keeps the same loss definitions as the original multi-resolution
    train.py, but removes the 32x32 and 96x96 branches entirely.
    """
    model.train()

    Loss = AverageMeter()
    Loss1 = AverageMeter()
    Loss2 = AverageMeter()

    total_batches = len(train_dataloader)
    use_temporal_data = args.loss_type in RATE_LOSS_TYPES

    for batch_idx, batch in enumerate(train_dataloader):
        if use_temporal_data:
            x, y, x_prev, y_prev = batch

            x = x.to(device)
            y = y.to(device)
            x_prev = x_prev.to(device)
            y_prev = y_prev.to(device)

            if args.dataug:
                x, y, x_prev, y_prev = dataug_temporal_physics(x, y, x_prev, y_prev)
        else:
            x, y = batch

            x = x.to(device)
            y = y.to(device)

            if args.dataug:
                x, y = dataug(x, y)

        mask = create_mask(x)
        alpha = args.alpha

        # ------------------------------------------------------------------
        # Build the same weighting quantity as the original train.py,
        # but only for size 64.
        # ------------------------------------------------------------------
        if args.loss_type in ("baseline", "torque_mismatch"):
            weight = 1

        elif args.loss_type == "divergence":
            wd = magnetic_divergence(x)

        elif args.loss_type == "gradient":
            wd = gradient_magnitude(x)

        elif args.loss_type == "winding":
            wd, _ = winding_density(x)

        elif args.loss_type == "exchange_energy":
            # Same convention as the original code:
            # all training datasets use Ax = 0.5e-6.
            wd = gradient_magnitude(x) ** 2

        elif args.loss_type == "gradient_mag_rate":
            wd = gradient_magnitude_rate(x, x_prev)

        elif args.loss_type == "gradient_tensor_rate":
            wd = gradient_tensor_rate(x, x_prev)

        elif args.loss_type == "exch_field_rate":
            wd = exchange_field_rate(x, x_prev)

        elif args.loss_type == "exch_torque_rate":
            wd = exchange_torque_rate(x, x_prev)

        elif args.loss_type == "exch_e_density_rate":
            wd = exchange_energy_density_rate(x, x_prev)

        elif args.loss_type == "demag_torque_rate":
            wd = demag_torque_rate(x, x_prev, y, y_prev)

        elif args.loss_type == "demag_field_rate":
            wd = demag_field_rate(y, y_prev)

        elif args.loss_type == "winding_density_rate":
            wd = winding_density_rate(x, x_prev)

        else:
            raise ValueError(f"Unknown loss_type: {args.loss_type}")

        if args.loss_type not in ("baseline", "torque_mismatch"):
            wd = wd.unsqueeze(1)
            weight = 1 + alpha * torch.abs(wd)

            if epoch == 0 and batch_idx == 0:
                wd_stats = {
                    "64": {
                        "min": wd.min().item(),
                        "max": wd.max().item(),
                        "mean": wd.mean().item(),
                        "std": wd.std().item(),
                        "abs_mean": torch.abs(wd).mean().item(),
                        "abs_max": torch.abs(wd).max().item(),
                        "p99": torch.quantile(
                            torch.abs(wd).flatten(), 0.99
                        ).item(),
                    }
                }

                with open(os.path.join(ex_path, f"{args.loss_type}_stats.json"), "w", encoding="utf-8") as f:
                    json.dump(wd_stats, f, indent=4)

        # ------------------------------------------------------------------
        # Size-64 NeuralMAG demagnetizing-field loss.
        # ------------------------------------------------------------------
        pred_y = model(x)

        loss1 = mse(ISLA(pred_y), y) * mask * weight
        loss2 = mse(pred_y, SLA(y)) * mask * weight
        loss = (loss1 + 1000 * loss2).mean()

        # Optional torque-projected auxiliary loss.
        if args.loss_type == "torque_mismatch":
            torque_loss = demag_torque_mismatch_loss(
                x, ISLA(pred_y), y
            )
            main_loss_before_torque = loss.item()
            loss = loss + args.torque_lambda * torque_loss

            if epoch == 0 and batch_idx == 0:
                torque_stats = {
                    "64": torque_loss.item(),
                    "main_loss_64_before_torque": main_loss_before_torque,
                    "total_loss_64_after_torque": loss.item(),
                    "torque_lambda": args.torque_lambda,
                }

                with open(
                    os.path.join(ex_path, "torque_mismatch_stats.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(torque_stats, f, indent=4)

        loss.backward()
        optim.step()
        optim.zero_grad()

        Loss1.update(loss1.mean().item(), x.size(0))
        Loss2.update(loss2.mean().item(), x.size(0))

        # Preserve the original script's reported training-loss convention:
        # the displayed/returned Loss tracks the physical-space MSE term.
        Loss.update(loss1.mean().item(), x.size(0))

        percentage = ((batch_idx + 1) / total_batches) * 100

        status_text = (
            f"\rTrain: epoch {epoch} [{percentage:3.0f}%] | "
            f"Loss {Loss.avg:.1f} | "
            f"Loss64 {Loss1.avg:.1f}/{Loss2.avg:.3f}"
        )
        sys.stdout.write(status_text)
        sys.stdout.flush()

    sys.stdout.write("\n")

    if epoch > 0 and epoch % 10 == 0:
        visualize(
            "train",
            epoch,
            ex_path,
            x,
            y,
            ISLA(pred_y),
            64,
        )

    return Loss.avg


def eval_64(epoch, model, dataloader):
    """Evaluate only on held-out 64x64 data."""
    model.eval()
    Loss = AverageMeter()

    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        mask = create_mask(x)

        with torch.no_grad():
            pred_y = model(x)
            loss = mse(ISLA(pred_y), y) * mask

        Loss.update(loss.mean().item(), x.size(0))

        percentage = ((batch_idx + 1) / total_batches) * 100
        status_text = (
            f"\rEval: epoch {epoch} [{percentage:3.0f}%] | "
            f"Loss64 {Loss.avg:.1f}"
        )
        sys.stdout.write(status_text)
        sys.stdout.flush()

    sys.stdout.write("\n")

    if epoch > 0 and epoch % 10 == 0:
        visualize(
            "eval",
            epoch,
            ex_path,
            x,
            y,
            ISLA(pred_y),
            64,
        )

    return Loss.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NeuralMAG U-Net training on size-64 data only"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="input batch size for 64x64 training (default: 100)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100,
        help="input batch size for 64x64 evaluation (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="learning rate (default: 0.005)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="number of epochs to train (default: 1000)",
    )

    parser.add_argument(
        "--kc",
        type=int,
        default=16,
        help="kernels of first layer (default: 16)",
    )
    parser.add_argument(
        "--inch",
        type=int,
        default=6,
        help="input channels (default: 6)",
    )
    parser.add_argument(
        "--cornum",
        type=int,
        default=1000,
        help="core number (default: 1000)",
    )
    parser.add_argument(
        "--ntest",
        type=int,
        default=20,
        help="number of held-out size-64 cases (default: 20)",
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=300,
        help="training split stop index for size-64 cases (default: 300)",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU used (default: 0)",
    )
    parser.add_argument(
        "--ex",
        type=float,
        default=1.0,
        help="experiment identifier (default: 1.0)",
    )
    parser.add_argument(
        "--dataug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable physical symmetry augmentation",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="weighting coefficient for weighted loss",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="baseline",
        help="loss weighting method",
    )
    parser.add_argument(
        "--torque-lambda",
        type=float,
        default=0.1,
        help="coefficient for torque-mismatch auxiliary loss",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="existing model checkpoint to continue training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader worker processes (default: 8)",
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(device, flush=True)
        print(
            f"GPU reserved : "
            f"{torch.cuda.memory_reserved()/1024**3:.2f} GB"
        )
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed(0)
    elif device.type == "mps":
        torch.mps.manual_seed(0)

    # ----------------------------------------------------------------------
    # Model and optimizer: unchanged from the multi-resolution train.py.
    # A checkpoint from this script has the same U-Net architecture/state_dict
    # format and can therefore be loaded by the existing MH evaluation code.
    # ----------------------------------------------------------------------
    model = UNet(
        kc=args.kc,
        inc=args.inch,
        ouc=args.inch,
    ).to(device)

    if args.model is not None:
        model.load_state_dict(
            torch.load(args.model, map_location=device)
        )
        print(f"Loaded model: {args.model}", flush=True)

    optim = optimizer.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.0001,
    )

    # ----------------------------------------------------------------------
    # ONLY size-64 data are loaded.
    # ----------------------------------------------------------------------
    data_path64_masked1 = (
        "../../../utils/Dataset/rate_change/w64/masked1"
    )
    data_path64_masked2 = (
        "../../../utils/Dataset/rate_change/w64/masked2"
    )
    data_path64_unmasked = (
        "../../../utils/Dataset/rate_change/w64/unmasked"
    )

    data_path64 = [
        data_path64_masked1,
        data_path64_masked2,
        data_path64_unmasked,
    ]

    print("Creating size-64 dataset only", flush=True)

    use_temporal_data = args.loss_type in RATE_LOSS_TYPES

    train_dataset64, test_dataset64 = dataset_prepare_temporal_physics(
        data_path64,
        ntest=args.ntest,
        ntrain=args.ntrain,
        cn=args.cornum,
        include_previous_train=use_temporal_data,
    )

    print_memory(msg="Memory used after preparing datasets")

    print(
        f"size-64 samples: train={len(train_dataset64)}, "
        f"test={len(test_dataset64)}",
        flush=True,
    )
    print(
        f"batch size 64: {args.batch_size}",
        flush=True,
    )

    train_dataloader64 = torch.utils.data.DataLoader(
        dataset=train_dataset64,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    test_dataloader64 = torch.utils.data.DataLoader(
        dataset=test_dataset64,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ex_path = os.path.join(
        f"./{args.loss_type}_contin",
        "size64_only",
        (
            f"{timestamp}_ex{args.ex}_bsz{args.batch_size}_"
            f"lr{args.lr}_Unet_kc{args.kc}_inch{args.inch}"
        ),
    )
    os.makedirs(ex_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(ex_path, "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    logging.info(f"Training resolution: 64x64 only")
    logging.info(f"Total parameters: {num_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        logging.info(f"Device: {device}")

    logging.info(f"PyTorch: {torch.__version__}")

    loss_train_list = []
    loss_test64_list = []
    epoch_list = []

    best_loss = float("inf")
    best_epoch = -1

    start_time = time.time()

    for epoch in range(args.epochs):
        loss_train = train(
            epoch,
            model,
            optim,
            train_dataloader64,
        )

        loss_train_list.append(loss_train)
        logging.info(
            "epoch: {} train_loss64: {:.2f}".format(
                epoch, loss_train
            )
        )

        loss_test64 = eval_64(
            epoch,
            model,
            test_dataloader64,
        )

        if epoch % 100 ==0:
            print_memory(msg=f"Memory used after training epoch number: {epoch}")

        logging.info(
            "Evaluate loss64: {:.1f}".format(loss_test64)
        )

        epoch_list.append(epoch)
        loss_test64_list.append(loss_test64)

        model_path = os.path.join(ex_path, "ckpt")
        os.makedirs(model_path, exist_ok=True)

        # Best checkpoint is now selected strictly by size-64 validation loss.
        if loss_test64 < best_loss:
            print(
                "loss_test64: {:.1f} < best_loss: {:.1f}\n".format(
                    loss_test64, best_loss
                )
            )
            best_loss = loss_test64
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                os.path.join(
                    model_path,
                    f"best_model_{best_loss:.1f}.pt",
                ),
            )

        plt.clf()
        plt.plot(
            epoch_list,
            loss_train_list,
            "r-",
            alpha=1,
            label="train_64",
        )
        plt.plot(
            epoch_list,
            loss_test64_list,
            "g-",
            alpha=1,
            label="test_64",
        )
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss-log")
        plt.yscale("log")
        plt.savefig(
            os.path.join(
                ex_path,
                f"loss_ex{args.ex}.png",
            )
        )

    elapsed = time.time() - start_time

    experiment_info = {
        "training_resolution": 64,
        "alpha": args.alpha,
        "torque_lambda": args.torque_lambda,
        "loss_type": args.loss_type,
        "learning_rate": args.lr,
        "batch_size_64": args.batch_size,
        "test_batch_size_64": args.test_batch_size,
        "epochs": args.epochs,
        "kernel_channels": args.kc,
        "input_channels": args.inch,
        "optimizer": "Adam",
        "betas": [0.9, 0.999],
        "weight_decay": 1e-4,
        "data_augmentation": args.dataug,
        "seed": 0,
        "best_epoch": best_epoch,
        "best_validation_loss_64": best_loss,
        "training_time_seconds": elapsed,
    }

    with open(
        os.path.join(ex_path, "experiment.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(experiment_info, f, indent=4)

    print_memory("Memory usage after training: ")

    logging.info(f"Best epoch: {best_epoch}")
    logging.info(
        f"Best size-64 validation loss: {best_loss}"
    )
    logging.info(
        f"Training time: {elapsed:.2f} seconds"
    )
    logging.info(
        f"Training time: {elapsed/60:.2f} minutes"
    )
    logging.info(
        json.dumps(experiment_info, indent=4)
    )

