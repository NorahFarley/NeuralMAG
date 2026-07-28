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
    "exchange_field_rate",
    "exchange_torque_rate",
    "exchange_energy_density_rate",
    "demag_torque_rate",
    "demag_field_rate",
    "winding_density_rate",
}


def _uses_alpha(loss_type):
    return loss_type not in ("baseline", "torque_mismatch")


def _loss_parameter_tag(args):
    """Tag only the coefficient that actually participates in this loss."""
    if args.loss_type == "torque_mismatch":
        return f"torque_lambda{args.torque_lambda:g}"
    if _uses_alpha(args.loss_type):
        return f"alpha{args.alpha:g}"
    return ""


def _loss_descriptor(args):
    if args.loss_type == "torque_mismatch":
        return f"loss={args.loss_type} torque_lambda={args.torque_lambda:g}"
    if _uses_alpha(args.loss_type):
        return f"loss={args.loss_type} alpha={args.alpha:g}"
    return f"loss={args.loss_type}"


def _format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _configure_epoch_logger(ex_path):
    """
    One record per completed epoch.

    The StreamHandler writes to stderr, which restores the old Slurm behavior:
    epoch summaries go to the job error log, while normal print() output goes to
    the job output log. A duplicate copy is saved inside the experiment folder.
    """
    logger = logging.getLogger("neuralmag_epoch")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    file_handler = logging.FileHandler(os.path.join(ex_path, "training.log"), mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


_observed_memory_peak = {
    "ram_total_with_workers_gb": 0.0,
    "ram_main_gb": 0.0,
    "ram_workers_gb": 0.0,
    "gpu_allocated_gb": 0.0,
    "gpu_reserved_gb": 0.0,
    "gpu_max_allocated_gb": 0.0,
    "gpu_max_reserved_gb": 0.0,
}


def _sample_memory_peak():
    """Sample main-process + DataLoader-worker RAM without printing anything."""
    try:
        stats = get_memory_stats()
    except Exception:
        return
    for key in _observed_memory_peak:
        value = float(stats.get(key, 0.0))
        if value > _observed_memory_peak[key]:
            _observed_memory_peak[key] = value


def _write_memory_record(ex_path, stage, epoch=None):
    """Append a compact JSON line so memory can be inspected while the job runs."""
    try:
        stats = get_memory_stats()
        _sample_memory_peak()
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": stage,
            "epoch": epoch,
            **stats,
            "observed_peak_so_far": dict(_observed_memory_peak),
        }
        with open(os.path.join(ex_path, "memory_usage.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\\n")
    except Exception as exc:
        print(f"Warning: could not record memory snapshot: {exc}", flush=True)


def _slurm_environment():
    keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
        "SLURM_JOB_NODELIST",
    )
    return {key: os.environ.get(key) for key in keys if os.environ.get(key) is not None}


def _print_active_loss_settings(args):
    print("Training configuration:", flush=True)
    print(f"  loss type: {args.loss_type}", flush=True)
    if args.loss_type == "torque_mismatch":
        print(f"  torque lambda: {args.torque_lambda:g}", flush=True)
    elif _uses_alpha(args.loss_type):
        print(f"  alpha: {args.alpha:g}", flush=True)
    print(f"  learning rate: {args.lr:g}", flush=True)
    print(f"  epochs: {args.epochs}", flush=True)


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

        elif args.loss_type == "exchange_field_rate":
            wd = exchange_field_rate(x, x_prev)

        elif args.loss_type == "exchange_torque_rate":
            wd = exchange_torque_rate(x, x_prev)

        elif args.loss_type == "exchange_energy_density_rate":
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

        if sys.stdout.isatty():
            percentage = ((batch_idx + 1) / total_batches) * 100
            status_text = (
                f"\rTrain: epoch {epoch} [{percentage:3.0f}%] | "
                f"Loss {Loss.avg:.1f} | "
                f"Loss64 {Loss1.avg:.1f}/{Loss2.avg:.3f}"
            )
            sys.stdout.write(status_text)
            sys.stdout.flush()

        if batch_idx == 0 or (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches:
            _sample_memory_peak()

    if sys.stdout.isatty():
        sys.stdout.write("\n")
        sys.stdout.flush()

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

        if sys.stdout.isatty():
            percentage = ((batch_idx + 1) / total_batches) * 100
            status_text = (
                f"\rEval: epoch {epoch} [{percentage:3.0f}%] | "
                f"Loss64 {Loss.avg:.1f}"
            )
            sys.stdout.write(status_text)
            sys.stdout.flush()

        if batch_idx == 0 or (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches:
            _sample_memory_peak()

    if sys.stdout.isatty():
        sys.stdout.write("\n")
        sys.stdout.flush()

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
    parser = argparse.ArgumentParser(description="NeuralMAG U-Net training on size-64 data only")
    parser.add_argument("--batch-size", type=int, default=100, help="input batch size for 64x64 training (default: 100)")
    parser.add_argument("--test-batch-size", type=int, default=100, help="input batch size for 64x64 evaluation (default: 100)")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate (default: 0.005)")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs to train (default: 1000)")
    parser.add_argument("--kc", type=int, default=16, help="kernels of first layer (default: 16)")
    parser.add_argument("--inch", type=int, default=6, help="input channels (default: 6)")
    parser.add_argument("--cornum", type=int, default=1000, help="core number (default: 1000)")
    parser.add_argument("--ntest", type=int, default=20, help="number of held-out size-64 cases (default: 20)")
    parser.add_argument("--ntrain", type=int, default=300, help="training split stop index for size-64 cases (default: 300)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU used (default: 0)")
    parser.add_argument("--ex", type=float, default=1.0, help="experiment identifier (default: 1.0)")
    parser.add_argument("--dataug", action=argparse.BooleanOptionalAction, default=True, help="enable physical symmetry augmentation")
    parser.add_argument("--alpha", type=float, default=0.5, help="weighting coefficient for alpha-weighted losses")
    parser.add_argument("--loss_type", type=str, default="baseline", help="loss weighting method")
    parser.add_argument("--torque-lambda", type=float, default=0.1, help="coefficient for torque-mismatch auxiliary loss")
    parser.add_argument("--model", type=str, default=None, help="existing model checkpoint to continue training")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker processes per DataLoader (default: 8)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    coefficient_tag = _loss_parameter_tag(args)
    folder_bits = [timestamp]
    if coefficient_tag:
        folder_bits.append(coefficient_tag)
    folder_bits.extend([
        f"ex{args.ex}",
        f"bsz{args.batch_size}",
        f"lr{args.lr}",
        f"Unet_kc{args.kc}",
        f"inch{args.inch}",
    ])
    ex_path = os.path.join(f"./{args.loss_type}_contin", "size64_only", "_".join(folder_bits))
    os.makedirs(ex_path, exist_ok=True)
    epoch_logger = _configure_epoch_logger(ex_path)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
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

    _print_active_loss_settings(args)
    print(f"  training resolution: 64x64 only", flush=True)
    print(f"  batch size: {args.batch_size}", flush=True)
    print(f"  test batch size: {args.test_batch_size}", flush=True)
    print(f"  DataLoader workers per loader: {args.num_workers}", flush=True)
    print(f"  output directory: {ex_path}", flush=True)
    print(f"  device: {device}", flush=True)

    # Save the launch parameters immediately, before the potentially long data-loading step.
    active_coefficient = None
    if args.loss_type == "torque_mismatch":
        active_coefficient = {"name": "torque_lambda", "value": args.torque_lambda}
    elif _uses_alpha(args.loss_type):
        active_coefficient = {"name": "alpha", "value": args.alpha}
    launch_parameters = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "output_directory": ex_path,
        "loss_type": args.loss_type,
        "active_loss_coefficient": active_coefficient,
        "arguments_as_parsed": vars(args),
        "device": str(device),
        "slurm": _slurm_environment(),
    }
    with open(os.path.join(ex_path, "run_parameters.json"), "w", encoding="utf-8") as f:
        json.dump(launch_parameters, f, indent=4)
    print(f"Launch parameters saved to {os.path.join(ex_path, 'run_parameters.json')}", flush=True)

    model = UNet(kc=args.kc, inc=args.inch, ouc=args.inch).to(device)
    if args.model is not None:
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"Loaded model: {args.model}", flush=True)

    optim = optimizer.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0001)
    print_memory("Memory after model initialization")
    _write_memory_record(ex_path, "after_model_initialization")

    data_path64 = [
        "../../../utils/Dataset/rate_change/w64/masked1",
        "../../../utils/Dataset/rate_change/w64/masked2",
        "../../../utils/Dataset/rate_change/w64/unmasked",
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

    print_memory("Memory after preparing datasets")
    _write_memory_record(ex_path, "after_dataset_preparation")
    print(f"size-64 samples: train={len(train_dataset64)}, test={len(test_dataset64)}", flush=True)

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

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_settings = {
        "loss_type": args.loss_type,
        "uses_alpha": _uses_alpha(args.loss_type),
        "alpha": args.alpha if _uses_alpha(args.loss_type) else None,
        "uses_torque_lambda": args.loss_type == "torque_mismatch",
        "torque_lambda": args.torque_lambda if args.loss_type == "torque_mismatch" else None,
    }
    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "output_directory": ex_path,
        "training_resolution": 64,
        "loss": loss_settings,
        "optimizer": {
            "name": "Adam",
            "learning_rate": args.lr,
            "betas": [0.9, 0.999],
            "weight_decay": 1e-4,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size_64": args.batch_size,
            "test_batch_size_64": args.test_batch_size,
            "num_workers_per_dataloader": args.num_workers,
            "simultaneous_train_dataloaders": 1,
            "simultaneous_eval_dataloaders": 1,
            "data_augmentation": args.dataug,
            "temporal_previous_state_loaded_for_training": use_temporal_data,
            "cornum": args.cornum,
            "ntest_cases_requested": args.ntest,
            "ntrain_split_stop": args.ntrain,
            "random_seed_torch": 0,
        },
        "samples": {
            "train_64": len(train_dataset64),
            "test_64": len(test_dataset64),
        },
        "model": {
            "kc": args.kc,
            "input_channels": args.inch,
            "total_parameters": num_params,
            "trainable_parameters": trainable_params,
            "continued_from_checkpoint": args.model,
        },
        "runtime": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "pytorch_version": torch.__version__,
            "slurm": _slurm_environment(),
        },
        "dataset_paths": {"64": data_path64},
    }
    with open(os.path.join(ex_path, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)
    print(f"Full reproducibility configuration saved to {os.path.join(ex_path, 'run_config.json')}", flush=True)
    print(f"Per-epoch timing/loss log saved to {os.path.join(ex_path, 'training.log')}", flush=True)
    print(f"Memory snapshots saved to {os.path.join(ex_path, 'memory_usage.jsonl')}", flush=True)

    loss_train_list = []
    loss_test64_list = []
    epoch_list = []
    best_loss = float("inf")
    best_epoch = -1
    epoch_durations = []
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        loss_train = train(epoch, model, optim, train_dataloader64)
        loss_test64 = eval_64(epoch, model, test_dataloader64)

        loss_train_list.append(loss_train)
        loss_test64_list.append(loss_test64)
        epoch_list.append(epoch)

        model_path = os.path.join(ex_path, "ckpt")
        os.makedirs(model_path, exist_ok=True)
        new_best = loss_test64 < best_loss
        if new_best:
            previous_best = best_loss
            best_loss = loss_test64
            best_epoch = epoch
            best_model_path = os.path.join(model_path, f"best_model_{best_loss:.1f}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(
                f"NEW BEST MODEL | epoch {epoch} | validation64={best_loss:.1f} "
                f"< previous={previous_best:.1f} | saved: {best_model_path}",
                flush=True,
            )

        epoch_time = time.time() - epoch_start
        epoch_durations.append(epoch_time)
        elapsed = time.time() - start_time
        recent_avg_epoch = sum(epoch_durations[-5:]) / len(epoch_durations[-5:])
        remaining_epochs = args.epochs - epoch - 1
        eta_seconds = recent_avg_epoch * remaining_epochs
        projected_finish = datetime.fromtimestamp(time.time() + eta_seconds).strftime("%Y-%m-%d %H:%M:%S")

        epoch_logger.info(
            f"epoch={epoch} ({epoch + 1}/{args.epochs}) | {_loss_descriptor(args)} | "
            f"train64={loss_train:.2f} | val64={loss_test64:.1f} | "
            f"best={best_loss:.1f}@epoch{best_epoch} new_best={'yes' if new_best else 'no'} | "
            f"epoch_time={_format_duration(epoch_time)} | elapsed={_format_duration(elapsed)} | "
            f"ETA={_format_duration(eta_seconds)} | projected_finish={projected_finish}"
        )

        _write_memory_record(ex_path, "epoch_complete", epoch=epoch)

        plt.clf()
        plt.plot(epoch_list, loss_train_list, "r-", alpha=1, label="train_64")
        plt.plot(epoch_list, loss_test64_list, "g-", alpha=1, label="test_64")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss-log")
        plt.yscale("log")
        plt.savefig(os.path.join(ex_path, f"loss_ex{args.ex}.png"))

    elapsed = time.time() - start_time
    experiment_info = {
        **run_config,
        "result": {
            "best_epoch": best_epoch,
            "best_validation_loss_64": best_loss,
            "training_time_seconds": elapsed,
            "training_time_human": _format_duration(elapsed),
            "observed_memory_peak": dict(_observed_memory_peak),
        },
    }
    with open(os.path.join(ex_path, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_info, f, indent=4)

    print_memory("Memory after training")
    _write_memory_record(ex_path, "training_complete")
    print(
        f"TRAINING COMPLETE | best epoch={best_epoch} | best validation64={best_loss:.1f} | "
        f"elapsed={_format_duration(elapsed)}",
        flush=True,
    )

