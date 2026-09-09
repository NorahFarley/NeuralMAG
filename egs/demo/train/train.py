import os
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
import logging

import json
import time
from datetime import datetime
import sys

import torch
import torch.optim as optimizer

from Unet import UNet
from data_load import (dataset_prepare, dataset_prepare_temporal_physics,)
from utils import * 

COMBINED_RATE_LOSS = "exchange_torque_gradient_tensor_rate"
RATE_ERROR_LOSS = "demag_field_rate_error"


RATE_LOSS_TYPES = {
    "gradient_mag_rate",
    "gradient_tensor_rate",
    "exchange_field_rate",
    "exchange_torque_rate",
    "exchange_energy_density_rate",
    "demag_torque_rate",
    "demag_field_rate",
    "winding_density_rate",
    COMBINED_RATE_LOSS,
    RATE_ERROR_LOSS,
}


def _uses_alpha(loss_type):
    return loss_type not in (
        "baseline",
        "torque_mismatch",
        COMBINED_RATE_LOSS,
        RATE_ERROR_LOSS,
    )


def _loss_parameter_tag(args):
    """Tag only the coefficient that actually participates in this loss."""
    if args.loss_type == "torque_mismatch":
        return f"torque_lambda{args.torque_lambda:g}"
    if args.loss_type == COMBINED_RATE_LOSS:
        return (
            f"alpha_torque{args.alpha_torque:g}_"
            f"alpha_grad{args.alpha_grad:g}"
        )
    if args.loss_type == RATE_ERROR_LOSS:
        return f"rate_error_lambda{args.rate_error_lambda:g}"
    if _uses_alpha(args.loss_type):
        return f"alpha{args.alpha:g}"
    return ""


def _loss_descriptor(args):
    if args.loss_type == "torque_mismatch":
        return f"loss={args.loss_type} torque_lambda={args.torque_lambda:g}"
    if args.loss_type == COMBINED_RATE_LOSS:
        return (
            f"loss={args.loss_type} alpha_torque={args.alpha_torque:g} "
            f"alpha_grad={args.alpha_grad:g}"
        )
    if args.loss_type == RATE_ERROR_LOSS:
        return (
            f"loss={args.loss_type} "
            f"rate_error_lambda={args.rate_error_lambda:g}"
        )
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
    elif args.loss_type == COMBINED_RATE_LOSS:
        print(f"  exchange-torque-rate alpha: {args.alpha_torque:g}", flush=True)
        print(f"  gradient-tensor-rate alpha: {args.alpha_grad:g}", flush=True)
    elif args.loss_type == RATE_ERROR_LOSS:
        print(f"  demag-field-rate-error lambda: {args.rate_error_lambda:g}", flush=True)
    elif _uses_alpha(args.loss_type):
        print(f"  alpha: {args.alpha:g}", flush=True)
    print(f"  learning rate: {args.lr:g}", flush=True)
    print(f"  epochs: {args.epochs}", flush=True)


def train(epoch, model, optim, train_dataloader1, train_dataloader2, train_dataloader3):
    model.train()
    Loss = AverageMeter()
    Loss11 = AverageMeter()
    Loss12 = AverageMeter()
    Loss21 = AverageMeter()
    Loss22 = AverageMeter()
    Loss31 = AverageMeter()
    Loss32 = AverageMeter()
    RateErrorLoss = AverageMeter()

    total_batches = min(len(train_dataloader1), len(train_dataloader2), len(train_dataloader3))
    use_temporal_data = (args.loss_type in RATE_LOSS_TYPES)

    #strat to train
    for batch_idx, (batch1, batch2, batch3) in enumerate(zip(train_dataloader1, train_dataloader2, train_dataloader3)):

        if use_temporal_data:
            x1, y1, x1_prev, y1_prev = batch1
            x2, y2, x2_prev, y2_prev = batch2
            x3, y3, x3_prev, y3_prev = batch3 

            x1 = x1.to(device)
            y1 = y1.to(device)
            x1_prev = x1_prev.to(device)
            y1_prev = y1_prev.to(device)

            x2 = x2.to(device)
            y2 = y2.to(device)
            x2_prev = x2_prev.to(device)
            y2_prev = y2_prev.to(device)

            x3 = x3.to(device)
            y3 = y3.to(device)
            x3_prev = x3_prev.to(device)
            y3_prev = y3_prev.to(device)  

            if args.dataug:
                x1, y1, x1_prev, y1_prev = dataug_temporal_physics(x1, y1, x1_prev, y1_prev)
                x2, y2, x2_prev, y2_prev = dataug_temporal_physics(x2, y2, x2_prev, y2_prev)
                x3, y3, x3_prev, y3_prev = dataug_temporal_physics(x3, y3, x3_prev, y3_prev)            

        else:      
            x1, y1 = batch1
            x2, y2 = batch2
            x3, y3 = batch3

            x1 = x1.to(device)
            y1 = y1.to(device)

            x2 = x2.to(device)
            y2 = y2.to(device)

            x3 = x3.to(device)
            y3 = y3.to(device)

            if args.dataug==True:
                x1, y1 = dataug(x1,y1)
                x2, y2 = dataug(x2,y2)
                x3, y3 = dataug(x3,y3) 

        mask1, mask2, mask3 = create_mask(x1), create_mask(x2), create_mask(x3)

        alpha = args.alpha

        if args.loss_type in ("baseline", "torque_mismatch", RATE_ERROR_LOSS):
            weight1 = 1
            weight2 = 1
            weight3 = 1   
            
        elif args.loss_type == "divergence":
            wd1 = magnetic_divergence(x1)
            wd2 = magnetic_divergence(x2)
            wd3 = magnetic_divergence(x3)

        elif args.loss_type == "gradient":
            wd1 = gradient_magnitude(x1)
            wd2 = gradient_magnitude(x2)
            wd3 = gradient_magnitude(x3)
        
        elif args.loss_type == "winding":
            wd1, _ = winding_density(x1)
            wd2, _ = winding_density(x2)
            wd3, _ = winding_density(x3)   

        elif args.loss_type == "exchange_energy": # exchange energy density
            # All training datasets use same Ax:0.5e-6 so it is not included in loss function
            wd1 = gradient_magnitude(x1)**2 
            wd2 = gradient_magnitude(x2)**2
            wd3 = gradient_magnitude(x3)**2

        elif args.loss_type == "gradient_mag_rate":
            wd1 = gradient_magnitude_rate(x1, x1_prev)
            wd2 = gradient_magnitude_rate(x2, x2_prev)
            wd3 = gradient_magnitude_rate(x3, x3_prev)

        elif args.loss_type == "gradient_tensor_rate":
            wd1 = gradient_tensor_rate(x1, x1_prev)
            wd2 = gradient_tensor_rate(x2, x2_prev)
            wd3 = gradient_tensor_rate(x3, x3_prev)

        elif args.loss_type == "exchange_field_rate":
            wd1 = exchange_field_rate(x1, x1_prev)            
            wd2 = exchange_field_rate(x2, x2_prev)
            wd3 = exchange_field_rate(x3, x3_prev)

        elif args.loss_type == "exchange_torque_rate":
            wd1 = exchange_torque_rate(x1, x1_prev)
            wd2 = exchange_torque_rate(x2, x2_prev)
            wd3 = exchange_torque_rate(x3, x3_prev)

        elif args.loss_type == "exchange_energy_density_rate":
            wd1 = exchange_energy_density_rate(x1, x1_prev)
            wd2 = exchange_energy_density_rate(x2, x2_prev)
            wd3 = exchange_energy_density_rate(x3, x3_prev)

        elif args.loss_type == "demag_torque_rate":
            wd1 = demag_torque_rate(x1, x1_prev, y1, y1_prev)
            wd2 = demag_torque_rate(x2, x2_prev, y2, y2_prev)
            wd3 = demag_torque_rate(x3, x3_prev, y3, y3_prev)

        elif args.loss_type == "demag_field_rate":
            wd1 = demag_field_rate(y1, y1_prev)
            wd2 = demag_field_rate(y2, y2_prev)
            wd3 = demag_field_rate(y3, y3_prev)

        elif args.loss_type == "winding_density_rate":
            wd1 = winding_density_rate(x1, x1_prev)
            wd2 = winding_density_rate(x2, x2_prev)
            wd3 = winding_density_rate(x3, x3_prev)

        elif args.loss_type == COMBINED_RATE_LOSS:
            torque_rate1 = exchange_torque_rate(x1, x1_prev)
            torque_rate2 = exchange_torque_rate(x2, x2_prev)
            torque_rate3 = exchange_torque_rate(x3, x3_prev)

            grad_rate1 = gradient_tensor_rate(x1, x1_prev)
            grad_rate2 = gradient_tensor_rate(x2, x2_prev)
            grad_rate3 = gradient_tensor_rate(x3, x3_prev)

        else:
            raise ValueError(f"Unknown loss_type: {args.loss_type}")

        if args.loss_type == COMBINED_RATE_LOSS:
            torque_rate1 = torch.abs(torque_rate1).unsqueeze(1)
            torque_rate2 = torch.abs(torque_rate2).unsqueeze(1)
            torque_rate3 = torch.abs(torque_rate3).unsqueeze(1)

            grad_rate1 = torch.abs(grad_rate1).unsqueeze(1)
            grad_rate2 = torch.abs(grad_rate2).unsqueeze(1)
            grad_rate3 = torch.abs(grad_rate3).unsqueeze(1)

            weight1 = (
                1
                + args.alpha_torque * torque_rate1
                + args.alpha_grad * grad_rate1
            )
            weight2 = (
                1
                + args.alpha_torque * torque_rate2
                + args.alpha_grad * grad_rate2
            )
            weight3 = (
                1
                + args.alpha_torque * torque_rate3
                + args.alpha_grad * grad_rate3
            )

            if epoch == 0 and batch_idx == 0:
                def tensor_stats(tensor):
                    return {
                        "min": tensor.min().item(),
                        "max": tensor.max().item(),
                        "mean": tensor.mean().item(),
                        "std": tensor.std().item(),
                        "p99": torch.quantile(tensor.flatten(), 0.99).item(),
                    }

                combined_stats = {
                    "formula": (
                        "1 + alpha_torque * abs(exchange_torque_rate) "
                        "+ alpha_grad * abs(gradient_tensor_rate)"
                    ),
                    "alpha_torque": args.alpha_torque,
                    "alpha_grad": args.alpha_grad,
                }

                for size, torque_rate, grad_rate, weight in (
                    ("32", torque_rate1, grad_rate1, weight1),
                    ("64", torque_rate2, grad_rate2, weight2),
                    ("96", torque_rate3, grad_rate3, weight3),
                ):
                    combined_stats[size] = {
                        "exchange_torque_rate": tensor_stats(torque_rate),
                        "gradient_tensor_rate": tensor_stats(grad_rate),
                        "weighted_exchange_torque_contribution": tensor_stats(
                            args.alpha_torque * torque_rate
                        ),
                        "weighted_gradient_tensor_contribution": tensor_stats(
                            args.alpha_grad * grad_rate
                        ),
                        "combined_weight": tensor_stats(weight),
                    }

                file_path = os.path.join(
                    ex_path, f"{args.loss_type}_stats.json"
                )
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(combined_stats, f, indent=4)

        elif args.loss_type not in ("baseline", "torque_mismatch", RATE_ERROR_LOSS):
            wd1 = wd1.unsqueeze(1)
            wd2 = wd2.unsqueeze(1)
            wd3 = wd3.unsqueeze(1)

            weight1 = 1 + alpha * torch.abs(wd1)
            weight2 = 1 + alpha * torch.abs(wd2)
            weight3 = 1 + alpha * torch.abs(wd3)
    
            if epoch == 0 and batch_idx == 0:
                wd_stats = {"32": {"min": wd1.min().item(),
                                   "max": wd1.max().item(),
                                   "mean": wd1.mean().item(),
                                    "std": wd1.std().item(),
                                    "abs_mean": torch.abs(wd1).mean().item(),
                                    "abs_max": torch.abs(wd1).max().item(),
                                    "p99": torch.quantile(torch.abs(wd1).flatten(),0.99).item()},

                            "64": {"min": wd2.min().item(),
                                    "max": wd2.max().item(),
                                    "mean": wd2.mean().item(),
                                    "std": wd2.std().item(),
                                    "abs_mean": torch.abs(wd2).mean().item(),
                                    "abs_max": torch.abs(wd2).max().item(),
                                    "p99": torch.quantile(torch.abs(wd2).flatten(),0.99).item()},

                            "96": {"min": wd3.min().item(),
                                    "max": wd3.max().item(),
                                    "mean": wd3.mean().item(),
                                    "std": wd3.std().item(),
                                    "abs_mean": torch.abs(wd3).mean().item(),
                                    "abs_max": torch.abs(wd3).max().item(),
                                    "p99": torch.quantile(torch.abs(wd3).flatten(),0.99).item()}}

                file_path = os.path.join(ex_path, f"{args.loss_type}_stats.json")

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(wd_stats, f, indent=4)


       # weight1 = 1 + alpha * wd1 + beta * wd1_2 #winding/gradient

        if args.loss_type == RATE_ERROR_LOSS:
            # Process one resolution at a time and backpropagate immediately.
            # This gives the same summed gradient as one large backward pass,
            # while avoiding retention of six U-Net activation graphs at once.
            optim.zero_grad()

            def train_rate_error_resolution(x, y, x_prev, y_prev, mask):
                pred_y = model(x)
                pred_y_prev = model(x_prev)

                # Standard NeuralMAG loss for the current state.
                base_physical = mse(ISLA(pred_y), y) * mask
                base_log = mse(pred_y, SLA(y)) * mask
                base_loss = (base_physical + 1000 * base_log).mean()

                # Error in the demagnetizing-field change per saved transition.
                # A common physical dt would scale both differences equally and
                # can therefore be absorbed into rate_error_lambda.
                pred_delta_physical = ISLA(pred_y) - ISLA(pred_y_prev)
                true_delta_physical = y - y_prev
                pred_delta_log = pred_y - pred_y_prev
                true_delta_log = SLA(y) - SLA(y_prev)

                rate_physical = (
                    mse(pred_delta_physical, true_delta_physical) * mask
                )
                rate_log = mse(pred_delta_log, true_delta_log) * mask
                rate_loss = (rate_physical + 1000 * rate_log).mean()

                total_loss = base_loss + args.rate_error_lambda * rate_loss
                total_loss.backward()

                return (
                    pred_y.detach(),
                    base_physical,
                    base_log,
                    base_loss,
                    rate_physical,
                    rate_log,
                    rate_loss,
                    total_loss,
                )

            (
                pred_y1, loss11, loss12, loss1,
                rate_physical1, rate_log1, rate_loss1, total_loss1,
            ) = train_rate_error_resolution(x1, y1, x1_prev, y1_prev, mask1)
            (
                pred_y2, loss21, loss22, loss2,
                rate_physical2, rate_log2, rate_loss2, total_loss2,
            ) = train_rate_error_resolution(x2, y2, x2_prev, y2_prev, mask2)
            (
                pred_y3, loss31, loss32, loss3,
                rate_physical3, rate_log3, rate_loss3, total_loss3,
            ) = train_rate_error_resolution(x3, y3, x3_prev, y3_prev, mask3)

            optim.step()
            optim.zero_grad()

            rate_loss_batch = (rate_loss1 + rate_loss2 + rate_loss3) / 3
            RateErrorLoss.update(
                rate_loss_batch.item(),
                x1.size(0) + x2.size(0) + x3.size(0),
            )

            if epoch == 0 and batch_idx == 0:
                rate_error_stats = {
                    "formula": (
                        "L_total = L_current + rate_error_lambda * "
                        "L_delta_Hdemag"
                    ),
                    "rate_definition": "difference per saved transition",
                    "rate_error_lambda": args.rate_error_lambda,
                    "32": {
                        "base_loss": loss1.item(),
                        "rate_physical_mse": rate_physical1.mean().item(),
                        "rate_log_mse": rate_log1.mean().item(),
                        "rate_loss": rate_loss1.item(),
                        "total_loss": total_loss1.item(),
                    },
                    "64": {
                        "base_loss": loss2.item(),
                        "rate_physical_mse": rate_physical2.mean().item(),
                        "rate_log_mse": rate_log2.mean().item(),
                        "rate_loss": rate_loss2.item(),
                        "total_loss": total_loss2.item(),
                    },
                    "96": {
                        "base_loss": loss3.item(),
                        "rate_physical_mse": rate_physical3.mean().item(),
                        "rate_log_mse": rate_log3.mean().item(),
                        "rate_loss": rate_loss3.item(),
                        "total_loss": total_loss3.item(),
                    },
                }
                with open(
                    os.path.join(ex_path, "demag_field_rate_error_stats.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(rate_error_stats, f, indent=4)

        else:
            #data1 size32
            pred_y1 = model(x1)
            loss11 = mse( ISLA(pred_y1), y1 ) * mask1 * weight1 #enlarge-scale predict Hd to label Hd 
            loss12 = mse( pred_y1, SLA(y1) ) * mask1  * weight1 #shrink-scale label Hd to predict Hd
            loss1 = ((loss11 + 1000 * loss12)).mean()

            #data2 size64
            pred_y2 = model(x2)
            loss21 = mse(ISLA(pred_y2), y2) * mask2 * weight2
            loss22 = mse(pred_y2, SLA(y2)) * mask2 * weight2
            loss2 = ((loss21 + 1000 * loss22)).mean()
            
            #data3 size96
            pred_y3 = model(x3)
            loss31 = mse(ISLA(pred_y3), y3) * mask3 * weight3
            loss32 = mse(pred_y3, SLA(y3)) * mask3 * weight3
            loss3 = ((loss31 + 1000 * loss32)).mean()

        if args.loss_type == "torque_mismatch":
            torque_loss1 = demag_torque_mismatch_loss(x1, ISLA(pred_y1), y1)
            torque_loss2 = demag_torque_mismatch_loss(x2, ISLA(pred_y2), y2)
            torque_loss3 = demag_torque_mismatch_loss(x3, ISLA(pred_y3), y3)

            loss1 = (loss1 + args.torque_lambda * torque_loss1)
            loss2 = (loss2 + args.torque_lambda * torque_loss2)
            loss3 = (loss3 + args.torque_lambda * torque_loss3)

            if epoch == 0 and batch_idx == 0:
                torque_stats = {
                    "32": torque_loss1.item(),
                    "64": torque_loss2.item(),
                    "96": torque_loss3.item(),
                    "main_loss_32": loss1.item(),
                    "main_loss_64": loss2.item(),
                    "main_loss_96": loss3.item(),
                    "torque_lambda": args.torque_lambda}

                with open(os.path.join(ex_path, "torque_mismatch_stats.json"), "w", encoding="utf-8") as f: json.dump(torque_stats, f, indent=4)

        if args.loss_type != RATE_ERROR_LOSS:
            loss = loss1 + loss2 + loss3

            loss.backward()
            optim.step()
            optim.zero_grad()
        
        Loss11.update( loss11.mean().item(),  x1.size(0) )
        Loss12.update( loss12.mean().item(),  x1.size(0) )
        Loss21.update( loss21.mean().item(),  x2.size(0) )
        Loss22.update( loss22.mean().item(),  x2.size(0) )
        Loss31.update( loss31.mean().item(),  x3.size(0) )
        Loss32.update( loss32.mean().item(),  x3.size(0) )
        Loss.update( ((loss11.mean()+loss21.mean()+loss31.mean())/3).item(),  x1.size(0)+x2.size(0)+x3.size(0) )

        # Keep the live batch progress only for a real interactive terminal.
        # Slurm redirects stdout to a file, so printing every batch there creates
        # enormous logs. In Slurm, the epoch summary is written once per epoch
        # through the stderr logger in main().
        if sys.stdout.isatty():
            percentage = ((batch_idx + 1) / total_batches) * 100
            status_text = (
                f"\rTrain: epoch {epoch} [{percentage:3.0f}%] | Loss {Loss.avg:.1f} | "
                f"Loss1 {Loss11.avg:.1f}/{Loss12.avg:.3f} | Loss2 {Loss21.avg:.1f}/{Loss22.avg:.3f} | "
                f"Loss3 {Loss31.avg:.1f}/{Loss32.avg:.3f}")
            sys.stdout.write(status_text)
            sys.stdout.flush()

        if batch_idx == 0 or (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches:
            _sample_memory_peak()

    if sys.stdout.isatty():
        sys.stdout.write('\n')
        sys.stdout.flush()

    #draw every 10 epoch
    if epoch > 0 and epoch % 10 == 0: 
        visualize('train', epoch, ex_path, x1, y1, ISLA(pred_y1), 32)
        visualize('train', epoch, ex_path, x2, y2, ISLA(pred_y2), 64)
        visualize('train', epoch, ex_path, x3, y3, ISLA(pred_y3), 96)

    rate_error_avg = (
        RateErrorLoss.avg if args.loss_type == RATE_ERROR_LOSS else None
    )
    return Loss.avg, rate_error_avg


def eval(epoch, model, dataloader1, dataloader2, dataloader3, dataloader4):
    model.eval()
    Loss = AverageMeter()
    Loss1 = AverageMeter()
    Loss2 = AverageMeter()
    Loss3 = AverageMeter()
    Loss4 = AverageMeter()

    total_batches = min(len(dataloader1), len(dataloader2), len(dataloader3), len(dataloader4))

    #strat to train
    for batch_idx, (batch1, batch2, batch3, batch4)  in enumerate(zip(dataloader1, dataloader2, dataloader3, dataloader4)):
        x1, y1 = batch1
        x2, y2 = batch2
        x3, y3 = batch3
        x4, y4 = batch4
        x1, y1, x2, y2, x3, y3, x4, y4 = x1.to(device), y1.to(device), x2.to(device), y2.to(device), x3.to(device), y3.to(device), x4.to(device), y4.to(device)

        mask1, mask2, mask3, mask4 = create_mask(x1), create_mask(x2), create_mask(x3), create_mask(x4)

        with torch.no_grad():
            #data1 size32
            pred_y1 = model(x1)
            loss1 = mse(ISLA(pred_y1), y1)*mask1

            #data2 size64
            pred_y2 = model(x2)
            loss2 = mse(ISLA(pred_y2), y2)*mask2
            
            #data3 size96
            pred_y3 = model(x3)
            loss3 = mse(ISLA(pred_y3), y3)*mask3

            #data4 size128
            pred_y4 = model(x4)
            loss4 = mse(ISLA(pred_y4), y4)*mask4

        
        Loss1.update( loss1.mean().item(),  x1.size(0) )
        Loss2.update( loss2.mean().item(),  x2.size(0) )
        Loss3.update( loss3.mean().item(),  x3.size(0) )
        Loss4.update( loss4.mean().item(),  x4.size(0) )
        Loss.update( ((loss1.mean()+loss2.mean()+loss3.mean()+loss4.mean())/4).item(), x1.size(0)+x2.size(0)+x3.size(0)+x4.size(0) )

        if sys.stdout.isatty():
            percentage = ((batch_idx + 1) / total_batches) * 100
            status_text = f"\rEval: epoch {epoch} [{percentage:3.0f}%] | Loss {Loss.avg:.1f} | Loss1 {Loss1.avg:.1f} | Loss2 {Loss2.avg:.1f} | Loss3 {Loss3.avg:.1f} | Loss4 {Loss4.avg:.1f}"
            sys.stdout.write(status_text)
            sys.stdout.flush()

        if batch_idx == 0 or (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches:
            _sample_memory_peak()

    if sys.stdout.isatty():
        sys.stdout.write('\n')
        sys.stdout.flush()
    
    #draw every 10 epoch
    if epoch > 0 and epoch % 10 == 0: 
        visualize('eval', epoch, ex_path, x1, y1, ISLA(pred_y1), 32)
        visualize('eval', epoch, ex_path, x2, y2, ISLA(pred_y2), 64)
        visualize('eval', epoch, ex_path, x3, y3, ISLA(pred_y3), 96)
        visualize('eval', epoch, ex_path, x4, y4, ISLA(pred_y4), 128)

    return Loss1.avg, Loss2.avg, Loss3.avg, Loss4.avg, Loss.avg


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Unet micromagnetics')
    parser.add_argument('--batch-size', type=int, default=100, help='input batch size for 32x32 training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=500, help='evaluation batch size for each resolution (default: 500)')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate (default: 0.005)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 1000)')
    parser.add_argument('--kc', type=int, default=16, help='kernels of first layer (default: 16)')
    parser.add_argument('--inch', type=int, default=6, help='input channels (default: 6)')
    parser.add_argument('--cornum', type=int, default=1000, help='core number (default: 1000)')
    parser.add_argument('--ntest', type=int, default=20, help='held-out cases per 32/64/96 resolution and eval cases for 128 (default: 20)')
    parser.add_argument('--ntrain', type=int, default=300, help='training split stop index per 32/64/96 resolution (default: 300)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU used (default: 0)')
    parser.add_argument('--ex', type=float, default=1.0, help='experiment identifier (default: 1.0)')
    parser.add_argument('--dataug', action=argparse.BooleanOptionalAction, default=True, help='enable physical symmetry augmentation')
    parser.add_argument('--alpha', type=float, default=0.5, help='weighting coefficient for alpha-weighted losses')
    parser.add_argument('--alpha-torque', type=float, default=0.5, help='exchange-torque-rate coefficient in the combined loss (default: 0.5)')
    parser.add_argument('--alpha-grad', type=float, default=0.5, help='gradient-tensor-rate coefficient in the combined loss (default: 0.5)')
    parser.add_argument('--rate-error-lambda', type=float, default=0.5, help='coefficient for the demagnetizing-field rate-error auxiliary loss (default: 0.5)')
    parser.add_argument('--loss_type', type=str, default='baseline', help='loss weighting method')
    parser.add_argument('--torque-lambda', type=float, default=0.1, help='coefficient for torque-mismatch auxiliary loss')
    parser.add_argument('--model', type=str, default=None, help='existing model to continue training')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader workers PER DataLoader (default: 8)')
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
    ex_path = os.path.join(f"./{args.loss_type}", "_".join(folder_bits))
    os.makedirs(ex_path, exist_ok=True)

    # Epoch records go to stderr (Slurm error log) and to epoch_summary.log.
    # Normal print() output stays on stdout (Slurm output log).
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
    print(f"  batch size 32 base: {args.batch_size}", flush=True)
    print(f"  test batch size: {args.test_batch_size}", flush=True)
    print(f"  DataLoader workers per loader: {args.num_workers}", flush=True)
    print(f"  output directory: {ex_path}", flush=True)
    print(f"  device: {device}", flush=True)

    # Save the launch parameters immediately, before the potentially long data-loading step.
    active_coefficient = None
    if args.loss_type == "torque_mismatch":
        active_coefficient = {"name": "torque_lambda", "value": args.torque_lambda}
    elif args.loss_type == COMBINED_RATE_LOSS:
        active_coefficient = {
            "alpha_torque": args.alpha_torque,
            "alpha_grad": args.alpha_grad,
        }
    elif args.loss_type == RATE_ERROR_LOSS:
        active_coefficient = {
            "name": "rate_error_lambda",
            "value": args.rate_error_lambda,
        }
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

    # Training data paths: 32, 64, 96. 128 is evaluation-only.
    data_path1 = [
        '../../../utils/Dataset/rate_change/w32/masked1',
        '../../../utils/Dataset/rate_change/w32/masked2',
        '../../../utils/Dataset/rate_change/w32/unmasked',
    ]
    data_path2 = [
        '../../../utils/Dataset/rate_change/w64/masked1',
        '../../../utils/Dataset/rate_change/w64/masked2',
        '../../../utils/Dataset/rate_change/w64/unmasked',
    ]
    data_path3 = [
        '../../../utils/Dataset/rate_change/w96/masked1',
        '../../../utils/Dataset/rate_change/w96/masked2',
        '../../../utils/Dataset/rate_change/w96/unmasked',
    ]
    data_path4 = [
        '../../../utils/Dataset/rate_change/w128/masked1',
        '../../../utils/Dataset/rate_change/w128/masked2',
        '../../../utils/Dataset/rate_change/w128/unmasked',
    ]

    print("Creating datasets", flush=True)
    use_temporal_data = args.loss_type in RATE_LOSS_TYPES
    train_dataset1, test_dataset1 = dataset_prepare_temporal_physics(
        data_path1, ntest=args.ntest, ntrain=args.ntrain, cn=args.cornum,
        include_previous_train=use_temporal_data,
    )
    train_dataset2, test_dataset2 = dataset_prepare_temporal_physics(
        data_path2, ntest=args.ntest, ntrain=args.ntrain, cn=args.cornum,
        include_previous_train=use_temporal_data,
    )
    train_dataset3, test_dataset3 = dataset_prepare_temporal_physics(
        data_path3, ntest=args.ntest, ntrain=args.ntrain, cn=args.cornum,
        include_previous_train=use_temporal_data,
    )
    test_dataset4 = dataset_prepare(
        data_path4, ntest=0, n128=args.ntest, ntrain=0, cn=args.cornum,
        mode='eval128',
    )

    print_memory("Memory after preparing datasets")
    _write_memory_record(ex_path, "after_dataset_preparation")

    bsz1 = args.batch_size
    bsz2 = round(bsz1 / (len(train_dataset1) / len(train_dataset2)))
    bsz3 = round(bsz1 / (len(train_dataset1) / len(train_dataset3)))

    print(
        f"training samples: 32={len(train_dataset1)}, 64={len(train_dataset2)}, 96={len(train_dataset3)}",
        flush=True,
    )
    print(
        f"testing/evaluation samples: 32={len(test_dataset1)}, 64={len(test_dataset2)}, "
        f"96={len(test_dataset3)}, 128={len(test_dataset4)}",
        flush=True,
    )
    print(f"training batch sizes: 32={bsz1}, 64={bsz2}, 96={bsz3}", flush=True)

    train_dataloader1 = torch.utils.data.DataLoader(
        dataset=train_dataset1, batch_size=bsz1, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    train_dataloader2 = torch.utils.data.DataLoader(
        dataset=train_dataset2, batch_size=bsz2, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    train_dataloader3 = torch.utils.data.DataLoader(
        dataset=train_dataset3, batch_size=bsz3, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    test_dataloader1 = torch.utils.data.DataLoader(
        dataset=test_dataset1, batch_size=args.test_batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    test_dataloader2 = torch.utils.data.DataLoader(
        dataset=test_dataset2, batch_size=args.test_batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    test_dataloader3 = torch.utils.data.DataLoader(
        dataset=test_dataset3, batch_size=args.test_batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    test_dataloader4 = torch.utils.data.DataLoader(
        dataset=test_dataset4, batch_size=args.test_batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    loss_settings = {
        "loss_type": args.loss_type,
        "uses_alpha": _uses_alpha(args.loss_type),
        "alpha": args.alpha if _uses_alpha(args.loss_type) else None,
        "uses_combined_rate_alphas": args.loss_type == COMBINED_RATE_LOSS,
        "alpha_torque": args.alpha_torque if args.loss_type == COMBINED_RATE_LOSS else None,
        "alpha_grad": args.alpha_grad if args.loss_type == COMBINED_RATE_LOSS else None,
        "combined_weight_formula": (
            "1 + alpha_torque * abs(exchange_torque_rate) "
            "+ alpha_grad * abs(gradient_tensor_rate)"
            if args.loss_type == COMBINED_RATE_LOSS
            else None
        ),
        "uses_rate_error_lambda": args.loss_type == RATE_ERROR_LOSS,
        "rate_error_lambda": (
            args.rate_error_lambda if args.loss_type == RATE_ERROR_LOSS else None
        ),
        "rate_error_quantity": (
            "demagnetizing-field difference per saved transition"
            if args.loss_type == RATE_ERROR_LOSS
            else None
        ),
        "uses_torque_lambda": args.loss_type == "torque_mismatch",
        "torque_lambda": args.torque_lambda if args.loss_type == "torque_mismatch" else None,
    }
    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "output_directory": ex_path,
        "loss": loss_settings,
        "optimizer": {
            "name": "Adam",
            "learning_rate": args.lr,
            "betas": [0.9, 0.999],
            "weight_decay": 1e-4,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size_32": bsz1,
            "batch_size_64": bsz2,
            "batch_size_96": bsz3,
            "test_batch_size": args.test_batch_size,
            "num_workers_per_dataloader": args.num_workers,
            "simultaneous_train_dataloaders": 3,
            "simultaneous_eval_dataloaders": 4,
            "data_augmentation": args.dataug,
            "temporal_previous_state_loaded_for_training": use_temporal_data,
            "cornum": args.cornum,
            "ntest_cases_requested": args.ntest,
            "ntrain_split_stop": args.ntrain,
            "random_seed_torch": 0,
        },
        "samples": {
            "train_32": len(train_dataset1),
            "train_64": len(train_dataset2),
            "train_96": len(train_dataset3),
            "test_32": len(test_dataset1),
            "test_64": len(test_dataset2),
            "test_96": len(test_dataset3),
            "eval_128": len(test_dataset4),
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
        "dataset_paths": {
            "32": data_path1,
            "64": data_path2,
            "96": data_path3,
            "128_eval": data_path4,
        },
    }
    with open(os.path.join(ex_path, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)
    print(f"Full reproducibility configuration saved to {os.path.join(ex_path, 'run_config.json')}", flush=True)
    print(f"Per-epoch timing/loss log saved to {os.path.join(ex_path, 'training.log')}", flush=True)
    print(f"Memory snapshots saved to {os.path.join(ex_path, 'memory_usage.jsonl')}", flush=True)

    loss_train_list = []
    loss_test_list1 = []
    loss_test_list2 = []
    loss_test_list3 = []
    loss_test_list4 = []
    epoch_list = []
    best_loss = float('inf')
    best_epoch = -1
    epoch_durations = []
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        loss_train, train_rate_error = train(
            epoch,
            model,
            optim,
            train_dataloader1,
            train_dataloader2,
            train_dataloader3,
        )
        loss_test1, loss_test2, loss_test3, loss_test4, avg = eval(
            epoch, model, test_dataloader1, test_dataloader2, test_dataloader3, test_dataloader4
        )

        loss_train_list.append(loss_train)
        epoch_list.append(epoch)
        loss_test_list1.append(loss_test1)
        loss_test_list2.append(loss_test2)
        loss_test_list3.append(loss_test3)
        loss_test_list4.append(loss_test4)

        # The original checkpoint criterion: average validation loss over 32/64/96.
        loss_test = (loss_test1 + loss_test2 + loss_test3) / 3
        model_path = os.path.join(ex_path, "ckpt")
        os.makedirs(model_path, exist_ok=True)
        new_best = loss_test < best_loss
        if new_best:
            previous_best = best_loss
            best_loss = loss_test
            best_epoch = epoch
            best_model_path = os.path.join(model_path, f"best_model_{best_loss:.1f}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(
                f"NEW BEST MODEL | epoch {epoch} | validation={best_loss:.1f} "
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

        # Exactly one logger call per completed epoch -> one line per epoch in stderr.
        rate_error_log = (
            f" | train_rate_error={train_rate_error:.3f}"
            if train_rate_error is not None
            else ""
        )
        epoch_logger.info(
            f"epoch={epoch} ({epoch + 1}/{args.epochs}) | {_loss_descriptor(args)} | "
            f"train={loss_train:.2f}{rate_error_log} | "
            f"val32={loss_test1:.1f} val64={loss_test2:.1f} "
            f"val96={loss_test3:.1f} val128={loss_test4:.1f} | "
            f"checkpoint_metric={loss_test:.1f} best={best_loss:.1f}@epoch{best_epoch} "
            f"new_best={'yes' if new_best else 'no'} | epoch_time={_format_duration(epoch_time)} | "
            f"elapsed={_format_duration(elapsed)} | ETA={_format_duration(eta_seconds)} | "
            f"projected_finish={projected_finish}"
        )

        _write_memory_record(ex_path, "epoch_complete", epoch=epoch)

        plt.clf()
        plt.plot(epoch_list, loss_train_list, 'r-', alpha=1, label='train_32_64_96')
        plt.plot(epoch_list, loss_test_list1, 'c-', alpha=1, label='test_32')
        plt.plot(epoch_list, loss_test_list2, 'g-', alpha=1, label='test_64')
        plt.plot(epoch_list, loss_test_list3, 'b-', alpha=1, label='test_96')
        plt.plot(epoch_list, loss_test_list4, 'm-', alpha=1, label='test_128')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss-log')
        plt.yscale('log')
        plt.savefig(os.path.join(ex_path, f'loss_ex{args.ex}.png'))

    elapsed = time.time() - start_time
    experiment_info = {
        **run_config,
        "result": {
            "best_epoch": best_epoch,
            "best_validation_loss_32_64_96_average": best_loss,
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
        f"TRAINING COMPLETE | best epoch={best_epoch} | best validation={best_loss:.1f} | "
        f"elapsed={_format_duration(elapsed)}",
        flush=True,
    )

