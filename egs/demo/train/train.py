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

RATE_LOSS_TYPES = {
    "gradient_mag_rate",
    "gradient_tensor_rate",
    "exch_field_rate",
    "exch_torque_rate",
    "exch_e_density_rate",
    "demag_torque_rate",
    "demag_field_rate",
    "winding_density_rate"}


def train(epoch, model, optim, train_dataloader1, train_dataloader2, train_dataloader3):
    model.train()
    Loss = AverageMeter()
    Loss11 = AverageMeter()
    Loss12 = AverageMeter()
    Loss21 = AverageMeter()
    Loss22 = AverageMeter()
    Loss31 = AverageMeter()
    Loss32 = AverageMeter()

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

        if args.loss_type in ("baseline", "torque_mismatch"):
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

        elif args.loss_type == "exch_field_rate":
            wd1 = exchange_field_rate(x1, x1_prev)            
            wd2 = exchange_field_rate(x2, x2_prev)
            wd3 = exchange_field_rate(x3, x3_prev)

        elif args.loss_type == "exch_torque_rate":
            wd1 = exchange_torque_rate(x1, x1_prev)
            wd2 = exchange_torque_rate(x2, x2_prev)
            wd3 = exchange_torque_rate(x3, x3_prev)

        elif args.loss_type == "exch_e_density_rate":
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

        else:
            raise ValueError(f"Unknown loss_type: {args.loss_type}")

        if args.loss_type not in ("baseline", "torque_mismatch"):
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
                    "main_loss_96": loss3.item()}

                with open(os.path.join(ex_path, "torque_mismatch_stats.json"), "w", encoding="utf-8") as f: json.dump(torque_stats, f, indent=4)

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

        percentage = ((batch_idx + 1) / total_batches) * 100

        status_text = (
            f"\rTrain: epoch {epoch} [{percentage:3.0f}%] | Loss {Loss.avg:.1f} | "
            f"Loss1 {Loss11.avg:.1f}/{Loss12.avg:.3f} | Loss2 {Loss21.avg:.1f}/{Loss22.avg:.3f} | "
            f"Loss3 {Loss31.avg:.1f}/{Loss32.avg:.3f}")
        sys.stdout.write(status_text)
        sys.stdout.flush()

    # Finish the terminal line at the end of the epoch
    sys.stdout.write('\n')

    #draw every 10 epoch
    if epoch > 0 and epoch % 10 == 0: 
        visualize('train', epoch, ex_path, x1, y1, ISLA(pred_y1), 32)
        visualize('train', epoch, ex_path, x2, y2, ISLA(pred_y2), 64)
        visualize('train', epoch, ex_path, x3, y3, ISLA(pred_y3), 96)

    return Loss.avg


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

        percentage = ((batch_idx + 1) / total_batches) * 100

        status_text = f"\rEval: epoch {epoch} [{percentage:3.0f}%] | Loss {Loss.avg:.1f} | Loss1 {Loss1.avg:.1f} | Loss2 {Loss2.avg:.1f} | Loss3 {Loss3.avg:.1f} | Loss4 {Loss4.avg:.1f}"
        sys.stdout.write(status_text)
        sys.stdout.flush()

    sys.stdout.write('\n')
    
    #draw every 10 epoch
    if epoch > 0 and epoch % 10 == 0: 
        visualize('eval', epoch, ex_path, x1, y1, ISLA(pred_y1), 32)
        visualize('eval', epoch, ex_path, x2, y2, ISLA(pred_y2), 64)
        visualize('eval', epoch, ex_path, x3, y3, ISLA(pred_y3), 96)
        visualize('eval', epoch, ex_path, x4, y4, ISLA(pred_y4), 128)

    return Loss1.avg, Loss2.avg, Loss3.avg, Loss4.avg, Loss.avg


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='Unet micromagnetics')
    parser.add_argument('--batch-size', type=int,   default=100,    help='input batch size for training (default: 100)')
    parser.add_argument('--lr',         type=float, default=0.005,  help='learning rate (default: 0.005)')
    parser.add_argument('--epochs',     type=int,   default=1000,   help='number of epochs to train (default: 1000)')
    
    parser.add_argument('--kc',        type=int,    default=16,     help='kernels of first layer (default: 16)')
    parser.add_argument('--inch',      type=int,    default=6,      help='input channels (default: 6)')
    parser.add_argument('--cornum',    type=int,    default=1000,   help='core number (default: 1000)')
    parser.add_argument('--ntest',     type=int,    default=20,     help='test number (default: 20)')
    parser.add_argument('--ntrain',    type=int,    default=300,    help='train number (default: 300)')

    parser.add_argument('--gpu',        type=int,   default=0,      help='GPU used (default: 0)')
    parser.add_argument('--ex',         type=float, default=1.0,    help='experiment (default: 0)')
    parser.add_argument('--dataug', action=argparse.BooleanOptionalAction, default=True, help='enable physical symmetry augmentation')    
    parser.add_argument('--alpha',      type=float, default=0.5,    help='weighting coefficient for weighted loss')
    parser.add_argument('--loss_type',  type=str,  default='baseline', help='loss weighting method')
    parser.add_argument('--torque-lambda', type=float, default=0.1, help='coefficient for torque-mismatch auxiliary loss')
    parser.add_argument('--model',      type=str,  default=None,     help='existing model to continue training')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(device, flush=True)
        print(f"GPU reserved : {torch.cuda.memory_reserved()/1024**3:.2f} GB")
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
    
    # Model, optimizer, and data loaders initialization
    model = UNet(kc=args.kc, inc=args.inch, ouc=args.inch).to(device)
    if args.model is not None:
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"Loaded model: {args.model}", flush=True)
    
    optim = optimizer.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0001)

    # #load data
    # data_path11 = '../../../utils/Dataset/data_Hd32_Hext1000_mask'
    # data_path12 = '../../../utils/Dataset/data_Hd32_Hext100_mask'
    # data_path13 = '../../../utils/Dataset/data_Hd32_Hext0'

    data_path11 = '../../../utils/Dataset/rate_change/w32/masked1'
    data_path12 = '../../../utils/Dataset/rate_change/w32/masked2'
    data_path13 = '../../../utils/Dataset/rate_change/w32/unmasked'

    # data_path21 = '../../../utils/Dataset/data_Hd64_Hext1000_mask'
    # data_path22 = '../../../utils/Dataset/data_Hd64_Hext100_mask'
    # data_path23 = '../../../utils/Dataset/data_Hd64_Hext0'

    data_path21 = '../../../utils/Dataset/rate_change/w64/masked1'
    data_path22 = '../../../utils/Dataset/rate_change/w64/masked2'
    data_path23 = '../../../utils/Dataset/rate_change/w64/unmasked'

    # data_path31 = '../../../utils/Dataset/data_Hd96_Hext1000_mask'
    # data_path32 = '../../../utils/Dataset/data_Hd96_Hext100_mask'
    # data_path33 = '../../../utils/Dataset/data_Hd96_Hext0'

    data_path31 = '../../../utils/Dataset/rate_change/w96/masked1'
    data_path32 = '../../../utils/Dataset/rate_change/w96/masked2'
    data_path33 = '../../../utils/Dataset/rate_change/w96/unmasked'

    # data_path41 = '../../../utils/Dataset/data_Hd128_Hext1000_mask'
    # data_path42 = '../../../utils/Dataset/data_Hd128_Hext100_mask'
    # data_path43 = '../../../utils/Dataset/data_Hd128_Hext0'

    data_path41 = '../../../utils/Dataset/rate_change/w128/masked1'
    data_path42 = '../../../utils/Dataset/rate_change/w128/masked2'
    data_path43 = '../../../utils/Dataset/rate_change/w128/unmasked'
    
    data_path1 = [data_path11, data_path12, data_path13]
    data_path2 = [data_path21, data_path22, data_path23]
    data_path3 = [data_path31, data_path32, data_path33]
    data_path4 = [data_path41, data_path42, data_path43]

    print("Creating datasets", flush=True)

    use_temporal_data = (args.loss_type in RATE_LOSS_TYPES)
    train_dataset1, test_dataset1 = dataset_prepare_temporal_physics(data_path1, ntest=args.ntest, ntrain=args.ntrain, cn=args.cornum, include_previous_train=use_temporal_data)
    train_dataset2, test_dataset2 = dataset_prepare_temporal_physics(data_path2, ntest=args.ntest, ntrain=args.ntrain, cn=args.cornum, include_previous_train=use_temporal_data)
    train_dataset3, test_dataset3 = dataset_prepare_temporal_physics(data_path3, ntest=args.ntest, ntrain=args.ntrain, cn=args.cornum, include_previous_train=use_temporal_data)

    # 128 is evaluation-only for every loss type
    test_dataset4 = dataset_prepare(data_path4, ntest=0, n128=args.ntest, ntrain=0, cn=args.cornum, mode='eval128')

    print_memory(msg="Memory used after preparing datasets")

    bsz1=args.batch_size
    print('samples 1 2 3:',len(train_dataset1), len(train_dataset2), len(train_dataset3))
    bsz2=round(bsz1 / (len(train_dataset1) / len(train_dataset2)))
    bsz3=round(bsz1 / (len(train_dataset1) / len(train_dataset3)))
    print('batch size 1 2 3: ',bsz1, bsz2, bsz3,'\n')


    train_dataloader1 = torch.utils.data.DataLoader(dataset=train_dataset1, batch_size=bsz1, shuffle=True,  num_workers=8, drop_last=False)
    train_dataloader2 = torch.utils.data.DataLoader(dataset=train_dataset2, batch_size=bsz2, shuffle=True,  num_workers=8, drop_last=False)
    train_dataloader3 = torch.utils.data.DataLoader(dataset=train_dataset3, batch_size=bsz3, shuffle=True,  num_workers=8, drop_last=False)
    
    test_dataloader1  = torch.utils.data.DataLoader(dataset=test_dataset1,  batch_size=500, shuffle=True,  num_workers=8, drop_last=False) 
    test_dataloader2  = torch.utils.data.DataLoader(dataset=test_dataset2,  batch_size=500, shuffle=True,  num_workers=8, drop_last=False)
    test_dataloader3  = torch.utils.data.DataLoader(dataset=test_dataset3,  batch_size=500, shuffle=True,  num_workers=8, drop_last=False)
    test_dataloader4  = torch.utils.data.DataLoader(dataset=test_dataset4,  batch_size=500, shuffle=True,  num_workers=8, drop_last=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #experiment path
    ex_path = os.path.join(f"./{args.loss_type}_contin",
                           f"{timestamp}_ex{args.ex}_bsz{bsz1}_lr{args.lr}_Unet_kc{args.kc}_inch{args.inch}",)
    os.makedirs(ex_path, exist_ok=True)

    # Set up logging
    logging.basicConfig(filename=ex_path + '/training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f"Total parameters: {num_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    logging.info(f"GPU: {torch.cuda.get_device_name(device)}")
    logging.info(f"PyTorch: {torch.__version__}")

    loss_train_list = []
    loss_test_list1 = []
    loss_test_list2 = []
    loss_test_list3 = []
    loss_test_list4 = []
    epoch_list = []
    best_loss = float('inf')
    best_epoch = -1

    start_time = time.time()

    for epoch in range(args.epochs): 
        #train
        loss_train = train(epoch, model, optim,  train_dataloader1, train_dataloader2, train_dataloader3)

        loss_train_list.append(loss_train)
        logging.info('epoch: {} loss: {:.2f}'.format(epoch, loss_train))
    
        #evaluate
        loss_test1, loss_test2, loss_test3, loss_test4, avg = eval(epoch, model, test_dataloader1, test_dataloader2, test_dataloader3, test_dataloader4)
        logging.info('Evaluate loss32: {:.1f} loss64: {:.1f} loss96: {:.1f} / loss128: {:.1f} avg: {:.1f}'
                    .format(loss_test1, loss_test2, loss_test3, loss_test4, avg))
        
        epoch_list.append(epoch)
        loss_test_list1.append(loss_test1)
        loss_test_list2.append(loss_test2)
        loss_test_list3.append(loss_test3)
        loss_test_list4.append(loss_test4)

        if epoch % 100 ==0:
            print_memory(msg=f"Memory used after training epoch number: {epoch}")

        #model save path
        model_path = os.path.join(ex_path, "ckpt")
        os.makedirs(model_path, exist_ok=True)

        #save best model checkpoint
        loss_test = (loss_test1+loss_test2+loss_test3)/3
        if loss_test < best_loss:
            print('loss_test: {:.1f} < best_loss: {:.1f} \n'.format(loss_test, best_loss))
            best_loss = loss_test
            best_epoch = epoch
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, f"{model_path}/best_model_{best_loss:.1f}.pt")

        # draw loss_train and loss_test
        plt.clf()
        plt.plot(epoch_list, loss_train_list, 'r-',  alpha=1, label='train_32_64_96')
        plt.plot(epoch_list, loss_test_list1, 'c-',  alpha=1, label='test_32')
        plt.plot(epoch_list, loss_test_list2, 'g-',  alpha=1, label='test_64')
        plt.plot(epoch_list, loss_test_list3, 'b-',  alpha=1, label='test_96')
        plt.plot(epoch_list, loss_test_list4, 'm-',  alpha=1, label='test_128')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss-log')
        plt.yscale('log')  # set y-axis scale to logarithmic
        plt.savefig(os.path.join(ex_path, 'loss_ex{}.png'.format(args.ex)))

    elapsed = time.time() - start_time

    experiment_info = {"alpha": args.alpha,
                       "learning_rate": args.lr,
                       "batch_size_32": bsz1,
                       "batch_size_64": bsz2,
                       "batch_size_96": bsz3,
                       "epochs": args.epochs,
                       "kernel_channels": args.kc,
                       "input_channels": args.inch,
                       "optimizer": "Adam",
                       "betas": [0.9, 0.999],
                       "weight_decay": 1e-4,
                       "data_augmentation": args.dataug,
                       "seed": 0,
                       "best_epoch": best_epoch,
                       "best_validation_loss": best_loss,
                       "training_time_seconds": elapsed}

    with open(os.path.join(ex_path, "experiment.json"), "w") as f:
        json.dump(experiment_info, f, indent=4)  

    print_memory("Memory usage after training: ")

    logging.info(f"Best epoch: {best_epoch}")
    logging.info(f"Best validation loss: {best_loss}")
    logging.info(f"Training time: {elapsed:.2f} seconds")
    logging.info(f"Training time: {elapsed/60:.2f} minutes")
    logging.info(json.dumps(experiment_info, indent=4))  

