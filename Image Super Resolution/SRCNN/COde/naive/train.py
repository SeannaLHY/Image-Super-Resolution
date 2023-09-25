import argparse
import os
import copy
import datetime
import pandas as pd
import numpy as np


import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models_batch import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr, calc_ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    time = datetime.datetime.now()
    args.outputs_dir = os.path.join(args.outputs_dir, str(time))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    time = datetime.datetime.now()
    log_file = os.path.join(args.outputs_dir, "log.txt")
    track = ""
    result = []

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    best_ssim = 0.0

    for epoch in range(args.num_epochs):
        

        start = datetime.datetime.utcnow()
        model.train()
        train_losses = AverageMeter()
        val_losses = AverageMeter()

        
        for data in train_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            train_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    
            
        end = datetime.datetime.utcnow()

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()


        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            loss = criterion(preds, labels)

            val_losses.update(loss.item(), len(inputs))
            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
            epoch_ssim.update(calc_ssim(preds, labels), len(inputs))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())
        if epoch_ssim.avg > best_ssim:
            best_epoch_ssim = epoch
            best_ssim = epoch_ssim.avg
            best_weights_ssim = copy.deepcopy(model.state_dict())


        runtime = (end - start).total_seconds()
        appendix = [epoch, float(epoch_psnr.avg), float(epoch_ssim.avg), float(runtime),train_losses.avg,val_losses.avg]
        result.append(appendix)
        print('eopch number: {}; eval psnr: {:.2f}; eval ssim: {:.4f}; runtime: {:.2f} seconds; train loss: {:.2f}; val_loss: {:.2f}'.format(epoch, epoch_psnr.avg, epoch_ssim.avg,runtime,train_losses.avg,val_losses.avg))


        track += 'eopch number: {}; eval psnr: {:.2f}; eval ssim: {:.4f}; runtime: {:.2f} seconds; train loss: {:.2f}; val_loss: {:.2f}'.format(epoch, epoch_psnr.avg, epoch_ssim.avg, runtime,train_losses.avg,val_losses.avg)
        track += "\n"
    
    track += 'best epoch (psnr): {}, psnr: {:.2f}'.format(best_epoch, best_psnr)
    track += 'best epoch (ssim): {}, ssim: {:.4f}'.format(best_epoch_ssim, best_ssim)

    with open(log_file, 'w') as f:
        f.write(track)
    f.close()
    df = pd.DataFrame(result, columns=['Epoch_num', 'AVG_PSNR(SRCNN)', 'SSIM(SRCNN)','Runtime','Train Loss','Validation Loss'])
    excel_file = os.path.join(args.outputs_dir, "train_process.xlsx")
    df.to_excel(excel_file)


    print('best epoch (psnr): {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    print('best epoch (ssim): {}, ssim: {:.4f}'.format(best_epoch_ssim, best_ssim))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best_psnr.pth'))
    torch.save(best_weights_ssim, os.path.join(args.outputs_dir, 'best_ssim.pth'))

