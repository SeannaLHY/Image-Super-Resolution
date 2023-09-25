import argparse
import os
import copy
import math
import datetime
import pandas as pd

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import RDN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr, calc_ssim, convert_rgb_to_y, denormalize


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=800)
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

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = RDN(scale_factor=args.scale,
                num_channels=3,
                num_features=args.num_features,
                growth_rate=args.growth_rate,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers).to(device)

    if args.weights_file is not None:
        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = TrainDataset(args.train_file, patch_size=args.patch_size, scale=args.scale)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size= args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    best_ssim = 0.0

    log_file = os.path.join(args.outputs_dir, "log.txt")
    track = ""
    result = []

    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1

        model.train()
        epoch_losses = AverageMeter()
        train_loss = 0
        begin = datetime.datetime.utcnow()


        for data in train_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            epoch_losses.update(loss.item(), len(inputs))
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end = datetime.datetime.utcnow()
        runtime = (end - begin).total_seconds()
        train_loss /= len(train_dataloader)




        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()

        val_loss = 0

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs)


            preds = convert_rgb_to_y(denormalize(preds.squeeze(0)), dim_order='chw')
            labels = convert_rgb_to_y(denormalize(labels.squeeze(0)), dim_order='chw')

            preds = preds[args.scale:-args.scale, args.scale:-args.scale]
            labels = labels[args.scale:-args.scale, args.scale:-args.scale]

            loss = criterion(preds, labels)


            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
            epoch_ssim.update(calc_ssim(preds, labels), len(inputs))

            val_loss += loss.item()

        val_loss/=len(eval_dataloader)

        print('eopch number: {}; eval psnr: {:.2f}; eval ssim: {:.4f}; runtime: {:.2f} seconds; train loss: {:.4f}; val_loss: {:.4f}'.format(
                epoch, epoch_psnr.avg, epoch_ssim.avg, runtime, train_loss, val_loss))

        appendix = [epoch, float(epoch_psnr.avg), float(epoch_ssim.avg), float(runtime), train_loss, val_loss]
        result.append(appendix)



        track += 'eopch number: {}; eval psnr: {:.2f}; eval ssim: {:.4f}; runtime: {:.2f} seconds; train loss: {:.4f}; val_loss: {:.4f}'.format(
            epoch, epoch_psnr.avg, epoch_ssim.avg, runtime, train_loss, val_loss)
        track += "\n"

        if epoch_psnr.avg > best_psnr:
            print("Best epoch is {}",epoch)
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())
            path_1 = 'best_psnr_{}.pth'.format(epoch)


        if epoch_ssim.avg > best_ssim:
            best_epoch_ssim = epoch
            best_ssim = epoch_ssim.avg
            best_weights_ssim = copy.deepcopy(model.state_dict())
            path_2 = 'best_ssim_{}.pth'.format(epoch)

    track += 'best epoch with metric psnr: {}, psnr: {:.2f}'.format(best_epoch, best_psnr)
    track += '\n'
    track += 'best epoch with metric ssim: {}, ssim: {:.2f}'.format(best_epoch_ssim, best_ssim)

    with open(log_file, 'w') as f:
        f.write(track)
    f.close()

    df = pd.DataFrame(result, columns=['Epoch_num', 'AVG_PSNR(RDN)', 'SSIM(RDN)', 'Runtime', 'Train Loss',
                                       'Validation Loss'])
    excel_file = os.path.join(args.outputs_dir, "train_process.xlsx")
    df.to_excel(excel_file)

    print('best epoch with metric psnr: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    print('best epoch with metric ssim: {}, ssim: {:.2f}'.format(best_epoch_ssim, best_ssim))

    torch.save(best_weights, os.path.join(args.outputs_dir, 'best_psnr.pth'))
    torch.save(best_weights_ssim, os.path.join(args.outputs_dir, 'best_ssim.pth'))
    print("WORK DONE")




