import argparse
import cv2
import glob
import os
import time
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
import wandb

from dataset import UnetDataset
from loss import GANLoss
from unet.discriminator import Discriminator
from unet.unet_model import UNet
from utils.utils import *


def update_learning_rate(sG, sD, oG):
    old_lr = oG.param_groups[0]['lr']
    sG.step()
    sD.step()
    lr = oG.param_groups[0]['lr']
    print(f'learning rate {old_lr} -> {lr}')


def train(args, device, log_file_path, save_dir):
    # wandb
    wandb_run = wandb.init(project='plant generation', name=f'{args.species}_{args.date}')

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(args.train_res, args.train_res), interpolation=transforms.InterpolationMode.NEAREST, antialias=False),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    source_img_path = sorted(glob.glob(f'/data/tentative_unet/real_{args.species}/train/source/*.png'))
    input_img_path = sorted(glob.glob(f'/data/tentative_unet/real_{args.species}/train/input/*.png'))

    dataset = UnetDataset(
        source_img_path=source_img_path,
        input_img_path=input_img_path,
        resolution=args.train_res,
        transform=train_transform
    )

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    print('setup for making dataset finished')

    # define Generatror, Discriminator
    netG = UNet(n_channels=3, n_classes=3).to(device)
    netD = Discriminator(image_channels=3, hidden_channels=16).to(device)

    initial_epoch = 0

    if args.resume > 0:
        state_dict = torch.load(f'{save_dir}/ckpt/model_{str(args.resume).zfill(3)}.pt')
        netG.load_state_dict(state_dict['state_dict_G'])
        netD.load_state_dict(state_dict['state_dict_D'])
        initial_epoch = args.resume

    # define optimizer
    optimizer_G = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=args.lr/2, betas=(0.5, 0.999))

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - args.train_epoch) / float(args.epochs_decay + 1)
        return lr_l

    scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

    # define criterion
    criterion_adv = GANLoss().to(device)
    criterion_l1 = torch.nn.L1Loss()

    for epoch in range(initial_epoch+1, args.train_epoch + args.epochs_decay + 1):
        epoch_start_time = time.time()
        netG.train()
        netD.train()
        update_learning_rate(sG=scheduler_G, sD=scheduler_D, oG=optimizer_G)
        for batch_id, (gt_img, input_img) in enumerate(train_loader):
            gt_img, input_img = gt_img.to(device), input_img.to(device)
            fake_img = netG(input_img)

            # # update D (GAN loss)
            optimizer_D.zero_grad()

            pred_fake = netD(fake_img.detach())
            loss_D_fake = criterion_adv(pred_fake, False)
            pred_real = netD(gt_img.detach())
            loss_D_real = criterion_adv(pred_real, True)

            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # update G (GAN loss + L1 loss)
            optimizer_G.zero_grad()

            pred_fake = netD(fake_img)
            loss_G_adv = criterion_adv(pred_fake, True)
            loss_G_l1 = criterion_l1(fake_img, gt_img)
            loss_G = args.weight_adv * loss_G_adv + args.weight_l1 * loss_G_l1
            loss_G.backward()
            optimizer_G.step()

            losses = {
                'D_fake': loss_D_fake.item(),
                'D_real': loss_D_real.item(),
                'G_GAN': loss_G_adv.item(),
                'G_L1': loss_G_l1.item()
            }

            # wandb
            wandb_run.log(losses)

            if (batch_id + 1) % args.print_freq == 0:
                message = f'Epoch: {epoch}, iters: {batch_id + 1}, loss_G: {loss_G.item()}, loss_D: {loss_D.item()}'
                print(message)
                with open(log_file_path, "a") as log_file:
                    log_file.write(f'{message}\n')

        if epoch % args.save_freq == 0:
            wandb_img = {}
            fake_numpy, gt_numpy, input_numpy = tensor2numpy(fake_img.squeeze(0)), tensor2numpy(gt_img.squeeze(0)), tensor2numpy(input_img.squeeze(0))
            cv2.imwrite(os.path.join(save_dir, 'img', f'epoch{str(epoch).zfill(3)}_fake.png'), cv2.cvtColor(fake_numpy, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(save_dir, 'img', f'epoch{str(epoch).zfill(3)}_real.png'), cv2.cvtColor(gt_numpy, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(save_dir, 'img', f'epoch{str(epoch).zfill(3)}_input.png'), cv2.cvtColor(input_numpy, cv2.COLOR_BGR2RGB))

            state_dict_G = netG.state_dict()
            state_dict_D = netD.state_dict()
            state_dict = {
                'state_dict_G': state_dict_G,
                'state_dict_D': state_dict_D
            }
            torch.save(state_dict, os.path.join(f'{save_dir}', 'ckpt', f'model_{str(epoch).zfill(2)}.pt'))

            wandb_img['fake'], wandb_img['real'], wandb_img['input'] = wandb.Image(fake_numpy), wandb.Image(
                gt_numpy), wandb.Image(input_numpy)
            wandb_run.log(wandb_img)

        print(
            f'Epoch {epoch} / {args.train_epoch + args.epochs_decay} finished. taken time: {time.time() - epoch_start_time}')


def main(args):
    save_dir = os.path.join('experiments', f'{args.species}_{args.date}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(os.path.join(save_dir, 'img'))
        os.mkdir(os.path.join(save_dir, 'ckpt'))

    device = torch.device(f'cuda:{str(args.gpu_id)}')

    # log file setting
    log_file_path = os.path.join(f'{save_dir}', 'log.txt')
    with open(log_file_path, 'a') as log_file:
        log_file.write('training start\n')
        log_file.write(f'weight of loss: GAN -> {args.weight_adv}, L1 -> {args.weight_l1}\n')

    train(args=args, device=device, log_file_path=log_file_path, save_dir=save_dir)


def get_args():
    parser = argparse.ArgumentParser(description='Train U-Net with GAN loss and L1 loss')
    parser.add_argument('--train_epoch', type=int, default=500, help='Number of epochs')
    parser.add_argument('--epochs_decay', type=int, default=0, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--date', type=str, required=True, help='yymmdd_vv when you conduct this program (vv is time of train)')
    parser.add_argument('--save_freq', type=int, default=5, help='interval saving model')
    parser.add_argument('--print_freq', type=int, default=50, help='interval printing log')
    parser.add_argument('--gpu_id', type=int, required=True, help='specify gpu id you want to use')
    parser.add_argument('--resume', type=int, default=0, help='load model_resume.pt')
    parser.add_argument('--species', type=str, required=True, help='plant species name')
    parser.add_argument('--train_res', type=int, default=1024, help='images are resized to this size')
    parser.add_argument('--weight_l1', type=float, default=1, help='weight of L1 loss')
    parser.add_argument('--weight_adv', type=float, default=1, help='weight of gan loss')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
