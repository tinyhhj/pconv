import argparse
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from unet import Unet
from loss import Loss, VGG16
from mask_generator import MaskGenerator
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='~/PycharmProjects/dataset/celeba')
parser.add_argument('--mask' , type=str, default='irregular_mask/disocclusion_img_mask')
parser.add_argument('--batch_size',type=int ,default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_iter',type=int, default=100000)
parser.add_argument('--input_size', type=int,nargs='+', default=[512,512])
parser.add_argument('--checkpoint',type=str, default='checkpoint')
parser.add_argument('--iter_log', type=int, default=1000)
parser.add_argument('--iter_save',type=int, default=1000)
parser.add_argument('--iter_sample',type=int, default=1000)
parser.add_argument('--iter_eval', type=int, default=1000)

args = parser.parse_args()

# make dir data checkpoint
os.makedirs(args.data,exist_ok=True)
os.makedirs(args.checkpoint,exist_ok=True)
sample_dir = os.path.join(args.checkpoint,'sample')
# if we do sample make dir
if args.iter_sample > 0:
    os.makedirs(sample_dir,exist_ok=True)


feature_extractor = VGG16()
criterion = Loss(feature_extractor)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(args.input_size),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
dataset = torchvision.datasets.ImageFolder(args.data, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
mask_generator = MaskGenerator(args.input_size,file_path=args.mask)
maskloader = torch.utils.data.DataLoader(mask_generator,shuffle=True,batch_size=args.batch_size)
maskiter = iter(maskloader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Unet(3,64).to(device)
cur_iter = model.load(args.checkpoint)
epoch = 0
need_train = True

optimizer = torch.optim.Adam(model.parameters(),args.lr)
while need_train:
    epoch += 1
    print(f'{epoch} running...')
    for images,_ in dataloader:
        if cur_iter >= args.max_iter:
            need_train = False
            print(f'[!] {cur_iter}/{args.max_iter} train finished')
            break

        try:
            masks = next(maskiter)
        except StopIteration:
            # maxlen over
            maskiter = iter(maskloader)
            masks = next(maskiter)

        optimizer.zero_grad()
        images,masks = images.to(device), masks.to(device,dtype=torch.float)

        out_img, _ = model(images,masks)
        # gt, in_mask, out_img
        total_loss = criterion(images, masks,out_img)
        total_loss.backward()
        optimizer.step()

        cur_iter += 1

        if args.iter_log and cur_iter % args.iter_log == 0:
            loss_valid = criterion.loss_valid
            loss_hole = criterion.loss_hole
            loss_perceptual = criterion.loss_perceptual
            loss_style = criterion.loss_style
            loss_total_variation = criterion.loss_total_variation
            print(f'[{cur_iter}/{args.max_iter}] total_loss: {total_loss.item():.4f} '
                  f'loss_valid: {loss_valid:.4f} loss_hole: {loss_hole:.4f} loss_perceptual: {loss_perceptual:.4f} '
                  f'loss_style: {loss_style:.4f} loss_total_variation: {loss_total_variation:.4f}')
        if args.iter_sample and cur_iter % args.iter_sample == 0:
            torchvision.utils.save_image(out_img.detach(), os.path.join(sample_dir, f'{cur_iter}_sample.jpg'))
        if args.iter_save and cur_iter % args.iter_save == 0:
            model.save(args.checkpoint,total_loss.item())








