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
import shutil
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='D:\git\crawler\zigbang\\train')
parser.add_argument('--mask' , type=str, default='D:\dataset\irregular_mask\irregular_mask\disocclusion_img_mask')
parser.add_argument('--val_data', type=str, default='D:\git\crawler\zigbang\\val')
parser.add_argument('--batch_size',type=int ,default=2)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_iter',type=int, default=100000)
parser.add_argument('--input_size', type=int,nargs='+', default=[512,512])
parser.add_argument('--checkpoint',type=str, default='checkpoint')
parser.add_argument('--iter_log', type=int, default=100)
parser.add_argument('--iter_save',type=int, default=1000)
parser.add_argument('--iter_sample',type=int, default=100)
parser.add_argument('--iter_eval', type=int, default=1000)
parser.add_argument('--mini-eval', type=int, default=5000)

args = parser.parse_args()

# make dir data checkpoint
os.makedirs(args.data,exist_ok=True)
os.makedirs(args.checkpoint,exist_ok=True)
sample_dir = os.path.join(args.checkpoint,'sample')
# if we do sample make dir
if args.iter_sample > 0:
    os.makedirs(sample_dir,exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


feature_extractor = VGG16(device)
criterion = Loss(feature_extractor,device)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(args.input_size),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(mean,std)
])
val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(args.input_size),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(mean, std)
])

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

dataset = torchvision.datasets.ImageFolder(args.data, transform=transform)
val_dataset = torchvision.datasets.ImageFolder(args.val_data, transform=val_transform)
val_dataset = torch.utils.data.random_split(val_dataset, [args.mini_eval, len(val_dataset) - args.mini_eval])
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)
mask_generator = MaskGenerator(args.input_size,file_path=args.mask)
maskloader = torch.utils.data.DataLoader(mask_generator,shuffle=False,batch_size=args.batch_size)
maskiter = iter(maskloader)

def record_losses(criterion, loss_valid_avg,loss_hole_avg,loss_perceptual_avg,loss_style_avg,loss_total_variation_avg):
    loss_valid = criterion.loss_valid.item() * criterion.loss_valid_weight
    loss_hole = criterion.loss_hole.item() * criterion.loss_hole_weight
    loss_perceptual = criterion.loss_perceptual.item() * criterion.loss_perceptual_weight
    loss_style = criterion.loss_style.item() * criterion.loss_style_weight
    loss_total_variation = criterion.loss_total_variation.item() * criterion.loss_total_variation_weight
    loss_valid_avg.update(loss_valid)
    loss_hole_avg.update(loss_hole)
    loss_perceptual_avg.update(loss_perceptual)
    loss_style_avg.update(loss_style)
    loss_total_variation_avg.update(loss_total_variation)


def validate(model,criterion, val_loader,maskloader):
    #eval
    model.eval()
    total_loss_avg = AverageMeter('total_loss',':.6f')
    loss_valid_avg = AverageMeter('loss_valid',':.4f')
    loss_hole_avg = AverageMeter('loss_hole',':.4f')
    loss_perceptual_avg = AverageMeter('loss_perceptual', ':.6f')
    loss_style_avg = AverageMeter('loss_style',':.6f')
    loss_total_variation_avg = AverageMeter('loss_total_variation',':.6f')
    progress = ProgressMeter(len(val_loader),
                             [total_loss_avg,
                              loss_valid_avg,
                              loss_hole_avg,
                              loss_perceptual_avg,
                              loss_style_avg,
                              loss_total_variation_avg],
                             prefix='Validate: ')
    maskiter = iter(maskloader)
    with torch.no_grad():
        for i,(images , _) in enumerate(val_dataloader,1):
            try:
                masks = next(maskiter)
            except StopIteration:
                # maxlen over
                maskiter = iter(maskloader)
                masks = next(maskiter)
            images, masks = images.to(device), masks.to(device, dtype=torch.float)
            out_img,_ = model(images,masks)
            total_loss = criterion(images, masks, out_img)

            total_loss_avg.update(total_loss.item())
            record_losses(criterion,loss_valid_avg,loss_hole_avg,loss_perceptual_avg,loss_style_avg,loss_total_variation_avg)

            if args.iter_log and i % args.iter_log == 0:
                progress.display(i)
    return total_loss_avg.avg


model = Unet(3,64).to(device)
cur_iter, best_loss = model.load(args.checkpoint)

def train(model, criterion, dataloader, maskloader,val_loader):
    global cur_iter
    epoch = 0
    need_train = True
    total_loss_avg = AverageMeter('total_loss', ':.6f')
    loss_valid_avg = AverageMeter('loss_valid', ':.4f')
    loss_hole_avg = AverageMeter('loss_hole', ':.4f')
    loss_perceptual_avg = AverageMeter('loss_perceptual', ':.6f')
    loss_style_avg = AverageMeter('loss_style', ':.6f')
    loss_total_variation_avg = AverageMeter('loss_total_variation', ':.6f')
    progress = ProgressMeter(len(dataloader),
                             [total_loss_avg,
                              loss_valid_avg,
                              loss_hole_avg,
                              loss_perceptual_avg,
                              loss_style_avg,
                              loss_total_variation_avg],
                             prefix='Train: ')
    optimizer = torch.optim.Adam(model.parameters(),args.lr)
    maskiter = iter(maskloader)
    while need_train:
        epoch += 1
        print(f'{epoch} running...')
        for i,(images,_) in enumerate(dataloader,1):
            #train
            model.train()
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
            total_loss_avg.update(total_loss.item())
            record_losses(criterion,loss_valid_avg,loss_hole_avg,loss_perceptual_avg,loss_style_avg,loss_total_variation_avg)
            total_loss.backward()
            optimizer.step()

            cur_iter += 1
            loss = 0
            if args.iter_log and cur_iter % args.iter_log == 0:
                progress.display(i)
            if args.iter_sample and cur_iter % args.iter_sample == 0:
                masked_img = (images.detach()) * masks + (1-masks)
                torchvision.utils.save_image(out_img.detach(), os.path.join(sample_dir, f'{cur_iter}_out.jpg'),normalize=True)
                torchvision.utils.save_image(masked_img, os.path.join(sample_dir, f'{cur_iter}_image.jpg'),normalize=True)
            if args.iter_eval and cur_iter % args.iter_eval == 0:
                loss = validate(model, criterion,val_loader,maskloader)
            if args.iter_save and cur_iter % args.iter_save == 0:
                # iter_eval == iter_save before save evalutate update loss
                model.save(args.checkpoint,cur_iter,loss)
                if best_loss > loss:
                    last_checkpoint = model.last_checkpoint(args.checkpoint,'.pth')[0]
                    shutil.copy(last_checkpoint, os.path.join(args.checkpoint, 'best_model.pth'))

train(model,criterion,dataloader,maskloader,val_dataloader)








