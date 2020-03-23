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
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='D:\git\crawler\zigbang\\train')
parser.add_argument('--mask' , type=str, default='D:\dataset\irregular_mask\irregular_mask\disocclusion_img_mask')
parser.add_argument('--val_data', type=str, default='D:\git\crawler\zigbang\\val')
parser.add_argument('--batch_size',type=int ,default=2)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_iter',type=int, default=1000000)
parser.add_argument('--input_size', type=int,nargs='+', default=[512,512])
parser.add_argument('--checkpoint',type=str, default='checkpoints/pconv')
parser.add_argument('--iter_log', type=int, default=1000)
parser.add_argument('--iter_save',type=int, default=1000)
parser.add_argument('--iter_sample',type=int, default=1000)
parser.add_argument('--iter_eval', type=int, default=1000)
parser.add_argument('--mini-eval', type=int, default=5000)
parser.add_argument('--iter-lr', type=int, default= 200000)
parser.add_argument('--dataset', type=str, default='folder')
parser.add_argument('--tensor-log',type=str,default='logs/pconv')

args,_ = parser.parse_known_args()

# make dir data checkpoint
# os.makedirs(args.data,exist_ok=True)
date = datetime.today().strftime("%Y/%m/%d")
os.makedirs(args.checkpoint,exist_ok=True)
sample_dir = os.path.join(args.checkpoint,'sample')
result_dir = os.path.join(args.checkpoint,date, 'results')
os.makedirs(result_dir,exist_ok=True)
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
    torchvision.transforms.Normalize(mean,std)
])
val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(args.input_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])

# writer
writer = SummaryWriter(args.tensor_log)

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

def reset_losses(*args):
    for loss in args:
        loss.reset()


def validate(model,criterion, val_loader,maskloader,cur_iter):
    #eval
    global state_dict
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
                writer.add_scalar('val_total_loss', total_loss_avg.avg, cur_iter)
            if args.iter_sample and i % args.iter_sample == 0:
                images = images * torch.as_tensor(std, dtype=torch.float32, device=device)[:,None,None] + torch.as_tensor(mean, dtype=torch.float32, device=device)[:,None,None]
                masked_img = (images) * masks + (1 - masks)
                torchvision.utils.save_image(out_img, os.path.join(sample_dir, f'{cur_iter}_out.jpg'))
                torchvision.utils.save_image(masked_img, os.path.join(sample_dir, f'{cur_iter}_image.jpg'))
                writer.add_image(f'{cur_iter}_sample', torchvision.utils.make_grid(torch.stack([out_img,masked_img])))
    return total_loss_avg.avg

def train(model, criterion, dataloader, maskloader,val_loader):
    global state_dict
    cur_iter = state_dict['iter']
    best_loss = state_dict['loss']
    lr = state_dict.get('lr', args.lr)

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
    optimizer = torch.optim.Adam(model.parameters(),lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
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

            assert (torch.unique(masks) == torch.tensor([0, 1])).all(), f'masks should only 0,1 {torch.unique(masks)}'

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
                writer.add_scalar('total_loss', total_loss_avg.avg, cur_iter)
                writer.add_scalar('loss_hole', loss_hole_avg.avg, cur_iter)
                writer.add_scalar('loss_perceptual', loss_perceptual_avg.avg, cur_iter)
            if args.iter_eval and cur_iter % args.iter_eval == 0:
                loss = validate(model, criterion,val_loader,maskloader,cur_iter)
                # after eval reset train loss
                reset_losses(*progress.meters)
            if args.iter_save and cur_iter % args.iter_save == 0:
                # iter_eval == iter_save before save evalutate update loss
                model.save(args.checkpoint,cur_iter,loss, optimizer.param_groups[0]['lr'])
                if best_loss > loss:
                    best_loss = loss
                    last_checkpoint = model.last_checkpoint(args.checkpoint,'.pth')[0]
                    shutil.copy(last_checkpoint, os.path.join(args.checkpoint, f'best_model.pth'))
            if args.iter_lr and cur_iter % args.iter_lr == 0:
                # decay lr
                scheduler.step()
model = Unet(3, 64)
state_dict = model.load(args.checkpoint)

def inference(image,mask):
    global model,result_dir
    model.eval()
    # 0 for hole 1 for valid
    mask = torch.where(mask>0, torch.tensor(0),torch.tensor(1))
    mask = torch.cat([mask,mask,mask]).float()
    out_img, _ = model(image.unsqueeze(0),mask.unsqueeze(0))
    result = out_img.squeeze(0) * (1-mask) + mask * image

    # save result
    time = datetime.today().strftime('%H_%M')
    savepoint = os.path.join(result_dir,f'{time}_{image.filename}')
    result_img = torchvision.transforms.ToPILImage()(result)
    result_img.save(savepoint)
    return result_img




if __name__ == '__main__':
    # model = Unet(3, 64).to(device)
    if args.dataset == 'folder':
        dataset = torchvision.datasets.ImageFolder(args.data, transform=transform)
        val_dataset = torchvision.datasets.ImageFolder(args.val_data, transform=val_transform)
        # val_dataset, _ = torch.utils.data.random_split(val_dataset, [args.mini_eval, len(val_dataset) - args.mini_eval])
    else:
        dataset = torchvision.datasets.LSUN(args.data,classes=['bedroom_train'],transform=transform)
        val_dataset = torchvision.datasets.LSUN(args.val_data, classes=['bedroom_val'],transform= transform)

    if args.mini_eval:
        if args.dataset == 'folder':
            val_dataset, _ = torch.utils.data.random_split(val_dataset, [args.mini_eval, len(val_dataset) - args.mini_eval])
        else:
            dataset,val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - args.mini_eval, args.mini_eval])

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)
    mask_generator = MaskGenerator(args.input_size, file_path=args.mask)
    maskloader = torch.utils.data.DataLoader(mask_generator, shuffle=False, batch_size=args.batch_size)
    maskiter = iter(maskloader)
    train(model.to(device),criterion,dataloader,maskloader,val_dataloader)








