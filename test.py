from unet import Unet
import torch
import torchvision
from PIL import Image
from mask_generator import MaskGenerator
import numpy as np
import copy
import math
from loss import VGG16, Loss

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((512,512)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

class mDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.image = transform(Image.open('sample_image.jpg'))
        self.mask_gen = MaskGenerator(512,547)
    def __getitem__(self, item):
        mask = torch.from_numpy(self.mask_gen[0]).float()
        masked_img = copy.deepcopy(self.image)
        # masked_img[mask == 0] = 1
        return [masked_img, mask], self.image
    def __len__(self):
        return 2*200

ds = mDataset()
dl = torch.utils.data.DataLoader(ds,batch_size=2)
device = torch.device('cuda')
feature_extractor = VGG16(device)
criterion = Loss(feature_extractor,device)
model = Unet(3,64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr =1e-4)
results = 'test_results'
import os
os.makedirs(results,exist_ok=True)
[fixed_masked,fixed_mask],fixed_orig = next(iter(dl))
fixed_masked = fixed_masked.to(device)
fixed_mask = fixed_mask.to(device)
fixed_orig = fixed_orig.to(device)
def save_image(i,fixed_masked,fixed_mask,training,loss):
    with torch.no_grad():
        model.eval()
        fixed_pred,_ = model(fixed_masked, fixed_mask)
        torchvision.utils.save_image(fixed_pred,os.path.join(results,f'pred_{i}.jpg'),normalize=True)
        print(fixed_pred.min(), fixed_pred.max(), fixed_pred.mean())
        print(loss.item())

        if not training:
            print(fixed_orig.min(), fixed_orig.max(), fixed_orig.mean())
            print(fixed_masked.min(), fixed_masked.max(), fixed_masked.mean())
            print(fixed_mask.min(), fixed_mask.max(), fixed_mask.mean())
            torchvision.utils.save_image(fixed_masked, os.path.join(results, f'masked_{i}.jpg'),normalize=True)
            torchvision.utils.save_image(fixed_orig, os.path.join(results, f'orig_{i}.jpg'),normalize=True)
save_image(0,fixed_masked, fixed_mask,False,torch.tensor(0))

for i in range(200):
    i += 1
    print(f'{i}..')
    for [masked, mask], orig in dl:
        model.train()
        masked = masked.to(device)
        mask = mask.to(device)
        orig = orig.to(device)
        optimizer.zero_grad()
        pred, _ = model(masked, mask)
        loss = criterion(orig,mask,pred)
        loss.backward()
        optimizer.step()

    save_image(i,fixed_masked, fixed_mask,True, loss)

