import cv2
from random import randint, seed
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from copy import deepcopy


class MaskGenerator:
    def __init__(self, size,rand_seed =None, file_path= None):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise ValueError('size must be int or tuple')
        self.channel = 3
        if rand_seed:
            seed(rand_seed)
        if file_path:
            files = [f for f in os.listdir(file_path)]
            files = [os.path.join(file_path,f) for f in files if any(filetype in f.lower() for filetype in ['.jpg','.png','.jpeg'])]
            self.files = files
    def random_mask(self):
        height, width = self.size
        mask = np.zeros(( height, width ,self.channel),np.uint8)
        thick = int((height + width) * 0.03)

        for _ in range(randint(1,20)):
            x1,x2 = randint(1,width), randint(1,width)
            y1,y2 = randint(1,height), randint(1,height)
            thickness = randint(3,thick)
            cv2.line(mask,(x1,y1),(x2,y2),(1,1,1),thickness)
        for _ in range(randint(1,20)):
            x1,y1 = randint(1,width), randint(1,height)
            radius = randint(3, thick)
            cv2.circle(mask, (x1,y1),radius,(1,1,1),-1)
        for _ in range(randint(1,20)):
            x1,y1 = randint(1,width), randint(1,height)
            s1,s2 = randint(1,width), randint(1,height)
            a1,a2,a3 = randint(3,180), randint(3,180), randint(3,180)
            thickness = randint(3,thick)
            cv2.ellipse(mask,(x1,y1),(s1,s2),a1,a2,a3,(1,1,1),thickness)
        return 1 - mask
    def load_mask(self):
        mask = cv2.imread(np.random.choice(self.files,1)[0])

        rand = np.random.randint(-180,180)
        M = cv2.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2),rand,1.5)
        mask= cv2.warpAffine(mask,M,(mask.shape[1],mask.shape[0]))

        rand = np.random.randint(5,47)
        kernel = np.ones((rand,rand),np.uint8)
        mask = cv2.erode(mask,kernel,iterations=1)

        x = np.random.randint(0,mask.shape[1] - self.size[1])
        y = np.random.randint(0,mask.shape[0] - self.size[0])
        mask = mask[y:y+self.size[0], x:x+self.size[1]]

        return (mask > 1).astype(np.uint8)
def test():
    g = MaskGenerator(512,43,'irregular_mask/disocclusion_img_mask')
    _,axes= plt.subplots(5,5,figsize=(20,20))
    axes = list(itertools.chain.from_iterable(axes))

    for i in range(len(axes)):
        mask = g.load_mask()
        axes[i].imshow(mask*255)
    plt.show()
def test_mask():
    img = cv2.imread('sample_image.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    shape = img.shape
    print(shape)

    g = MaskGenerator((shape[0],shape[1]),43)
    mask = g.random_mask()
    masked_img = deepcopy(img)
    masked_img[mask==0] = 255

    _, axes = plt.subplots(1,3,figsize=(20,5))

    axes[0].imshow(img)
    axes[1].imshow(mask*255)
    axes[2].imshow(masked_img)
    import torch
    import torchvision

    totensor = torchvision.transforms.ToTensor()
    formatted_img = masked_img
    formatted_mask = np.expand_dims(mask,0)
    formatted_img = totensor(formatted_img).unsqueeze(0)
    formatted_mask = torch.from_numpy(formatted_mask.transpose(0, 3, 1, 2)).float()
    print(torch.unique(formatted_img))
    print(formatted_mask.size())

    import unet
    self = unet.Unet(3,64)
    # self(torch.randn(1,3,512,512), torch.randn(1,3,512,512))
    out_img1, out_mask1 = self.pconv1(formatted_img, formatted_mask)
    out_img2, out_mask2 = self.pconv2(out_img1, out_mask1)
    out_img3, out_mask3 = self.pconv3(out_img2, out_mask2)
    out_img4, out_mask4 = self.pconv4(out_img3, out_mask3)
    out_img5, out_mask5 = self.pconv5(out_img4, out_mask4)
    out_img6, out_mask6 = self.pconv6(out_img5, out_mask5)
    out_img7, out_mask7 = self.pconv7(out_img6, out_mask6)
    out_img8, out_mask8 = self.pconv8(out_img7, out_mask7)
    #
    _,axes = plt.subplots(2,4,figsize=(20,5))
    axes[0][0].imshow(out_mask1[0,0,:,:],cmap='gray',vmin=0,vmax=1)
    axes[0][1].imshow(out_mask2[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
    axes[0][2].imshow(out_mask3[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
    axes[0][3].imshow(out_mask4[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
    axes[1][0].imshow(out_mask5[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
    axes[1][1].imshow(out_mask6[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
    axes[1][2].imshow(out_mask7[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
    axes[1][3].imshow(out_mask8[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
    plt.show()

test_mask()



