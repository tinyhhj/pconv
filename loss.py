import torchvision
import torch

class VGG16:
    def __init__(self,device):
        vgg16 = torchvision.models.vgg16(pretrained=True)
        # pool layer 1 2 3
        def require_grad_false(m):
            for p in m.parameters():
                p.requires_grad = False
        self.enc1 = torch.nn.Sequential(*vgg16.features[:5]).to(device)
        self.enc2 = torch.nn.Sequential(*vgg16.features[5:10]).to(device)
        self.enc3 = torch.nn.Sequential(*vgg16.features[10:17]).to(device)
        self.mean = [0.485, 0.456, 0.406]
        self.std =  [0.229, 0.224, 0.225]
        self.normalize = torchvision.transforms.Normalize(self.mean,self.std)
        require_grad_false(self.enc1)
        require_grad_false(self.enc2)
        require_grad_false(self.enc3)

    def __call__(self, in_img):
        normalize = []
        for  img in in_img:
            normalize.append(self.normalize(img))
        in_img = torch.stack(normalize)
        out1 = self.enc1(in_img)
        out2 = self.enc2(out1)
        out3 = self.enc3(out2)
        return out1, out2 , out3


class Loss(torch.nn.Module):
    def __init__(self, extractor,device):
        super().__init__()
        self.extractor = extractor
        self.l1 = torch.nn.L1Loss()
        self.loss_valid = None
        self.loss_hole = None
        self.loss_perceptual = None
        self.loss_style = None
        self.loss_total_variation = None
        self.device = device

        # weight for loss
        self.loss_valid_weight = 1
        self.loss_hole_weight = 6
        self.loss_perceptual_weight = 0.05
        self.loss_style_weight = 120
        self.loss_total_variation_weight = 0.1

    def forward(self, gt,in_mask,out_img):
        # mask 1 valid , 0 hole
        i_comp = out_img * (1-in_mask) + gt * in_mask
        self.loss_hole = self.l1((1-in_mask)*out_img,(1-in_mask)*gt)
        self.loss_valid = self.l1(in_mask*out_img, in_mask*gt)

        feature_out = self.extractor(out_img)
        feature_comp = self.extractor(i_comp)
        feature_gt = self.extractor(gt)
        self.loss_perceptual = self.perceptual_loss(feature_out, feature_comp, feature_gt)
        self.loss_style = self.style_loss(feature_out,feature_comp,feature_gt)
        self.loss_total_variation = self.total_variation(i_comp, in_mask)
        # total loss
        return self.loss_valid * self.loss_valid_weight \
               + self.loss_hole * self.loss_hole_weight \
               + self.loss_perceptual * self.loss_perceptual_weight \
               + self.loss_style * self.loss_style_weight \
               + self.loss_total_variation * self.loss_total_variation_weight


    def perceptual_loss(self,out, comp, gt):
        loss = 0
        for i in range(len(out)):
            loss += self.l1(out[i], gt[i])
            loss += self.l1(comp[i], gt[i])
        return loss

    def style_loss(self, out, comp, gt):
        style_out = 0
        style_comp = 0
        for i in range(len(out)):
            # B C H W
            cc = out[i].shape[1] ** 2
            gram_out = self.gram_matrix(out[i])
            gram_gt = self.gram_matrix(gt[i])
            gram_comp = self.gram_matrix(comp[i])
            style_out += self.l1(gram_out,gram_gt) / cc
            style_comp += self.l1(gram_comp, gram_gt)/cc
        return style_out + style_comp


    #https: // github.com / pytorch / examples / blob / master / fast_neural_style / neural_style / utils.py
    def gram_matrix(self,y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram
    def dilation(self, i_comp, mask):
        b,c,h,w = i_comp.size()
        mb,mc,mh,mw = mask.size()

        # mask 1 valid , 0 hole
        kernel = torch.ones(mc,c,3,3).to(self.device)
        dilated_mask =  torch.nn.functional.conv2d(1-mask,kernel,padding=1)
        dilated_mask = torch.clamp(dilated_mask, 0, 1)
        return dilated_mask


    def total_variation(self, i_comp, mask):
        dilated_mask = self.dilation(i_comp, mask)
        i_comp = i_comp * dilated_mask
        return self.l1(i_comp[:,:,1:,:], i_comp[:,:,:-1,:]) + self.l1(i_comp[:,:,:,1:] , i_comp[:,:,:,:-1])

def test_mask_dilation():
    from PIL import Image
    import numpy as np

    png = Image.open('test.png')
    png = torchvision.transforms.ToTensor()(png)
    loss = Loss(None)
    # mask = torch.randn(1,1,64,64).float()
    # f = 0.1
    # print(torch.sum(mask>=f))
    # mask[mask<f] = 0.
    # mask[mask >= f] = 1.
    mask = png.unsqueeze(0)
    import matplotlib.pyplot as plt
    plt.title('black hole')
    plt.imshow(mask.squeeze().numpy(), cmap='gray')
    nmask = loss.dilation(mask,mask)
    plt.figure()
    plt.title('white bigger hole')
    plt.imshow(nmask.squeeze().numpy(), cmap='gray')
    plt.show()














