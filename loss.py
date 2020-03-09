import torchvision
import torch

class VGG16:
    def __init__(self):
        vgg16 = torchvision.models.vgg16(pretrained=True)
        # pool layer 1 2 3
        def require_grad_false(m):
            for p in m.parameters():
                p.requires_grad = False
        self.enc1 = torch.nn.Sequential(*vgg16.features[:5])
        self.enc2 = torch.nn.Sequential(*vgg16.features[5:10])
        self.enc3 = torch.nn.Sequential(*vgg16.features[10:17])

        require_grad_false(self.enc1)
        require_grad_false(self.enc2)
        require_grad_false(self.enc3)
    def __call__(self, in_img):
        out1 = self.enc1(in_img)
        out2 = self.enc2(out1)
        out3 = self.enc3(out2)
        return out1, out2 , out3


class Loss(torch.nn.Module):
    def __init__(self, extractor):
        self.extractor = extractor
        self.l1 = torch.nn.L1Loss()
        self.loss_valid = None
        self.loss_hole = None
        self.loss_perceptual = None
        self.loss_style = None
        self.total_variation = None

    def forward(self, in_img,in_mask,out_img,gt):
        # mask 1 valid , 0 hole
        i_comp = out_img * (1-in_mask) + gt * in_mask
        self.loss_hole = self.l1((1-in_mask)*out_img,(1-in_mask)*gt)
        self.loss_valid = self.l1(in_mask*out_img, in_mask*gt)

        feature_out = self.extractor(out_img)
        feature_comp = self.extractor(i_comp)
        feature_gt = self.extractor(gt)
        self.loss_perceptual = self.perceptual_loss(feature_out, feature_comp, feature_gt)
        self.loss_style = self.style_loss(feature_out,feature_comp,feature_gt)
        # self.total_variation =


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
    # def total_variation(self, img):



img = torch.randn(4,3,512,512)
vgg = VGG16()
out1, out2 ,out3 = vgg(img)
print(out1.shape)
print(out2.shape)
print(out3.shape)











