import torch
from partialconv2d import PartialConv2d

class Pconv(PartialConv2d):
    def __init__(self,*args, batch=None,act_fn = None, **kwargs):
        super().__init__(*args,**kwargs)
        self.batch = batch
        self.act_fn = act_fn
    def forward(self, input, mask_in=None):
        out_img, out_mask = super().forward(input,mask_in)
        if self.batch:
            out_img = self.batch(out_img)
        if self.act_fn:
            self.act_fn(out_img)
        return out_img, out_mask




class Unet(torch.nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        def batch(cin):
            return torch.nn.BatchNorm2d(cin)
        def active(encoder=True):
            if encoder:
                return torch.nn.ReLU(True)
            else:
                return torch.nn.LeakyReLU(0.2, True)
        # 64
        self.pconv1 = Pconv(cin,cout,7,2,3,multi_channel=True, return_mask=True, act_fn=active())
        # 128
        self.pconv2 = Pconv(cout,cout*2,5,2,2,multi_channel=True, return_mask=True,batch=batch(cout*2), act_fn=active())
        # 256
        self.pconv3 = Pconv(cout*2, cout * 4, 5, 2, 2, multi_channel=True, return_mask=True,
                            batch=batch(cout* 4), act_fn=active())
        # 512
        self.pconv4 = Pconv(cout * 4, cout * 8, 3, 2, 1, multi_channel=True, return_mask=True,
                            batch=batch(cout* 8), act_fn=active())
        for i in range(5,5+4):
            name = f'pconv{i}'
            setattr(self, name,Pconv(cout * 8, cout * 8, 3, 2, 1, multi_channel=True, return_mask=True,
                            batch=batch(cout*8), act_fn=active()))

        # 512
        for i in range(9,13):
            name = f'pconv{i}'
            setattr(self,name,Pconv(cout*16, cout*8,3,1,1,multi_channel=True, return_mask=True,
                            batch=batch(cout*8), act_fn=active(False)))
        self.pconv13 = Pconv(cout*12, cout*4,3,1,1,batch=batch(cout*4),act_fn=active(False),multi_channel=True, return_mask=True)
        self.pconv14 = Pconv(cout*6, cout*2,3,1,1,batch=batch(cout*2),act_fn=active(False),multi_channel=True, return_mask=True)
        self.pconv15 = Pconv(cout*3, cout,3,1,1,batch=batch(cout),act_fn=active(False),multi_channel=True, return_mask=True)
        self.pconv16 = Pconv(cout+cin,cin,3,1,1,multi_channel=True, return_mask=True)

    def forward(self, in_img, in_mask):
        # encoder
        out_img1,out_mask1 = self.pconv1(in_img,in_mask)
        out_img2,out_mask2 = self.pconv2(out_img1,out_mask1)
        out_img3,out_mask3 = self.pconv3(out_img2,out_mask2)
        out_img4,out_mask4 = self.pconv4(out_img3,out_mask3)
        out_img5,out_mask5 = self.pconv5(out_img4,out_mask4)
        out_img6,out_mask6 = self.pconv6(out_img5,out_mask5)
        out_img7,out_mask7 = self.pconv7(out_img6,out_mask6)
        out_img8,out_mask8 = self.pconv8(out_img7,out_mask7)

        #decoder
        def upsample(img, mask):
            return torch.nn.Upsample(scale_factor=2)(img),torch.nn.Upsample(scale_factor=2)(mask)
        def concat(img1,img2,mask1,mask2):
            return torch.cat([img1,img2],dim=1),torch.cat([mask1,mask2],dim=1)

        out_img8,out_mask8 = upsample(out_img8,out_mask8)
        out_img8,out_mask8 = concat(out_img8, out_img7, out_mask8, out_mask7)
        out_img8,out_mask8 = self.pconv9(out_img8,out_mask8)

        out_img8, out_mask8 = upsample(out_img8,out_mask8)
        out_img8, out_mask8 = concat(out_img8,out_img6,out_mask8,out_mask6)
        out_img8, out_mask8 = self.pconv10(out_img8, out_mask8)

        out_img8, out_mask8 = upsample(out_img8, out_mask8)
        out_img8, out_mask8 = concat(out_img8, out_img5, out_mask8, out_mask5)
        out_img8, out_mask8 = self.pconv11(out_img8, out_mask8)

        out_img8, out_mask8 = upsample(out_img8, out_mask8)
        out_img8, out_mask8 = concat(out_img8, out_img4, out_mask8, out_mask4)
        out_img8, out_mask8 = self.pconv12(out_img8, out_mask8)

        out_img8, out_mask8 = upsample(out_img8, out_mask8)
        out_img8, out_mask8 = concat(out_img8, out_img3, out_mask8, out_mask3)
        out_img8, out_mask8 = self.pconv13(out_img8, out_mask8)

        out_img8, out_mask8 = upsample(out_img8, out_mask8)
        out_img8, out_mask8 = concat(out_img8, out_img2, out_mask8, out_mask2)
        out_img8, out_mask8 = self.pconv14(out_img8, out_mask8)

        out_img8, out_mask8 = upsample(out_img8, out_mask8)
        out_img8, out_mask8 = concat(out_img8, out_img1, out_mask8, out_mask1)
        out_img8, out_mask8 = self.pconv15(out_img8, out_mask8)

        out_img8, out_mask8 = upsample(out_img8, out_mask8)
        out_img8, out_mask8 = concat(out_img8, in_img, out_mask8, in_mask)
        out_img8, out_mask8 = self.pconv16(out_img8, out_mask8)

        return out_img8, out_mask8

if __name__ =='__main__':
    model = Unet(3,64)
    img = torch.randn(1,3,512,512)
    mask  =torch.randn(1,1,512,512)
    mask = torch.cat([mask,mask,mask],dim=1)
    print(img.shape, mask.shape)
    out_img, out_mask = model(img,mask)
    print(out_img.shape, out_mask.shape)







