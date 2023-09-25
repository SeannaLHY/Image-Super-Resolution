import torch
import numpy as np
from PIL import Image

def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def denormalize(img):
    img = img.mul(255.0).clamp(0.0, 255.0)
    return img


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr


def calc_psnr(img1, img2, max=255.0):
    return 10. * ((max ** 2) / ((img1 - img2) ** 2).mean()).log10()

def calc_ssim(im1,im2):
    mu1=im1.mean()
    mu2=im2.mean()
    sigma1= np.sqrt(((im1-mu1)**2).mean().cpu())
    sigma2= np.sqrt(((im2-mu2)**2).mean().cpu())
    sigma12=((im1-mu1)*(im2-mu2)).mean()
    k1,k2,L=0.01,0.03,1
    C1=(k1*L)**2
    C2=(k2*L)**2
    C3=C2/2
    l12=(2*mu1*mu2+C1)/(mu1**2+mu2**2+C1)
    c12=(2*sigma1*sigma2+C2)/(sigma1**2+sigma2**2+C2)
    s12=(sigma12+C3)/(sigma1*sigma2+C3)
    ssim=l12*c12*s12
    return ssim


def nearestNeighborScaling(source, newWid, newHt):
    target = Image.new('RGB', (newWid, newHt))
    width = source.width
    height = source.height
    for x in range(0, newWid):
      for y in range(0, newHt):
        srcX = int( round( float(x) / float(newWid) * float(width) ) )
        srcY = int( round( float(y) / float(newHt) * float(height) ) )
        srcX = min( srcX, width-1)
        srcY = min( srcY, height-1)
        srcColor = source.getpixel((srcX,srcY))
        target.putpixel((x,y),srcColor)
    return target



class AverageMeter(object):
    def __init__(self):
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
