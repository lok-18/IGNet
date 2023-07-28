import os
import argparse
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import fusion_dataset
from model_gnn import backbone
from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image
from rgb2ycbcr import RGB2YCrCb, YCrCb2RGB

import time

def test():
    fusion_model_path = './model/IGNet.pth'
    fusion_model = backbone()
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.gpu >= 0:
        fusion_model.to(device)

    fusion_model.load_state_dict(torch.load(fusion_model_path))
    print('Model loading...')

    ir_path = './test_images/ir'
    vis_path = './test_images/vis'

    test_dataset = fusion_dataset(ir_path=ir_path, vis_path=vis_path)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)

    with torch.no_grad():
        for it, (image_vis, image_ir, name) in enumerate(test_loader):
            image_vis = Variable(image_vis)
            image_ir = Variable(image_ir)
            if args.gpu >= 0:
                image_vis = image_vis.to(device)
                image_ir = image_ir.to(device)
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            # inputs = [image_vis_ycrcb, image_ir]
            logits = fusion_model(image_vis_ycrcb, image_ir)  # inputs
            fusion_ycrcb = torch.cat(
                (logits, image_vis_ycrcb[:, 1:2, :, :], image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)

            st = time.time()
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                ed = time.time()
                print('file_name: {0}'.format(save_path))
                print('Time:', ed - st)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test backbone')
    parser.add_argument('--model_name', '-M', type=str, default='backbone')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    fused_dir = './results/'
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    test()
    print('Test finish!')

