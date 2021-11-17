import os
import numpy as np
import pydensecrf.densecrf as dcrf
from skimage.io import imread, imsave
import cv2
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
from tqdm import tqdm
import argparse

# arg
parser = argparse.ArgumentParser()

parser.add_argument("--GauSxy", type=int, default=3)
parser.add_argument("--BilSxy", type=int, default=80)
parser.add_argument("--BilSrgb", type=int, default=13)
parser.add_argument("--test_path", type=str, default='./test')
parser.add_argument("--ann_path", type=str, default='./result')
parser.add_argument("--save_path", type=str, default='./crf_output')

args = parser.parse_args()

test_path = args.test_path
ann_path = args.ann_path
save_path = args.save_path

# CRF
os.makedirs(save_path, exist_ok=True)

for i in tqdm(range(400)):
    # # apply CRF
    img = imread(os.path.join(test_path, str(i).zfill(4)+'.jpg'))
    img = cv2.resize(img, (512, 512))
    anno_rgb = imread(os.path.join(ann_path, str(i).zfill(4)+'.png')).astype(np.uint32)
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)

    colors, labels = np.unique(anno_lbl, return_inverse=True)

    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16

    n_labels = len(set(labels.flat)) #- int(HAS_UNK)

    # Example using the DenseCRF2D code
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(args.GauSxy, args.GauSxy), compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(args.BilSxy, args.BilSxy), srgb=(args.BilSrgb, args.BilSrgb, args.BilSrgb), rgbim=img,
                            compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(45)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)
    MAP = colorize[MAP, :]

    crf_output = MAP.reshape((img.shape[0],img.shape[1], 3))

    imsave(os.path.join(save_path, str(i).zfill(4)+'.png'), crf_output)