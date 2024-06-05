import os
import numpy as np
from tqdm import tqdm

import utils
from workspace import *

from PIL import Image

import matplotlib.pyplot as plt

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

sdf_samples_dir = os.path.join(data_dir, sdf_samples_subdir, source_name)
if not os.path.isdir(sdf_samples_dir):
    os.makedirs(sdf_samples_dir)

normalization_param_dir = os.path.join(data_dir, normalization_param_subdir, source_name)
if not os.path.isdir(normalization_param_dir):
    os.makedirs(normalization_param_dir)


remove_files = ["brick", "phone", "device", "pencil", "spring", "watch"]

files = os.listdir(source_dir)
files = [file for file in files if all(rf not in file for rf in remove_files)]


for file in tqdm(files):
    sdf_savepath = os.path.join(sdf_samples_dir, os.path.splitext(file)[0])
    norm_savepath = os.path.join(normalization_param_dir, os.path.splitext(file)[0])

    if (os.path.isfile(sdf_savepath + ".npy") and os.path.isfile(norm_savepath + ".npz")):
        continue

    pimg = Image.open(os.path.join(source_dir, file)).resize((sidelen, sidelen))
    img = np.array(pimg)
    img[img > 0] = 1

    #center image and rotate randomly
    ys, xs = np.nonzero(img)
    xm = np.mean(xs)
    ym = np.mean(ys)
    pimg = pimg.rotate(np.random.randint(0, 360), center=(xm, ym), translate=(-xm+sidelen/2,-ym+sidelen/2))
    img = np.array(pimg)
    img[img > 0] = 1

    #generate sdf samples
    sdf_transform = utils.img_to_sdf(img)
    coordinates = utils.get_coordinate_grid(sidelen)
    
    sdf_samples = np.append(coordinates, sdf_transform.reshape(-1, 1), axis=1)
    surface_samples = sdf_samples[np.abs(sdf_transform.reshape(-1)) < 0.1]
    surface_samples = surface_samples[np.resize(np.random.permutation(len(surface_samples)), num_surface_samples)]
    sdf_samples = np.append(sdf_samples, surface_samples, axis=0)

    size, center, moments2, moments3 = utils.get_normalization_params(img)

    np.save(sdf_savepath, sdf_samples)
    np.savez(norm_savepath, size=size, center=center, moments2=moments2, moments3=moments3)