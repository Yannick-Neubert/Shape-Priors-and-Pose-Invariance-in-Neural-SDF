import os
import numpy as np
import scipy
import torch

from workspace import *

def get_normalization_params(img):
    '''
    calculates normalization parameters of a shape
    param img: input shape
    return: size, (xbar, ybar), (2nd order moments: l2, l3), (3rd order momements: l3, l4)
    '''
    img = img.astype('int64')

    def raw_moment(img, i_order, j_order):
        nrows, ncols = img.shape
        y_indices, x_indicies = np.mgrid[:nrows, :ncols]
        return (img * x_indicies**i_order * y_indices**j_order).sum()

    m00 = img.sum()
    m10 = raw_moment(img, 1,0)
    m01 = raw_moment(img, 0,1)
    m20 = raw_moment(img, 2,0)
    m11 = raw_moment(img, 1,1)
    m02 = raw_moment(img, 0,2)
    m30 = raw_moment(img, 3,0)
    m21 = raw_moment(img, 2,1)
    m12 = raw_moment(img, 1,2)
    m03 = raw_moment(img, 0,3)

    x_centroid = m10 / m00
    y_centroid = m01 / m00

    mu20 = (m20 - x_centroid * m10) / m00
    mu11 = (m11 - x_centroid * m01) / m00
    mu02 = (m02 - y_centroid * m01) / m00

    mu30 = (m30 - 3*x_centroid * m20 + 2*x_centroid**2 * m10) / m00
    mu21 = (m21 - 2*x_centroid * m11 - y_centroid * m20 + 2*x_centroid**2 * m01) / m00
    mu12 = (m12 - 2*y_centroid * m11 - x_centroid * m02 + 2*y_centroid**2 * m10) / m00
    mu03 = (m03 - 3*y_centroid * m02 + 2*y_centroid**2 * m01) / m00

    lnd2 = (mu20 - mu02)/2
    lnd3 = mu11

    lrd3 = mu30 + mu12
    lrd4 = mu21 + mu03
    
    return m00, (x_centroid, y_centroid), (lnd2, lnd3), (lrd3, lrd4)

def get_coordinate_grid(sidelen):
    pixel_coords = np.stack(np.mgrid[:sidelen,:sidelen], axis=-1)[None,...].astype(np.float32)
    pixel_coords /= sidelen    
    pixel_coords -= 0.5
    pixel_coords = np.reshape(pixel_coords, (-1,2))
    return pixel_coords

def img_to_sdf(img):
    neg_distances = scipy.ndimage.distance_transform_edt(img)
    sd_img = img - 1
    signed_distances = scipy.ndimage.distance_transform_edt(sd_img) - neg_distances
    signed_distances /= img.shape[1]
    return signed_distances

def sdf_to_img(sdf):
    img = np.zeros(sdf.shape)
    img[sdf <= 0] = 1
    return img

def samples_to_img(sdf_samples):
    # if len(sdf_samples.shape) == 1:
    #     return sdf_to_img(sdf_samples.reshape(sidelen, sidelen))
    if True:
        _, indices = np.unique(sdf_samples[:, 0:2], axis=0, return_index=True)      # get unique indices
        sdf_samples = sdf_samples[indices]                                          # filter duplicate rows
        sdf_samples = sdf_samples[np.lexsort((sdf_samples[:,1], sdf_samples[:,0]))] # unshuffle coordinates
        sdf = sdf_samples[:,2].reshape(sidelen, sidelen)
    return sdf_to_img(sdf)

def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1)).item()

def decode_latent(decoder, lat_vec, g=torch.tensor([0])):
    x = torch.tensor(get_coordinate_grid(sidelen))
    z = lat_vec.repeat(x.shape[0], 1)
    pred = decoder(g, x, z)
    return sdf_to_img(pred.detach().reshape(sidelen, sidelen))
