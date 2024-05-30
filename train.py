from decoder import Decoder
from utils import *
import data

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

import matplotlib.pyplot as plt ###

##checkpoint logs

def save_model(epoch, decoder, lat_vecs, optimizer, name):
    path = os.path.join(model_dir, model_params_subdir)
    if not os.path.isdir(path):
        os.makedirs(path)

    torch.save({
        'epoch': epoch,
        'model_state_dict': decoder.state_dict(),
        'lat_vecs': lat_vecs.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(path, name))

def save_logs(loss, lat_mag, moment, time, name):
    path = os.path.join(model_dir, model_logs_subdir)
    if not os.path.isdir(path):
        os.makedirs(path)
    
    torch.save({
        'loss': loss,
        'lat_mag': lat_mag,
        'moment': moment,
        'time': time
    }, os.path.join(path, name))


decoder = Decoder()
train = data.SDFDataset(data_dir, "train")
train_loader = DataLoader(train, shuffle=True, batch_size=batch_size)
num_shapes = len(train)

lat_vecs = torch.nn.Embedding(num_shapes, latent_size, max_norm=1)
torch.nn.init.normal_(lat_vecs.weight.data, 0.0, sigma / np.sqrt(latent_size))

loss_l1 = torch.nn.L1Loss(reduction="sum")
optimizer_all = torch.optim.Adam(
    [
        {
            "params": decoder.parameters(),
            "lr": lr,
        },
        {
            "params": lat_vecs.parameters(),
            "lr": lr,
        },
    ]
)
optimizer_angle = torch.optim.Adam(decoder.parameters(), lr=lra)

loss_log = []
lat_mag_log = []
time_log = []
mom_log = []


for epoch in (range(num_epochs + 1)):
    decoder.train()

    start = time.time()
    loss_total = 0

    for sdf_data, moments, indices in train_loader:
        optimizer_all.zero_grad()

        sdf_data.requires_grad = False
        moments.requires_grad = False
        
        sdf_data = sdf_data.reshape(-1, 3)
        num_sdf_samples = sdf_data.shape[0]

        x = sdf_data[:, 0:2]
        sdf_gt = sdf_data[:, 2].unsqueeze(1)
        l3, l4 = moments[:, 0], moments[:, 1]
        g = torch.atan2(l4, l3).unsqueeze(1).repeat(1, num_samp_per_scene).view(-1)
        indices = indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1)
        batch_vecs = lat_vecs(indices)

        pred_sdf = decoder(g, x, batch_vecs)

        sdf_loss = loss_l1(pred_sdf, sdf_gt) / num_sdf_samples
        reg_loss = code_reg_lambda * torch.sum(torch.norm(batch_vecs, dim=1)) / num_sdf_samples

        loss = sdf_loss + reg_loss
        loss.backward(retain_graph=True)
        loss_total += loss.item()
        optimizer_all.step()

    end = time.time()
    seconds_elapsed = end - start

    loss_log.append(loss_total)
    lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))
    time_log.append(seconds_elapsed)

####################################################################################

    norm = lat_mag_log[-1]
    bound = np.sqrt(3*norm**2 / latent_size)
    rand_vecs = torch.nn.Embedding(num_shapes, latent_size, max_norm=1)
    torch.nn.init.uniform_(rand_vecs.weight.data, -bound, bound)

    moment_loss = 0
    all_indices = range(num_shapes)
    batches = [all_indices[i:i + batch_size] for i in range(0, num_shapes, batch_size)]
    for batch in (batches):
        y_coords, x_coords = torch.meshgrid(torch.arange(sidelen, dtype=torch.float32), torch.arange(sidelen, dtype=torch.float32), indexing="xy")
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1)[None, ...]
        pixel_coords /= sidelen
        pixel_coords -= 0.5
        pixel_coords = pixel_coords.view(-1, 2)
        x = pixel_coords

        indices = torch.tensor(batch).unsqueeze(-1).repeat(1, x.shape[0]).view(-1)
        x = x.repeat(len(batch), 1)

        z = rand_vecs(indices)
        g = torch.tensor([0])

        x.requires_grad = False
        g.requires_grad = False
        z.detach()

        pred = decoder(g, x, z)

        i = 0
        for _ in batch:
            pred_sdf = pred[i * sidelen**2:(i+1)*sidelen**2]
            pred_sdf = pred_sdf.reshape((sidelen, sidelen))
            pred_sdf[pred_sdf <= 0] = 0
            pred_sdf[pred_sdf > 0] = 1
            pred_sdf = 1 - pred_sdf
            img = pred_sdf
            i += 1
        
            def raw_moment(img, i_order, j_order):
                nrows, ncols = img.shape
                y_indices, x_indices = torch.meshgrid(torch.arange(nrows), torch.arange(ncols), indexing='ij')
                return (img * (x_indices**i_order) * (y_indices**j_order)).sum()

            m00 = img.sum()
            if m00 == 0:
                continue
            
            m10 = raw_moment(img, 1, 0)
            m01 = raw_moment(img, 0, 1)
            m20 = raw_moment(img, 2, 0)
            m11 = raw_moment(img, 1, 1)
            m02 = raw_moment(img, 0, 2)
            m30 = raw_moment(img, 3, 0)
            m21 = raw_moment(img, 2, 1)
            m12 = raw_moment(img, 1, 2)
            m03 = raw_moment(img, 0, 3)

            x_centroid = m10 / m00
            y_centroid = m01 / m00

            mu30 = (m30 - 3 * x_centroid * m20 + 2 * x_centroid**2 * m10) / m00
            mu21 = (m21 - 2 * x_centroid * m11 - y_centroid * m20 + 2 * x_centroid**2 * m01) / m00
            mu12 = (m12 - 2 * y_centroid * m11 - x_centroid * m02 + 2 * y_centroid**2 * m10) / m00
            mu03 = (m03 - 3 * y_centroid * m02 + 2 * y_centroid**2 * m01) / m00

            l3 = mu30 + mu12
            l4 = mu21 + mu03

            angle = torch.rad2deg(torch.atan2(l4, l3))
            moment_loss += torch.abs(angle)
            optimizer_angle.zero_grad()
            print(angle)
            angle.backward()
            print("BACKWARD!")
            optimizer_angle.step()


    mom_log.append(moment_loss)
    if epoch > 1 and epoch % 50 == 0:
        save_model(epoch, decoder, lat_vecs, optimizer_all, f"cp_{epoch}.pth")
        save_logs(loss_log, lat_mag_log, mom_log, time_log, f"log_{epoch}.pth")

save_model(epoch, decoder, lat_vecs, optimizer_all, "latest.pth")
save_logs(loss_log, lat_mag_log, time_log, mom_log, "latest.pth")