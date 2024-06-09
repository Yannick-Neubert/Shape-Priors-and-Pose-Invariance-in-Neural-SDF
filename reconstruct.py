from workspace import *
from utils import *

import data
from decoder import Decoder

import torch
from tqdm import tqdm
import time

start = time.time()
test_seen = data.SDFDataset(data_dir, "test_seen", order, load_ram=True)
test_unseen = data.SDFDataset(data_dir, "test_unseen", order, load_ram=True)
end = time.time()
print("Data loaded in ", end-start, " seconds")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


decoder = Decoder().to(device)
cp = torch.load(os.path.join(model_dir, model_params_subdir, "latest.pth"), map_location=device)
decoder.load_state_dict(cp["model_state_dict"])
decoder.eval()

loss_l1 = torch.nn.L1Loss(reduction="sum")

logs_cp = torch.load(os.path.join(model_dir, model_logs_subdir, "latest.pth"), map_location=device)
max_norm = logs_cp["lat_mag"][-1]

def enforce_max_norm(tensor, max_norm):
    with torch.no_grad():
        norm = tensor.norm(dim=-1, keepdim=True)
        desired = torch.clamp(norm, max=max_norm)
        tensor *= (desired / norm)


def reconstruct(data, infer_pose=False):
    latent = torch.ones(1, latent_size).normal_(0, sigma).to(device)
    latent.requires_grad = True
    if infer_pose:
        g = torch.pi * (torch.rand(1) - 0.5)
        g = g.to(device)
        g.requires_grad = True
        optimizer = torch.optim.Adam([latent, g], lr=lrz)
    else:
        optimizer = torch.optim.Adam([latent], lr=lrz)

    loss_l1 = torch.nn.L1Loss()

    sdf_data, _, _ = data

    num_iters = num_pose_inference_iters if pose_inference else num_recon_iters
    for it in range(num_iters+1):
        decoder.eval()
        sdf_data, moments, _ = data
        sdf_data = sdf_data.reshape(-1, 3).to(device)

        sdf_gt = sdf_data[:, 2].unsqueeze(1)
        x = sdf_data[:, 0:2]
        if not infer_pose:
            if order == 2:
                l2, l3 = moments[0], moments[1]
                g = (torch.atan2(l3, l2)/2).unsqueeze(0).to(device)
            else:
                l3, l4 = moments[0], moments[1]
                g = torch.atan2(l4, l3).unsqueeze(0).to(device)

        optimizer.zero_grad()

        z = latent.expand(num_samp_per_scene, -1)

        pred_sdf = decoder(g, x, z)


        l1 = loss_l1(pred_sdf, sdf_gt)
        loss = l1 + code_reg_lambda * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if pose_inference:
            enforce_max_norm(latent, max_norm)

    return latent.detach().cpu()

if __name__ == "__main__":
    save_dir = os.path.join(model_dir, reconstructions_subdir)
    suffix = "_pose_inferred" if pose_inference else ""

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    lat_vecs = []
    for i in tqdm(range(len(test_seen))):
        datum = test_seen[i]
        z = reconstruct(datum, infer_pose=pose_inference)
        lat_vecs.append(z)

    lat_vecs = torch.stack(lat_vecs, dim=1)[0]
    emb = torch.nn.Embedding(len(lat_vecs), latent_size)
    emb.weight.data.copy_(lat_vecs)
    lat_vecs = emb

    torch.save(lat_vecs, os.path.join(save_dir, f"seen{suffix}.pth"))

    lat_vecs = []
    for i in tqdm(range(len(test_unseen))):
        datum = test_unseen[i]
        z = reconstruct(datum, infer_pose=pose_inference)
        lat_vecs.append(z)

    lat_vecs = torch.stack(lat_vecs, dim=1)[0]
    emb = torch.nn.Embedding(len(lat_vecs), latent_size)
    emb.weight.data.copy_(lat_vecs)
    lat_vecs = emb
    torch.save(lat_vecs, os.path.join(save_dir, f"unseen{suffix}.pth"))