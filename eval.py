from workspace import *
from utils import *
import data

from decoder import Decoder

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

device = torch.device("cpu")
decoder = Decoder()

# epoch = 50
# cp = torch.load(os.path.join(model_dir, model_params_subdir, f"cp_{epoch}.pth"), map_location=device)
# logs_cp = torch.load(os.path.join(model_dir, model_logs_subdir, f"log_{epoch}.pth"), map_location=device)
cp = torch.load(os.path.join(model_dir, model_params_subdir, "latest.pth"), map_location=device)
logs_cp = torch.load(os.path.join(model_dir, model_logs_subdir, "latest.pth"), map_location=device)

decoder.load_state_dict(cp["model_state_dict"])
num_shapes = cp["lat_vecs"]["weight"].shape[0]
train_vecs = torch.nn.Embedding(num_shapes, latent_size, max_norm=1)
train_vecs.load_state_dict(cp["lat_vecs"])

loss_log = logs_cp["loss"]
lat_mag_log = logs_cp["lat_mag"]
mom_log = logs_cp["moment"]
time_log = logs_cp["time"]

loss_l1 = torch.nn.L1Loss(reduction="sum")

train = data.SDFDataset(data_dir, "train", order, load_ram=False)
test_seen = data.SDFDataset(data_dir, "test_seen", order, load_ram=True)
test_unseen = data.SDFDataset(data_dir, "test_unseen", order, load_ram=True)

mag = (get_mean_latent_vector_magnitude(train_vecs))
bound = np.sqrt(3*mag**2 / latent_size)
rand_vecs = torch.nn.Embedding(num_shapes, latent_size, max_norm=1)
torch.nn.init.uniform_(rand_vecs.weight.data, -bound, bound)

if pose_inference:
    seen_vecs = torch.load(os.path.join(model_dir, reconstructions_subdir, "seen_pose_inferred.pth"), map_location=device)
    unseen_vecs = torch.load(os.path.join(model_dir, reconstructions_subdir, "unseen_pose_inferred.pth"), map_location=device)
else:
    seen_vecs = torch.load(os.path.join(model_dir, reconstructions_subdir, "seen.pth"), map_location=device)
    unseen_vecs = torch.load(os.path.join(model_dir, reconstructions_subdir, "unseen.pth"), map_location=device)

decoder.eval()

def eval_recon(gt, latent):
    loss_recon = 0
    num_shapes = len(gt)

    for i in tqdm(range(num_shapes)):
        sdf_gt, (m1, m2), _ = gt[i]
        _, indices = np.unique(sdf_gt[:, 0:2], axis=0, return_index=True)
        sdf_gt = sdf_gt[indices]                                         
        sdf_gt = sdf_gt[np.lexsort((sdf_gt[:,1], sdf_gt[:,0]))]
        sdf_gt = sdf_gt[:,2]

        if order == 2:
            l2, l3 = m1, m2
            g = (torch.atan2(l3, l2)/2)
        else:
            l3, l4 = m1, m2
            g = (torch.atan2(l4, l3))

        x = torch.tensor(get_coordinate_grid(sidelen))
        g = torch.tensor([g])
        z = latent(torch.tensor([i]))
        z = z.repeat(x.shape[0], 1)

        pred = decoder(g, x, z)
        pred = pred.detach().flatten()

        loss_recon += loss_l1(sdf_gt, pred)

    loss_avg = loss_recon / num_shapes
    return loss_avg

def eval_norm(latent):
    loss_norm = 0
    num_shapes = latent.num_embeddings

    for i in tqdm(range(num_shapes)):
        z = latent(torch.tensor([i]))

        img = decode_latent(decoder, z)

        if order == 2:
            _, _, (l2, l3), _ = get_normalization_params(img)
            angle = abs(np.degrees(np.arctan2(l3, l2)/2))
        else:
            _, _, _, (l3, l4) = get_normalization_params(img)
            angle = abs(np.degrees(np.arctan2(l4, l3)))

        loss_norm += angle

    loss_avg = loss_norm / num_shapes
    return loss_avg

def plot_recon(latent, rows, cols, indices):
    rows, cols = 5, 10
    fig, axs = plt.subplots(rows, cols, figsize=(14, 4))
    x = torch.tensor(get_coordinate_grid(sidelen))
    g = torch.tensor([0])

    # shapes = np.random.permutation(num_shapes)
    shapes = indices
    for i, ax in enumerate(axs.ravel()):
    # if True:
    # for i in tqdm(range(num_shapes)):
        z = train_vecs(torch.tensor(shapes[i]))
        # z = rand_vecs(torch.tensor(shapes[i]))

        z = z.repeat(x.shape[0], 1)

        pred = decoder(g, x, z)

        sdf = np.append(x.numpy(), pred.detach().numpy(), axis=1)
        img = samples_to_img(sdf)

        ax.imshow(img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def plot_latent():
    cp_freq = 50
    epochs = range(cp_freq, num_epochs+1, cp_freq)
    indices = np.arange(0, len(train), 200)

    fig, axs = plt.subplots(len(epochs), len(indices), figsize=(6,6))

    for epoch, ax in zip(epochs, axs):
        # if epoch == 0:
        #     for i, ax in zip(indices, ax.ravel()):
        #         sdf, _, _ = train[i]
        #         img = samples_to_img(sdf)
        #         ax.imshow(img, cmap="gray")
        #         ax.set_xticks([])
        #         ax.set_yticks([])
        #     continue

        cp = torch.load(os.path.join(model_dir, model_params_subdir, f"cp_{epoch}.pth"), map_location=device)
        logs_cp = torch.load(os.path.join(model_dir, model_logs_subdir, f"log_{epoch}.pth"), map_location=device)

        decoder.load_state_dict(cp["model_state_dict"])
        num_shapes = cp["lat_vecs"]["weight"].shape[0]
        train_vecs = torch.nn.Embedding(num_shapes, latent_size, max_norm=1)
        train_vecs.load_state_dict(cp["lat_vecs"])

        mag = (get_mean_latent_vector_magnitude(train_vecs))
        bound = np.sqrt(3*mag**2 / latent_size)
        rand_vecs = torch.nn.Embedding(num_shapes, latent_size, max_norm=1)
        torch.nn.init.uniform_(rand_vecs.weight.data, -bound, bound)

        decoder.eval()

        for i, ax in zip(indices, ax.ravel()):
            z = rand_vecs(torch.tensor([i]))
            img = decode_latent(decoder, z)
            ax.imshow(img, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if epoch == num_epochs:
                _, _, (l2, l3), _ = get_normalization_params(img)
                angle = (np.degrees(np.arctan2(l3, l2)/2))
                ax.set_xlabel(r"$\theta$ = " + f"%.2f" % angle)
    
    for e, ax in zip(epochs, axs.T[0].ravel()):
        ax.set_ylabel(e)

    # axs[0,0].set_ylabel("gt")
    plt.show()
    
def interpolate():
    ii = [7, 9, 12, 31]
    n = 8
    fig, axes = plt.subplots(len(ii), n+2+1, figsize=(14, 4))
    for id, axs in zip(ii, axes):
        i1, i2 = 18*id, 18*id+1
        z1 = train_vecs(torch.tensor([i1]))
        z2 = train_vecs(torch.tensor([i2]))

        zs = [z1]
        for i in range(1, n + 1):
            t = i / (n + 1)
            z = torch.lerp(z1, z2, t)
            zs.append(z)

        zs.append(z2)
        
        for ax, z in zip(axs.ravel(), zs):
            img = decode_latent(decoder, z)
            _, _, (l2, l3), _ = get_normalization_params(img)
            angle = np.degrees(np.arctan2(l3, l2)/2)
            
            ax.set_title(r"%.2f" % angle)
            ax.imshow(img, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
        
        zs = train_vecs(torch.arange(18*id, 18*(id+1)))
        z = torch.mean(zs, dim=0)
        im = decode_latent(decoder, z)

        _, _, (l2, l3), _ = get_normalization_params(im)   
        angle = abs(np.degrees(np.arctan2(l3, l2)/2))
        axs[-1].imshow(im)
        axs[-1].set_title(f"%.2f" % angle)
        axs[-1].set_xticks([])
        axs[-1].set_yticks([])

    plt.tight_layout()
    plt.show()

def plots():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    ax1.plot(loss_log)
    ax1.set_title("Loss")
    ax2.plot(lat_mag_log)
    ax2.set_title("Norm")
    ax3.plot(mom_log)
    ax3.set_title("Orientation")
    plt.show()

def means():
    loss = 0
    # num_classes = int(train_vecs.num_embeddings / 18)
    num_classes = int(unseen_vecs.num_embeddings / 20)
    for i in tqdm(range(num_classes)):
        zs = unseen_vecs(torch.arange(18*i, 18*(i+1)))
        z = torch.mean(zs, dim=0)
        im = decode_latent(decoder, z)

        _, _, (l2, l3), _ = get_normalization_params(im)   
        angle = abs(np.degrees(np.arctan2(l3, l2)/2))
        # plt.imshow(im)
        # plt.title(f"%.2f" % angle)
        # plt.show()
        loss += angle
    print(loss / num_classes)

def plot_inference():
    s = 3
    indices = np.arange(s*20, (s+1)*20, 2)
    # indices = np.arange(0, 100, 10)

    fig, axs = plt.subplots(2, len(indices), figsize=(2*len(indices), 2))
    for i, ax0, ax1 in zip(indices, axs[0].ravel(), axs[1].ravel()):
        sdf,  _, _ = test_seen[i]
        # sdf,  _, _ = test_unseen[i]
        im = samples_to_img(sdf)
        ax0.imshow(im, cmap="gray")

        z = seen_vecs(torch.tensor([i]))
        # z = unseen_vecs(torch.tensor([i]))
        im = decode_latent(decoder, z)
        ax1.imshow(im, cmap="gray")

        _, _, (l2, l3), _ = get_normalization_params(im)
        angle = (np.degrees(np.arctan2(l3, l2)/2))
        ax1.set_xlabel(r"$\theta$ = " + f"%.2f" % angle)        

        ax0.set_xticks([])
        ax0.set_yticks([])
        ax1.set_xticks([])
        ax1.set_yticks([])
    
    axs[0,0].set_ylabel("gt")
    axs[1,0].set_ylabel("inferred")

    plt.show()

def eval():
    print(f"Net{order}" + f", pose inferrence: {pose_inference}")

    if pose_inference:
        loss_seen = eval_recon(test_seen, seen_vecs)
        loss_unseen = eval_recon(test_unseen, unseen_vecs)
        print(f"seen l1: {loss_seen:.3f}")
        print(f"unseen l1: {loss_unseen:.3f}")

        norm_seen = eval_norm(seen_vecs)
        norm_unseen = eval_norm(unseen_vecs)
        print(f"seen norm: {norm_seen:.3f}")
        print(f"unseen norm: {norm_unseen:.3f}")

    else:
        loss_train = eval_recon(train, train_vecs)
        loss_seen = eval_recon(test_seen, seen_vecs)
        loss_unseen = eval_recon(test_unseen, unseen_vecs)

        print(f"train l1: {loss_train:.3f}")
        print(f"seen l1: {loss_seen:.3f}")
        print(f"unseen l1: {loss_unseen:.3f}")

        norm_train = eval_norm(train_vecs)
        norm_seen = eval_norm(seen_vecs)
        norm_unseen = eval_norm(unseen_vecs)
        norm_rand = eval_norm(rand_vecs)

        print(f"train norm: {norm_train:.3f}")
        print(f"seen norm: {norm_seen:.3f}")
        print(f"unseen norm: {norm_unseen:.3f}")
        print(f"rand norm: {norm_rand:.3f}")

def plot_recon_wrapper():
    indices = np.arange(train_vecs.num_embeddings)[::18]
    print(indices)
    plot_recon(train_vecs, 5, 9, indices)

def plot_inferred():
    for i in range(len(test_seen)):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        sdf, _, _ = test_seen[i]
        ax1.imshow(samples_to_img(sdf))
        
        z = seen_vecs(torch.tensor(i))
        img = decode_latent(decoder, z)
        _, _, (l2, l3), _ = get_normalization_params(img)
        ax2.imshow(img)
        ax2.set_title(np.degrees(np.arctan2(l3, l2)/2))
        plt.show()
    
    
# eval()
# plot_recon_wrapper()
# interpolate()
# plots()
# plot_inference()
means()
# plot_latent()
# plot_inferred()

# plt.imshow(decode_latent(decoder, seen_vecs(torch.tensor([0]))))
# plt.show()

    