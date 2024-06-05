import os

# In order to reproduce the results presented in the thesis:

# DO CHANGE THESE ----------------------------------------
source_dir = "../mpeg7"
order = 2
pose_inference = False


# DONT CHANGE THESE --------------------------------------
data_dir = "data"
normalization_param_subdir = "NormalizationParameters"
sdf_samples_subdir = "SdfSamples"
model_dir = f"model_mom{order}"
model_params_subdir = "ModelParameters"
model_logs_subdir = "Logs"
reconstructions_subdir = "Reconstructions"

source_name = os.path.split(source_dir)[-1]

sidelen = 128
num_surface_samples = 10000
num_samp_per_scene = sidelen**2 + num_surface_samples

latent_size = 128
hidden_dim = 256
batch_size = 16
num_epochs = 250

num_recon_iters = 250
num_pose_inference_iters = 1000

lr = 1e-3
lra = 1e-3
lrz = 1e-3

sigma = 0.01
code_reg_lambda = sigma**2

load_ram = True