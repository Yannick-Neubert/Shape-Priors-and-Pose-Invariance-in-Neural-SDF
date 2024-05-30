import torch
from torch.utils.data import Dataset
from utils import *
import os

class SDFDataset(Dataset):
    def __init__(self, source, split, order, load_ram=True):
        self.source = source
        self.load_ram = load_ram
        self.order = "moments2" if order == 2 else "moments3"

        files = os.listdir(os.path.join(source, sdf_samples_subdir, source_name))
        files.sort()

        test_classes = -5 * 20
        match split:
            case "train": self.npyfiles = list(np.concatenate([files[x+2:x+20] for x in range(0, len(files[:test_classes]), 20)]))
            case "test_seen": self.npyfiles = list(np.concatenate([files[x:x+2] for x in range(0, len(files[:test_classes]), 20)]))
            case "test_unseen": self.npyfiles = files[test_classes:]
        
        self.npyfiles.sort()

        if self.load_ram:
            self.loaded_data = []
            for file in self.npyfiles:
                file = os.path.splitext(file)[0]
                sdf = np.load(os.path.join(self.source, sdf_samples_subdir, source_name, file) + ".npy")
                moments = np.load(os.path.join(self.source, normalization_param_subdir, source_name, file) + ".npz")[self.order]
                
                sdf = torch.from_numpy(sdf)[torch.randperm(sdf.shape[0])].float()
                moments = torch.from_numpy(moments).float()

                self.loaded_data.append([sdf, moments])

    def __len__(self):
        return len(self.npyfiles)
    
    def __getitem__(self, index):
        if self.load_ram:
            datum = self.loaded_data[index]
            return datum[0], datum[1], index
        else:
            file = os.path.splitext(self.npyfiles[index])[0]
            sdf = np.load(os.path.join(self.source, sdf_samples_subdir, source_name, file) + ".npy")
            moments = np.load(os.path.join(self.source, normalization_param_subdir, source_name, file) + ".npz")[self.order]

            sdf = torch.from_numpy(sdf)[torch.randperm(sdf.shape[0])].float()
            moments = torch.from_numpy(moments).float()

        return sdf, moments, index