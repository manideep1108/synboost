import shutil
import torch
import wandb
import os



def load_ckp(wandb_base_path, name, epoch):      

    # load check point
    path = name + "/" + name + "_" + str(epoch) + ".pth"
    load_path = os.path.join(wandb_base_path, path)
    wandb.restore(path)
    checkpoint = torch.load(load_path)

    return checkpoint
