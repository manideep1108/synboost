import argparse
import yaml
import os
from tqdm import tqdm
import numpy as np
import shutil
from PIL import Image
import gc
import torch.backends.cudnn as cudnn
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage, ToTensor
import wandb

from trainers.dissimilarity_trainer import DissimilarityTrainer
from util import trainer_util
from util import trainer_util, metrics
from util.iter_counter import IterationCounter
from util.image_logging import ImgLogging
from util import visualization
from util import wandb_utils

from pytorch_lightning import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--seed', type=str, default='0', help='seed for experiment')
# parser.add_argument('--wandb_Api_key', type=str, default='None', help='Wandb_API_Key (Environment Variable)')
# parser.add_argument('--wandb_resume', type=bool, default=False, help='Resume Training')
# parser.add_argument('--wandb_run_id', type=str, default=None, help='Previous Run ID for Resuming')
# parser.add_argument('--wandb_run', type=str, default=None, help='Name of wandb run')
# parser.add_argument('--wandb_project', type=str, default="MLRC_Synboost", help='wandb project name')
# parser.add_argument('--wandb', type=bool, default=True, help='Log to wandb')
# parser.add_argument('--pre_epoch', type=int, default=0, help='Previous epoch Number to resume training')
# parser.add_argument('--epochs', type=int, default=16, help='No. of epochs to run ')
# parser.add_argument('--name', type=str, default='latest', help='file Name of the resuming run')
opts = parser.parse_args()
cudnn.benchmark = True

# Load experiment setting
with open(opts.config, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)


exp_name = config['experiment_name'] + opts.seed

print('Starting experiment named: %s'%exp_name)

config['gpu_ids'] = opts.gpu_ids
gpu_info = trainer_util.activate_gpus(config)

print('Starting Training...')
best_val_loss = float('inf')
best_map_metric = 0
iter = 0


from trainers.dissimilarity_trainer_lightning import SynboostDataModule,Synboost_trainer


datamodule = SynboostDataModule(config)
model = Synboost_trainer(config)

trainer = Trainer(max_epochs=2, gpus=config['gpu_ids'])
trainer.fit(model, datamodule=datamodule)







