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
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
from pytorch_lightning.callbacks import LearningRateMonitor

#from trainers.dissimilarity_trainer import DissimilarityTrainer
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
#parser.add_argument('--wandb_resume', type=bool, default = False, help='Resume Training')
#parser.add_argument('--artifact_path', type=str, default= 's', help='Path of artifact to load weights and Resume Run')
#parser.add_argument('--wandb_run_id', type=str, default=None, help='Previous Run ID for Resuming')
#parser.add_argument('--wandb_run', type=str, default=None, help='Name of wandb run')
#parser.add_argument('--max_epoch', type=int, default=13, help="Number of epochs u want to run")
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

#get wandb information
wandb_Api_key = config["wandb_config"]['wandb_Api_key']
wandb_resume = config["wandb_config"]['wandb_resume']
wandb_run_id = config["wandb_config"]['wandb_run_id']
wandb_run = config["wandb_config"]['wandb_run']
max_epoch = config["wandb_config"]['max_epoch']
artifact_path = config["wandb_config"]['artifact_path']

#logs into wandb with given api key
os.environ["WANDB_API_KEY"] = wandb_Api_key

exp_name = config['experiment_name'] + opts.seed

print('Starting experiment named: %s'%exp_name)

config['gpu_ids'] = opts.gpu_ids
gpu_info = trainer_util.activate_gpus(config)

print('Starting Training...')
best_val_loss = float('inf')
best_map_metric = 0
iter = 0

print(wandb_resume)

if wandb_resume:
    wandb_logger = WandbLogger(project='MLRC_Synboost',name = wandb_run ,log_model='all', resume='allow', id=wandb_run_id) # log all new checkpoints during training

else:
    wandb_logger = WandbLogger(project='MLRC_Synboost', log_model='all',name = wandb_run, resume=None) # log all new checkpoints during training


checkpoint_callback = ModelCheckpoint(
    monitor='val_loss/dataloader_idx_0',
    save_top_k =2,                 
    mode='min',                    
    every_n_epochs=1,             
    save_on_train_epoch_end=True,
    save_last = True    
    )

lr_monitor = LearningRateMonitor(logging_interval='epoch')



from trainers.dissimilarity_trainer_lightning import SynboostDataModule,Synboost_trainer


datamodule = SynboostDataModule(config)
model = Synboost_trainer(config)
wandb_logger.watch(model,log='all')  # logs histogram of gradients and parameters

print(wandb_resume)
resume_path = None

if wandb_resume:
    run = wandb.init()  
    artifact = run.use_artifact(artifact_path, type='model')
    artifact_dir = artifact.download()  #should change these lines so that user can specify path (now just for testing)
    model =Synboost_trainer.load_from_checkpoint(Path(artifact_dir)/'model.ckpt', config=config )
    resume_path = "artifacts/" + artifact_path + "/model.ckpt"


trainer = Trainer(max_epochs=max_epoch, gpus=1, log_every_n_steps=1, logger=wandb_logger,  callbacks=[checkpoint_callback, lr_monitor],resume_from_checkpoint=resume_path)
trainer.fit(model, datamodule=datamodule)                                                                                                            
print("Calling finish")
wandb.finish()






