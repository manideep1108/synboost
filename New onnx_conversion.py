import argparse
import yaml
import torch.backends.cudnn as cudnn
import torch
from PIL import Image
import numpy as np
import os
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
from itertools import product
from numpy.linalg import norm
from image_dissimilarity.util.load import load_ckp
from image_dissimilarity.util import wandb_utils
from image_dissimilarity.util.load import load_ckp
import cv2
import torchvision 
import wandb

from image_dissimilarity.util import trainer_util, metrics
from image_dissimilarity.util.iter_counter import IterationCounter
from image_dissimilarity.models.dissimilarity_model import DissimNet, DissimNetPrior

import os
from PIL import Image
import numpy as np
import cv2
from collections import OrderedDict
import shutil
import torch
from torch.backends import cudnn
import torch.onnx
import torch.nn as nn
import torchvision.transforms as transforms
import onnx
import onnxruntime

from options.test_options import TestOptions

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
#parser.add_argument('--weights', type=str, default='[0.70, 0.1, 0.1, 0.1]', help='weights for ensemble testing [model, entropy, mae, distance]')
parser.add_argument('--wandb_Api_key', type=str, default='None', help='Wandb_API_Key (Environment Variable)')
parser.add_argument('--wandb_resume', type=bool, default=False, help='Resume Training')
parser.add_argument('--wandb_run_id', type=str, default=None, help='Previous Run ID for Resuming')
parser.add_argument('--wandb_run', type=str, default=None, help='Name of wandb run')
parser.add_argument('--wandb_project', type=str, default="MLRC_Synboost", help='wandb project name')
parser.add_argument('--wandb', type=bool, default=True, help='Log to wandb')
parser.add_argument('--epoch', type=int, default=12, help='best epoch number in wandb')

opts = parser.parse_args()

def remove_all_spectral_norm(item):
    if isinstance(item, nn.Module):
        try:
            nn.utils.remove_spectral_norm(item)
        except Exception:
            pass
        
        for child in item.children():
            remove_all_spectral_norm(child)
    
    if isinstance(item, nn.ModuleList):
        for module in item:
            remove_all_spectral_norm(module)
    
    if isinstance(item, nn.Sequential):
        modules = item.children()
        for module in modules:
            remove_all_spectral_norm(module)

def convert_dissimilarity_model(
    config='./image_dissimilarity/configs/test/road_anomaly_configuration.yaml',
    model_name='dissimilarity.onnx'):
    import sys
    sys.path.insert(0, './image_dissimilarity')
    from image_dissimilarity.models.dissimilarity_model import DissimNetPrior
    from image_dissimilarity.util import trainer_util
    import yaml
    

    with open(opts.configs, 'r') as stream:
        configs = yaml.load(stream, Loader=yaml.FullLoader)

    # get experiment information
    exp_name = configs['experiment_name']
    save_fdr = configs['save_folder']
    epoch = configs['which_epoch']
    store_fdr = configs['store_results']
    store_fdr_exp = os.path.join(configs['store_results'], exp_name)



    # Load experiment setting
    with open(config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    # get experiment information
    # exp_name = config['experiment_name']
    # save_fdr = config['save_folder']
    # epoch = config['which_epoch']


    # checks if we are using prior images
    prior = config['model']['prior']
    # Get data loaders
    cfg_test_loader = config['test_dataloader']
    # adds logic to dataloaders (avoid repetition in config file)
    cfg_test_loader['dataset_args']['prior'] = prior

    dataloader = trainer_util.get_dataloader(cfg_test_loader['dataset_args'], cfg_test_loader['dataloader_args'])
    
    # get model
    
    diss_model = DissimNetPrior(**config['model']).cuda()
    
    # diss_model.eval()
    # model_path = os.path.join('image_dissimilarity', save_fdr, exp_name, '%s_net_%s.pth' % (epoch, exp_name))
    # model_weights = torch.load(model_path)
    # diss_model.load_state_dict(model_weights)

            # get model
    if configs['model']['prior']:
        diss_model = DissimNetPrior(**configs['model']).cuda()
    elif 'vgg' in configs['model']['architecture']:
        diss_model = DissimNet(**configs['model']).cuda()
    else:
        raise NotImplementedError()
    
    diss_model.eval()


    use_wandb = opts.wandb
    wandb_resume = opts.wandb_resume
    wandb_utils.init_wandb(config=configs, key=opts.wandb_Api_key,wandb_project= opts.wandb_project, wandb_run=opts.wandb_run, wandb_run_id=opts.wandb_run_id, wandb_resume=opts.wandb_resume)
    diss_model.eval()
    if use_wandb and wandb_resume:
        checkpoint = load_ckp(configs["wandb_config"]["model_path_base"], "best", 12)
        diss_model.load_state_dict(checkpoint['state_dict'], strict=False)

    remove_all_spectral_norm(diss_model)
    # Input to the model

    for i, data_i in enumerate(dataloader):
        original = data_i['original'].cuda()
        semantic = data_i['semantic'].cuda()
        synthesis = data_i['synthesis'].cuda()
        entropy = data_i['entropy'].cuda()
        mae = data_i['mae'].cuda()
        distance = data_i['distance'].cuda()
        
        torch_out = diss_model(original, synthesis, semantic, entropy, mae, distance)
        break
    
    # Export the model
    torch.onnx.export(diss_model,  # model being run
                      (original, synthesis, semantic, entropy, mae, distance),  # model input (or a tuple for multiple inputs)
                      model_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})
    
    ort_session = onnxruntime.InferenceSession(model_name)
    
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    input_feeds = {}
    input_feeds[ort_session.get_inputs()[0].name] = to_numpy(original)
    input_feeds[ort_session.get_inputs()[1].name] = to_numpy(synthesis)
    input_feeds[ort_session.get_inputs()[2].name] = to_numpy(semantic)
    input_feeds[ort_session.get_inputs()[3].name] = to_numpy(entropy)
    input_feeds[ort_session.get_inputs()[4].name] = to_numpy(mae)
    input_feeds[ort_session.get_inputs()[5].name] = to_numpy(distance)

    ort_outs = ort_session.run(None, input_feeds)
    
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-03)
    
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")





if __name__ == '__main__':
    # Load experiment setting

    convert_dissimilarity_model('./image_dissimilarity/configs/test/road_anomaly_configuration.yaml')