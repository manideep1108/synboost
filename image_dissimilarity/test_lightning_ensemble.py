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
from util.load import load_ckp
from util import wandb_utils
from util.load import load_ckp

from util import trainer_util, metrics
from util.iter_counter import IterationCounter
from models.dissimilarity_model import DissimNet, DissimNetPrior

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
#parser.add_argument('--weights', type=str, default='[0.70, 0.1, 0.1, 0.1]', help='weights for ensemble testing [model, entropy, mae, distance]')

opts = parser.parse_args()
cudnn.benchmark = True
#weights = ast.literal_eval(opts.weights)

def normalize(weights):
    # calculate l1 vector norm
    result = norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result


# grid search weights
def grid_search(model_num=4):
    # define weights to consider
    d = {}
    w = [0, 1, 2, 3]
    best_score, best_roc, best_ap, best_weights = 1.0, 0, 0, None
    best = -1
    # iterate all possible combinations (cartesian product)
    for weights in product(w, repeat=model_num):
        # skip if all weights are equal
        if len(set(weights)) == 1:
            continue
        # hack, normalize weight vector
        weights = normalize(weights)
        if str(weights) in d:
            continue
        else:
            d[str(weights)] = 0
        # evaluate weights
        score_roc, score_ap, score_fp = evaluate_ensemble(weights)
        print('Weights: %s Score_FP: %.3f Score_ROC:%.3f Score_AP:%.3f' % (weights, score_fp, score_roc, score_ap))
        if score_ap - score_fp > best:
            best_score, best_weights, best_roc, best_ap = score_fp, weights, score_roc, score_ap
            best = score_ap - score_fp
            print('>BEST SO FAR %s Score_FP: %.3f Score_ROC:%.3f Score_AP:%.3f' % (best_weights, best_score, best_roc, best_ap))
    return list(best_weights), best_score, best_roc, best_ap

def evaluate_ensemble(weights_f):
    # create memory locations for results to save time while running the code
    dataset = cfg_test_loader['dataset_args']
    h = int((dataset['crop_size']/dataset['aspect_ratio']))
    w = int(dataset['crop_size'])
    flat_pred = np.zeros(w*h*len(test_loader), dtype='float32')
    flat_labels = np.zeros(w*h*len(test_loader), dtype='float32')
    
    with torch.no_grad():
        for i, data_i in enumerate(test_loader):
            original = data_i['original'].cuda()
            semantic = data_i['semantic'].cuda()
            synthesis = data_i['synthesis'].cuda()
            label = data_i['label'].cuda()
        
            if prior:
                entropy = data_i['entropy'].cuda()
                mae = data_i['mae'].cuda()
                distance = data_i['distance'].cuda()
                outputs = softmax(diss_model(original, synthesis, semantic, entropy, mae, distance))
                
            else:
                outputs = softmax(diss_model(original, synthesis, semantic))
            (softmax_pred, predictions) = torch.max(outputs,dim=1)
            
            soft_pred = outputs[:,1,:,:]*weights_f[0] + entropy*weights_f[1] + mae*weights_f[2] + distance*weights_f[3]
            flat_pred[i*w*h:i*w*h+w*h] = torch.flatten(soft_pred).detach().cpu().numpy()
            flat_labels[i*w*h:i*w*h+w*h] = torch.flatten(label).detach().cpu().numpy()
            # Save results
            predicted_tensor = predictions * 1
            label_tensor = label * 1
            
            file_name = os.path.basename(data_i['original_path'][0])
            label_img = Image.fromarray(label_tensor.squeeze().cpu().numpy().astype(np.uint8))
            soft_img = Image.fromarray((soft_pred.squeeze().cpu().numpy()*255).astype(np.uint8))
            predicted_img = Image.fromarray(predicted_tensor.squeeze().cpu().numpy().astype(np.uint8))
            predicted_img.save(os.path.join(store_fdr_exp, 'pred', file_name))
            soft_img.save(os.path.join(store_fdr_exp, 'soft', file_name))
            label_img.save(os.path.join(store_fdr_exp, 'label', file_name))
    
    if config['test_dataloader']['dataset_args']['roi']:
        invalid_indices = np.argwhere(flat_labels == 255)
        flat_labels = np.delete(flat_labels, invalid_indices)
        flat_pred = np.delete(flat_pred, invalid_indices)
    
    results = metrics.get_metrics(flat_labels, flat_pred)
    return results['auroc'], results['AP'], results['FPR@95%TPR']

if __name__ == '__main__':
    # Load experiment setting
    with open(opts.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    #get wandb information
    wandb_Api_key = config["wandb_config"]['wandb_Api_key']
    wandb_resume = config["wandb_config"]['wandb_resume']
    wandb_run_id = config["wandb_config"]['wandb_run_id']
    wandb_run = config["wandb_config"]['wandb_run']
    wandb_project = config["wandb_config"]['wandb_project']
    epoch = config["wandb_config"]['best_epoch']

    #Logs into wandb with given api key
    os.environ["WANDB_API_KEY"] = wandb_Api_key
    
    # get experiment information
    exp_name = config['experiment_name']
    save_fdr = config['save_folder']
    store_fdr = config['store_results']
    store_fdr_exp = os.path.join(config['store_results'], exp_name)
    
    if not os.path.isdir(store_fdr):
        os.mkdir(store_fdr)
    
    if not os.path.isdir(store_fdr_exp):
        os.mkdir(store_fdr_exp)
    
    if not os.path.isdir(os.path.join(store_fdr_exp, 'pred')):
        os.mkdir(os.path.join(store_fdr_exp, 'label'))
        os.mkdir(os.path.join(store_fdr_exp, 'pred'))
        os.mkdir(os.path.join(store_fdr_exp, 'soft'))
    
    # Activate GPUs
    config['gpu_ids'] = opts.gpu_ids
    gpu_info = trainer_util.activate_gpus(config)
    
    # checks if we are using prior images
    prior = config['model']['prior']
    # Get data loaders
    cfg_test_loader = config['test_dataloader']
    # adds logic to dataloaders (avoid repetition in config file)
    cfg_test_loader['dataset_args']['prior'] = prior
    test_loader = trainer_util.get_dataloader(cfg_test_loader['dataset_args'], cfg_test_loader['dataloader_args'])
    
    # Added functionality to access vgg16, resnet18, resnet101 encoders
    if 'vgg' in config['model']['architecture']:
        if config['model']['prior']:
            diss_model = DissimNetPrior(**config['model']).cuda()
        else:
            diss_model = DissimNet(**config['model']).cuda()

    else:
        raise NotImplementedError()
    
    wandb_resume = wandb_resume
    wandb_utils.init_wandb(config=config, key=wandb_Api_key,wandb_project= wandb_project, wandb_run=wandb_run, wandb_run_id=wandb_run_id, wandb_resume=wandb_resume)
    diss_model.eval()

    if wandb_resume:
        run = wandb.init()  
        artifact = run.use_artifact(opts.artifact_path, type='model')
        artifact_dir = artifact.download()  #should change these lines so that user can specify path (now just for testing)
        model =Synboost_trainer.load_from_checkpoint(Path(artifact_dir)/'model.ckpt', config=config )
        resume_path = "artifacts/" + opts.artifact_path + "/model.ckpt"


    trainer = Trainer(max_epochs=opts.max_epoch, gpus=1, log_every_n_steps=1, logger=wandb_logger,  callbacks=[checkpoint_callback, lr_monitor],resume_from_checkpoint=resume_path)
    trainer.fit(model, datamodule=datamodule)                                                                                                            
    print("Calling finish")
    wandb.finish()
    if wandb_resume:
        checkpoint = load_ckp(config["wandb_config"]["model_path_base"], "best", epoch)
        diss_model.load_state_dict(checkpoint['state_dict'])
    
    softmax = torch.nn.Softmax(dim=1)
    best_weights, best_score, best_roc, best_ap = grid_search()
    print('Best weights: %s Score_FP: %.3f Score_ROC:%.3f Score_AP:%.3f' % (best_weights, best_score, best_roc, best_ap))