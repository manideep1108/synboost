import torch
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

import sys
sys.path.append("..")
from image_dissimilarity.util import trainer_util
from image_dissimilarity.models.dissimilarity_model import DissimNet, DissimNetPrior

class DissimilarityTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, config, seed=0, resume=False, epoch=None):
        
        trainer_util.set_seed(seed)
        
        cudnn.enabled = True
        self.config = config
        
        if config['gpu_ids'] != -1:
            self.gpu = 'cuda'
        else:
            self.gpu = 'cpu'
        
        if config['model']['prior']:
            self.diss_model = DissimNetPrior(**config['model']).cuda(self.gpu)
        elif 'vgg' in config['model']['architecture']:
            self.diss_model = DissimNet(**config['model']).cuda(self.gpu)
        else:
            raise NotImplementedError()

        # get pre-trained model
        if config["wandb_config"]["wandb"] and resume:
            wandb_load_file_path = "checkpoints/Epoch_" + str(opts.pre_epoch) + "pth"
            wandb.restore(file_path)
            checkpoint = torch.load(checkpoint_fpath)
            full_model_path = config["wandb_config"]["model_path_base"] + wandb_load_file_path

            print('Loading pretrained weights from %s (epoch: %s)' % (full_model_path, epoch))
            model_weights = torch.load(full_model_path)
            self.diss_model.load_state_dict(model_weights, strict=False)
            # NOTE: For old models, there were some correlation weights created that were not used in the foward pass. That's the reason to include strict=False
            
        print('Printing Model Parameters')
        print(self.diss_model.parameters)
        
        lr_config = config['optimizer']
        lr_options = lr_config['parameters']
        if lr_config['algorithm'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.diss_model.parameters(), lr=lr_options['lr'],
                                             weight_decay=lr_options['weight_decay'],)
        elif lr_config['algorithm'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.diss_model.parameters(),
                                              lr=lr_options['lr'],
                                              weight_decay=lr_options['weight_decay'],
                                              betas=(lr_options['beta1'], lr_options['beta2']))
        else:
            raise NotImplementedError
        
        if lr_options['lr_policy'] == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=lr_options['patience'], factor=lr_options['factor'])
        else:
            raise NotImplementedError
        
        self.old_lr = lr_options['lr']
        
        if config['training_strategy']['class_weight']:
            if not config['training_strategy']['class_weight_cityscapes']:
                if config['train_dataloader']['dataset_args']['void']:
                    label_path = os.path.join(config['train_dataloader']['dataset_args']['dataroot'], 'labels_with_void_no_ego/')
                else:
                    label_path = os.path.join(config['train_dataloader']['dataset_args']['dataroot'], 'labels/')
                    
                full_loader = trainer_util.loader(label_path, batch_size='all')
                print('Getting class weights for cross entropy loss. This might take some time.')
                class_weights = trainer_util.get_class_weights(full_loader, num_classes=2)
                #print("class weights are")
                #print(class_weights)
                torch.save(class_weights,"class_weights.pth")
            else:
                if config['train_dataloader']['dataset_args']['void']:
                    class_weights = [1.54843156, 8.03912212]
                else:
                    class_weights = [1.46494611, 16.5204619]
            print('Using the following weights for each respective class [0,1]:', class_weights)
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, weight=torch.FloatTensor(class_weights).to("cuda")).cuda(self.gpu)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255).cuda(self.gpu)
        
    def run_model_one_step(self, original, synthesis, semantic, label):
        self.optimizer.zero_grad()
        predictions = self.diss_model(original, synthesis, semantic)
        model_loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1).cuda())
        model_loss.backward()
        self.optimizer.step()
        self.model_losses = model_loss
        self.generated = predictions
        return model_loss.item(), predictions
        
    def run_validation(self, original, synthesis, semantic, label):
        predictions = self.diss_model(original, synthesis, semantic)
        model_loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1).cuda())
        return model_loss.item(), predictions

    def run_model_one_step_prior(self, original, synthesis, semantic, label, entropy, mae, distance):
        self.optimizer.zero_grad()
        predictions = self.diss_model(original, synthesis, semantic, entropy, mae, distance)
        model_loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1).cuda())
        model_loss.backward()
        self.optimizer.step()
        self.model_losses = model_loss
        self.generated = predictions
        return model_loss.item(), predictions

    def run_validation_prior(self, original, synthesis, semantic, label, entropy, mae, distance):
        predictions = self.diss_model(original, synthesis, semantic, entropy, mae, distance)
        model_loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1).cuda())
        return model_loss.item(), predictions

    def get_latest_losses(self):
        return {**self.model_loss}

    def get_latest_generated(self):
        return self.generated

    def save(self, save_dir, name, epoch):

        checkpoint = {
            'epoch': epoch,
            'state_dict': self.diss_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'criterion': self.criterion.state_dict()
        }

        if not os.path.isdir(os.path.join(save_dir, name)):
            os.mkdir(os.path.join(save_dir, name))
        
        save_filename = '%s_net_%s.pth' % (epoch, name)
        save_path = os.path.join(save_dir, name, save_filename)
        torch.save(checkpoint, save_path)  # net.cpu() -> net

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.config['training_strategy']['niter']:
            lrd = self.config['optimizer']['parameters']['lr'] / self.config['training_strategy']['niter_decay']
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
            
    def update_learning_rate_schedule(self, val_loss):
        self.scheduler.step(val_loss)
        lr = [group['lr'] for group in self.optimizer.param_groups][0]
        print('Current learning rate is set for %f' %lr)

if __name__ == "__main__":
    import yaml
    
    config = '../configs/default_configuration.yaml'
    gpu_ids = 0

    with open(config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    config['gpu_ids'] = gpu_ids
    trainer = DissimilarityTrainer(config)
