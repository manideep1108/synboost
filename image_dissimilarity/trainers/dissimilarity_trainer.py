import torch
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from util.load import load_ckp

import sys
sys.path.append("..")
from image_dissimilarity.util import trainer_util
from image_dissimilarity.models.dissimilarity_model import DissimNet, DissimNetPrior, ResNet18DissimNet, ResNet18DissimNetPrior, ResNet101DissimNetPrior

class DissimilarityTrainer:
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """
    def __init__(self, config, wandb=True, resume=False, epoch=0, name="latest",seed=0):
        
        trainer_util.set_seed(seed)
        
        cudnn.enabled = True
        self.config = config
        self.wandb = wandb
        self.resume = resume
        
        if config['gpu_ids'] != -1:
            self.gpu = 'cuda'
        else:
            self.gpu = 'cpu'
        
        # Added functionality to access vgg16, resnet18, resnet101 encoders
         if 'vgg' in config['model']['architecture']:
            if config['model']['prior']:
                self.diss_model = DissimNetPrior(**config['model']).cuda(self.gpu)
            else:
                self.diss_model = DissimNet(**config['model']).cuda(self.gpu)
                
        elif 'resnet18' in config['model']['architecture']:
            if config['model']['prior']:
                self.diss_model = ResNet18DissimNetPrior(**config['model']).cuda(self.gpu)
            else:
                self.diss_model = ResNet18DissimNet(**config['model']).cuda(self.gpu)
                
        elif 'resnet101' in config['model']['architecture'] and config['model']['prior']:
            self.diss_model = ResNet101DissimNetPrior(**config['model']).cuda(self.gpu)
        else:
            raise NotImplementedError()

        
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


       # get pre-trained model
        if self.wandb and self.resume:
            self.checkpoint = load_ckp(config["wandb_config"]["model_path_base"], name, epoch)
            self.diss_model.load_state_dict(self.checkpoint['state_dict'], strict=False)
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.scheduler.load_state_dict(self.checkpoint['scheduler'])
            self.criterion.load_state_dict(self.checkpoint['criterion'])
            
            # NOTE: For old models, there were some correlation weights created that were not used in the foward pass. That's the reason to include strict=False
            
        print('Printing Model Parameters')
        print(self.diss_model.parameters)

    def return_iter(self):
        if self.wandb and self.resume:
             return self.checkpoint["idx_train"]
        else:
            return 0
        

        
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

    def save(self, save_dir, base_dir, name, epoch, wandb_bool, idx_train=None):

        checkpoint = {
            'epoch': epoch,
            'state_dict': self.diss_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'criterion': self.criterion.state_dict(),
            'idx_train': idx_train,

        }

        if not os.path.isdir(os.path.join(save_dir, name)):
            os.mkdir(os.path.join(save_dir, name))
        
        save_filename = '%s_%s.pth' % (name, epoch)
        save_path = os.path.join(save_dir, name, save_filename)
        torch.save(checkpoint, save_path)  # net.cpu() -> net
        if wandb_bool:
          wandb.save(save_path, base_path = base_dir, policy = 'live')

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
