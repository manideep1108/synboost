from image_dissimilarity.data.cityscapes_dataset import CityscapesDataset
from torch.utils.data import DataLoader
from image_dissimilarity.models.dissimilarity_model import DissimNet, DissimNetPrior
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import image_dissimilarity.configs.train.default_configuration.yaml as config_file
softmax = torch.nn.Softmax(dim=1)


#for now I have hard coded this should look over it once
with open(config_file, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

cfg_test_loader1 = config['test_dataloader1']   
dataset = cfg_test_loader1['dataset_args']
h = int((dataset['crop_size']/dataset['aspect_ratio']))
w = int(dataset['crop_size'])   #should figure this out



class SynboostDataModule(pl.LightningDataModule):
    def __init__(config):
        super().__init__()
    
        self.cfg= config

    def setup(self):
        # Assign train/val/test datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = CityscapesDataset(self.cfg["train_dataloader"]['dataset_args'])

        if stage == "val" or stage is None:
            self.validation_dataset = CityscapesDataset(self.cfg["val_dataloader"]['dataset_args'])
            self.test_dataset1 = CityscapesDataset(self.cfg["test_dataloader1"]['dataset_args'])
            self.test_dataset2 = CityscapesDataset(self.cfg["test_dataloader2"]['dataset_args'])
            self.test_dataset3 = CityscapesDataset(self.cfg["test_dataloader3"]['dataset_args'])
           # self.test_dataset4 = CityscapesDataset(self.cfg["test_dataloader4"]['dataset_args'])

        # if stage == "test" or stage is None:
        #     self.test_dataset1 = CityscapesDataset(self.cfg["test_dataloader1"]['dataset_args'])
        #     self.test_dataset2 = CityscapesDataset(self.cfg["test_dataloader2"]['dataset_args'])
        #     self.test_dataset3 = CityscapesDataset(self.cfg["test_dataloader3"]['dataset_args'])
        #     self.test_dataset4 = CityscapesDataset(self.cfg["test_dataloader4"]['dataset_args'])


    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.cfg["train_dataloader"]['dataloader_args'])
    
    def val_dataloader(self):
        return [ DataLoader(self.validation_dataset, self.cfg["val_dataloader"]['dataloader_args']),
            DataLoader(self.test_dataset1, self.cfg["test_dataloader1"]['dataloader_args']),
            DataLoader(self.test_dataset2, self.cfg["test_dataloader2"]['dataloader_args']),
            DataLoader(self.test_dataset3, self.cfg["test_dataloader3"]['dataloader_args']),
            #DataLoader(self.test_dataset4, self.cfg["test_dataloader4"]['dataloader_args'])
        ]

    # def test_dataloader(self):
    #     return [
    #         DataLoader(self.test_dataset1, self.cfg["test_dataloader1"]['dataloader_args']),
    #         DataLoader(self.test_dataset2, self.cfg["test_dataloader2"]['dataloader_args']),
    #         DataLoader(self.test_dataset3, self.cfg["test_dataloader3"]['dataloader_args']),
    #         DataLoader(self.test_dataset4, self.cfg["test_dataloader4"]['dataloader_args'])
    #     ]




class Synboost_trainer(pl.LightningModule):
    def __init__(self,config):
        super().__init__()

        self.cfg = config
        self.data_module = SynboostDataModule()
        self.test_loader1_size = len(self.datamodule.test_dataloader()[0])
        self.test_loader2_size = len(self.datamodule.test_dataloader()[1])
        self.test_loader3_size = len(self.datamodule.test_dataloader()[2])
        #self.test_loader4_size = len(self.datamodule.test_dataloader()[3])
        
        self.flat_pred = [np.zeros(h*w*self.test_loader1_size),np.zeros(h*w*self.test_loader2_size),np.zeros(h*w*self.test_loader3_size),np.zeros(h*w*self.test_loader4_size)]
        self.flat_labels = [np.zeros(h*w*self.test_loader1_size),np.zeros(h*w*self.test_loader2_size),np.zeros(h*w*self.test_loader3_size),np.zeros(h*w*self.test_loader4_size)]
        
        if cfg['model']['prior']:
            self.diss_model = DissimNetPrior(**self.cfg['model'])
        elif 'vgg' in self.cfg['model']['architecture']:
            self.diss_model = DissimNet(**self.cfg['model'])

        if self.cfg['training_strategy']['class_weight']:
            if not self.cfg['training_strategy']['class_weight_cityscapes']:
                if self.cfg['train_dataloader']['dataset_args']['void']:
                    label_path = os.path.join(self.cfg['train_dataloader']['dataset_args']['dataroot'], 'labels_with_void_no_ego/')
                else:
                    label_path = os.path.join(self.cfg['train_dataloader']['dataset_args']['dataroot'], 'labels/')
                    
                full_loader = trainer_util.loader(label_path, batch_size='all')
                print('Getting class weights for cross entropy loss. This might take some time.')
                class_weights = trainer_util.get_class_weights(full_loader, num_classes=2)
                #print("class weights are")
                #print(class_weights)
                torch.save(class_weights,"class_weights.pth")
            else:
                if self.cfg['train_dataloader']['dataset_args']['void']:
                    class_weights = [1.54843156, 8.03912212]
                else:
                    class_weights = [1.46494611, 16.5204619]
            print('Using the following weights for each respective class [0,1]:', class_weights)
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, weight=torch.FloatTensor(class_weights))
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255).cuda(self.gpu)


    def  training_step(self,batch,batch_idx):
        #iter_counter.record_one_iteration()
        original = batch['original']
        semantic = batch['semantic']
        synthesis = batch['synthesis']
        label = batch['label']
        
        # Training
        if prior:
            entropy = batch['entropy']
            mae = batch['mae']
            distance = batch['distance']
            predictions = self.diss_model(original, synthesis, semantic, entropy, mae, distance)
            loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1))
            
        else:
            predictions = self.diss_model(original, synthesis, semantic)
            loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1))
        
        self.log("train_iter_losss",loss)
        return loss
        # if opts.wandb:
        #     wandb.log({"Loss_iter_train": model_loss, "train_idx": idx_train})
        # iter+=1
        # idx_train +=1
            

    def training_epoch_end(self, training_step_outputs):
        print("Training Loss after epoch %f is : "% (self.trainer.current_epoch), training_step_outputs.mean())  #self.trainer.current_epoch
        self.log('avg_loss_train', training_step_outputs.mean())



    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        original = batch['original']
        semantic = batch['semantic']
        synthesis = batch['synthesis']   
        label = batch['label']
            
        if prior:
            entropy = batch['entropy']
            mae = batch['mae']
            distance = batch['distance']
    
            # Evaluating
            predictions = self.diss_model(original, synthesis, semantic, entropy, mae, distance)
            loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1))
        else:
            predictions = self.diss_model(original, synthesis, semantic)
            loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1))


        if(dataloader_idx== 1 or dataloader_idx== 2 or dataloader_idx== 3 ):
            outputs = softmax(outputs)
            (softmax_pred, predictions) = torch.max(outputs, dim=1)
            self.flat_pred[dataloader_idx][batch_idx * w * h:batch_idx * w * h + w * h] = torch.flatten(outputs[:, 1, :, :])
            self.flat_labels[dataloader_idx][batch_idx * w * h:batch_idx * w * h + w * h] = torch.flatten(label)

        return loss       


    def validation_epoch_end(self, validation_step_outputs, dataloader_idx=0):
        
        if  dataloader_idx==0:
            self.log('avg_loss_val', validation_step_outputs.mean())

        elif dataloader_idx== 1 or dataloader_idx== 2 or dataloader_idx== 3:
            results = metrics.get_metrics(self.flat_labels[idx], self.flat_pred[idx])
            log_dic = {"test_loss": validation_step_outputs.mean(), "mAP": results['AP'], "FPR@95TPR": results['FPR@95%TPR'], "AU_ROC": results['auroc']}
            self.flat_pred[dataloader_idx] = np.zeros(h*w*self.test_loader%f_size)%dataloader_idx
            self.flat_labels[dataloader_idx] = np.zeros(h*w*self.test_loader%f_size)%dataloader_idx
    

    def configure_optimizers(self):
        if self.cfg['optimizer']['algorithm'] == 'SGD':
                optimizer = torch.optim.SGD(self.diss_model.parameters(), lr=self.cfg['optimizer']['parameters']['lr'],
                                        weight_decay=self.cfg['optimizer']['parameters']['weight_decay'],)
        elif self.cfg['optimizer']['algorithm'] == 'Adam':
                optimizer = torch.optim.Adam(self.diss_model.parameters(),
                                        lr=self.cfg['optimizer']['parameters']['lr'],
                                        weight_decay=self.cfg['optimizer']['parameters']['weight_decay'],
                                        betas=(self.cfg['optimizer']['parameters']['beta1'], self.cfg['optimizer']['parameters']['beta2']))
        else:
                raise NotImplementedError

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, 'min', patience=self.cfg['optimizer']['parameters']['patience'], factor=self.cfg['optimizer']['parameters']['factor']),
                "monitor": self.val_loss ,     #should check if I should change the variable name
                "interval": "epoch",
                "frequency": 1  
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


