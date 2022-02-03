import os
from PIL import Image
import numpy as np
import cv2
from collections import OrderedDict
import shutil
import torch
import random
from natsort import natsorted
from torch.backends import cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

from options.test_options import TestOptions
import sys
sys.path.insert(0, './image_segmentation')
import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

TestOptions = TestOptions()
opt = TestOptions.parse()

if not opt.no_segmentation:
    assert_and_infer_cfg(opt, train_mode=False)
    cudnn.benchmark = False
    torch.cuda.empty_cache()

    # Get segmentation Net
    opt.dataset_cls = cityscapes
    net = network.get_net(opt, criterion=None)
    net = torch.nn.DataParallel(net).cuda()
    print('Segmentation Net built.')
    net, _ = restore_snapshot(net, optimizer=None, snapshot=opt.snapshot, restore_optimizer_bool=False)
    net.eval()
    print('Segmentation Net Restored.')

    # Get RGB Original Images
    data_dir = opt.demo_folder
    images = os.listdir(data_dir)
    if len(images) == 0:
        print('There are no images at directory %s. Check the data path.' % (data_dir))
    else:
        print('There are %d images to be processed.' % (len(images)))
    images.sort()

    # Transform images to Tensor based on ImageNet Mean and STD
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

    # Create save directory
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)

    soft_fdr = os.path.join(opt.results_dir, 'entropy')
    soft_fdr_2 = os.path.join(opt.results_dir, 'logit_distance')
    semantic_fdr = os.path.join(opt.results_dir, 'semantic')

    if not os.path.exists(soft_fdr):
        os.makedirs(soft_fdr)

    if not os.path.exists(soft_fdr_2):
        os.makedirs(soft_fdr_2)

    if not os.path.exists(semantic_fdr):
        os.makedirs(semantic_fdr)

    # creates temporary folder to adapt format to image synthesis
    if not os.path.exists(os.path.join(opt.results_dir, 'temp')):
        os.makedirs(os.path.join(opt.results_dir, 'temp'))
        os.makedirs(os.path.join(opt.results_dir, 'temp', 'gtFine', 'val'))
        os.makedirs(os.path.join(opt.results_dir, 'temp', 'leftImg8bit', 'val'))

    softmax = torch.nn.Softmax(dim=1)

    # Loop around all figures
    for img_id, img_name in enumerate(images):
        img_dir = os.path.join(data_dir, img_name)
        img = Image.open(img_dir).convert('RGB')
        img.save(os.path.join(opt.results_dir, 'temp', 'leftImg8bit', 'val', img_name[:-4] + '_leftImg8bit.png'))
        img_tensor = img_transform(img)

        # predict
        with torch.no_grad():
            pred = net(img_tensor.unsqueeze(0).cuda())
            print('%04d/%04d: Segmentation Inference done.' % (img_id + 1, len(images)))

            outputs = softmax(pred)

            # get entropy
            softmax_pred = torch.sum(-outputs*torch.log(outputs), dim=1)
            softmax_pred = (softmax_pred - softmax_pred.min()) / softmax_pred.max()

            # get logit distance
            distance, _ = torch.topk(outputs, 2, dim=1)
            max_logit = distance[:, 0, :, :]
            max2nd_logit = distance[:, 1, :, :]
            result = max_logit - max2nd_logit
            map_logit = 1 - (result - result.min()) / result.max()

        pred_og = pred.cpu().numpy().squeeze()
        softmax_pred_og = softmax_pred.cpu().numpy().squeeze()
        map_logit = map_logit.cpu().numpy().squeeze()
        pred = np.argmax(pred_og, axis=0)

        softmax_pred_og = softmax_pred_og* 255
        map_logit = map_logit * 255
        pred_name = 'entropy_' + img_name
        pred_name_2 = 'distance_' + img_name
        pred_name_3 = 'pred_mask_' + img_name
        cv2.imwrite(os.path.join(soft_fdr, pred_name), softmax_pred_og)
        cv2.imwrite(os.path.join(soft_fdr_2, pred_name_2), map_logit)
        cv2.imwrite(os.path.join(semantic_fdr, pred_name_3), pred)

        # save label-based predictions, e.g. for submission purpose
        label_out = np.zeros_like(pred)
        for label_id, train_id in opt.dataset_cls.id_to_trainid.items():
            label_out[np.where(pred == train_id)] = label_id
        #cv2.imwrite(os.path.join(semantic_label_fdr, pred_name), label_out)
        cv2.imwrite(os.path.join(opt.results_dir, 'temp', 'gtFine', 'val', pred_name[:-4] + '_instanceIds.png'), label_out)
        cv2.imwrite(os.path.join(opt.results_dir, 'temp', 'gtFine', 'val', pred_name[:-4] + '_labelIds.png'), label_out)

    print('Segmentation Results saved.')

print('Starting Image Synthesis Process')

import sys
sys.path.insert(0, './image_synthesis')
import data
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html

world_size = 1
rank = 0

# Corrects where dataset is in necesary format
opt.dataroot = os.path.join(opt.results_dir, 'temp')

opt.world_size = world_size
opt.gpu = 0
opt.mpdist = False

dataloader = data.create_dataloader(opt, world_size, rank)


model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt, rank)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break
    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1], gray=True)

webpage.save()

synthesis_fdr = os.path.join(opt.results_dir, 'synthesis')

# Cleans output folders and deletes temporary files
if not os.path.exists(synthesis_fdr):
    os.makedirs(synthesis_fdr)

source_fdr = os.path.join(opt.results_dir, 'image-synthesis/test_latest/images/synthesized_image')
for image_name in os.listdir(source_fdr):
    shutil.move(os.path.join(source_fdr,image_name), os.path.join(synthesis_fdr,image_name))
    
shutil.rmtree(os.path.join(opt.results_dir, 'image-synthesis'))
shutil.rmtree(os.path.join(opt.results_dir, 'temp'))

data_origin = 'spade'
soft_fdr = os.path.join(opt.results_dir, 'mae_features_' + data_origin)
    
if not os.path.exists(soft_fdr):
    os.makedirs(soft_fdr)
    
original_paths = [os.path.join(data_dir, image)
                       for image in os.listdir(os.path.join(data_dir))]
synthesis_paths = [os.path.join(synthesis_fdr, image)
                                    for image in os.listdir(os.path.join(synthesis_fdr))]
original_paths = natsorted(original_paths)
synthesis_paths = natsorted(synthesis_paths)

tensors_list = []
def _flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
for index in range(len(original_paths)):
    image_path = original_paths[index]
    image = Image.open(image_path)
    syn_image_path = synthesis_paths[index]
    syn_image = Image.open(syn_image_path)
    flip_ran = random.random() > 0.5
    image = _flip(image, flip_ran)
    syn_image = _flip(syn_image, flip_ran)
    w = 512
    h = round(512 / 2)
    image_size = (h, w)
    common_transforms = [transforms.Resize(size=image_size, interpolation=Image.NEAREST),transforms.ToTensor()]
    base_transforms = transforms.Compose(common_transforms)
    
    syn_image_tensor = base_transforms(syn_image)
    image_tensor = base_transforms(image)
    print(syn_image_tensor.shape)
    norm_transform = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) #imageNet normamlization
    syn_image_tensor = norm_transform(syn_image_tensor)
    image_tensor = norm_transform(image_tensor)
    tensors_list.append({
                      'original': image_tensor,
                      
                      'synthesis': syn_image_tensor,
                      
                      'original_path': image_path,
                      
                      'syn_image_path': syn_image_path,
                      
                      })
    

# activate GPUs
gpu_ids = '0'
gpu = int(gpu_ids)
print(gpu)

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

from  torch.nn.modules.upsampling import Upsample
up5 = Upsample(scale_factor=16, mode='bicubic', align_corners=True)
up4 = Upsample(scale_factor=8, mode='bicubic', align_corners=True)
up3 = Upsample(scale_factor=4, mode='bicubic', align_corners=True)
up2 = Upsample(scale_factor=2, mode='bicubic', align_corners=True)
up1 = Upsample(scale_factor=1, mode='bicubic', align_corners=True)
to_pil = ToPILImage()

# Going through visualization loader
weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
vgg = VGG19().cuda(gpu)



with torch.no_grad():
    for i, data_i in enumerate(tensors_list):
        print('Generating image %i out of %i'%(i+1, len(tensors_list)))
        img_name = os.path.basename(data_i['original_path'])
        head_tail = os.path.split(img_name)
        img_name = head_tail[1]
        original = data_i['original'].cuda(gpu)
        synthesis = data_i['synthesis'].cuda(gpu)
        synthesis = torch.unsqueeze(synthesis, 0)
        original = torch.unsqueeze(original, 0)
        x_vgg, y_vgg = vgg(original), vgg(synthesis)
        feat5 = torch.mean(torch.abs(x_vgg[4] - y_vgg[4]), dim=1).unsqueeze(1)
        feat4 = torch.mean(torch.abs(x_vgg[3] - y_vgg[3]), dim=1).unsqueeze(1)
        feat3 = torch.mean(torch.abs(x_vgg[2] - y_vgg[2]), dim=1).unsqueeze(1)
        feat2 = torch.mean(torch.abs(x_vgg[1] - y_vgg[1]), dim=1).unsqueeze(1)
        feat1 = torch.mean(torch.abs(x_vgg[0] - y_vgg[0]), dim=1).unsqueeze(1)

        img_5 = up5(feat5)
        img_4 = up4(feat4)
        img_3 = up3(feat3)
        img_2 = up2(feat2)
        img_1 = up1(feat1)

        combined = weights[0] * img_1 + weights[1] * img_2 + weights[2] * img_3 + weights[3] * img_4 + weights[
            4] * img_5
        min_v = torch.min(combined.squeeze())
        max_v = torch.max(combined.squeeze())
        combined = (combined.squeeze() - min_v) / (max_v - min_v)

        combined = to_pil(combined.cpu())
        pred_name = 'mea_' + img_name
        combined.save(os.path.join(soft_fdr, pred_name))
