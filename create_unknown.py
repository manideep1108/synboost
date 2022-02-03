import os
from PIL import Image
from natsort import natsorted
import numpy as np
import random

from options.test_options import TestOptions
TestOptions = TestOptions()
opt = TestOptions.parse()
import sys
from image_dissimilarity.util import visualization
import image_dissimilarity.data.cityscapes_labels as cityscapes_labels


import torchvision
from torchvision.transforms import ToPILImage
import cv2
from collections import OrderedDict
import shutil
import torch
from torch.backends import cudnn
import torchvision.transforms as transforms



trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid
#objects_to_change = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33] # equal_prob
objects_to_change = [7, 7, 7, 8, 11, 12, 13, 20, 21, 22, 24, 24, 24, 25, 26, 26, 26, 26, 26, 26, 26, 27, 28, 32, 33] # optmized
#objects_to_change = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28, 31, 32, 33] # no person

instance_path = opt.instances_og
semantic_path = opt.semantic_og
original_path = opt.demo_folder
save_dir = opt.results_dir
visualize=False
dynamic=True

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

if not os.path.isdir(os.path.join(save_dir, 'labels')):
    os.mkdir(os.path.join(save_dir, 'labels'))

if not os.path.isdir(os.path.join(save_dir, 'semantic_labelId')):
    os.mkdir(os.path.join(save_dir, 'semantic_labelId'))

if not os.path.isdir(os.path.join(save_dir, 'semantic')):
    os.mkdir(os.path.join(save_dir, 'semantic'))

if not os.path.isdir(os.path.join(save_dir, 'original')):
    os.mkdir(os.path.join(save_dir, 'original'))

# for creating synthesis later
if not os.path.exists(os.path.join(save_dir, 'temp')):
    os.makedirs(os.path.join(save_dir, 'temp'))
    os.makedirs(os.path.join(save_dir, 'temp', 'gtFine', 'val'))
    os.makedirs(os.path.join(save_dir, 'temp', 'leftImg8bit', 'val'))

semantic_paths = [os.path.join(semantic_path, image)
                        for image in os.listdir(semantic_path)]
instance_paths = [os.path.join(instance_path, image)
                        for image in os.listdir(instance_path)]
original_paths = [os.path.join(original_path, image)
                    for image in os.listdir(original_path)]

semantic_paths = natsorted(semantic_paths)
instance_paths = natsorted(instance_paths)
original_paths = natsorted(original_paths)

for idx, (semantic, instance, original) in enumerate(zip(semantic_paths, instance_paths, original_paths)):
    print('Generating image %i our of %i' %(idx+1, len(semantic_paths)))

    semantic_img = np.array(Image.open(semantic))
    instance_img = np.array(Image.open(instance))
    original_img = Image.open(original)
    
    unique_classes = [sample for sample in np.unique(instance_img) if len(str(sample)) == 5]

    how_many = int(random.random()*len(unique_classes))

    final_mask = np.zeros(np.shape(instance_img))
    new_semantic_map = np.copy(semantic_img)

    # Make final mask by selecting each instance to replace at random
    for _ in range(how_many):
        # instance to change
        instance_idx = int(random.random()*len(unique_classes))
        instance_change = unique_classes.pop(instance_idx)

        # get mask where instance is located
        mask = np.where(instance_img==instance_change, 1, 0)

        while True:
            new_instance_idx = int(random.random()*len(objects_to_change))
            new_instance_id = objects_to_change[new_instance_idx]

            # ensure we don't replace by the same class
            if new_instance_id != int((str(instance_change)[:2])):
                break
        np.place(new_semantic_map, mask, new_instance_id)
        final_mask += mask

    # also mark dynamic labels (Optional)
    if dynamic:
        mask = np.where(semantic_img==5, 1, 0)
        final_mask += mask

    new_semantic_name = os.path.basename(semantic).replace('labelIds', 'unknown_labelIds')
    new_semantic_train_name = os.path.basename(semantic).replace('labelIds', 'unknown_trainIds')
    new_label_name = os.path.basename(instance).replace('instanceIds', 'unknown')
    old_semantic_name = os.path.basename(semantic)
    new_original_name = os.path.basename(original).replace('leftImg8bit', 'unknown_leftImg8bit')

    mask_img = Image.fromarray((final_mask).astype(np.uint8))

    if visualize:
        if not os.path.isdir(os.path.join(save_dir, 'old_semantic')):
            os.mkdir(os.path.join(save_dir, 'old_semantic'))
            
        # Correct labels to train ID for old semantic
        semantic_copy = semantic_img.copy()
        for k, v in id_to_trainid.items():
            semantic_copy[semantic_img == k] = v
        semantic_img = semantic_copy.astype(np.uint8)

        # Correct labels to train ID for new semantic
        semantic_copy = new_semantic_map.copy()
        for k, v in id_to_trainid.items():
            semantic_copy[new_semantic_map == k] = v
        new_semantic_map =semantic_copy.astype(np.uint8)

        new_semantic_img =visualization.colorize_mask(new_semantic_map)
        old_semantic_img = visualization.colorize_mask(semantic_img)

        # save images
        mask_img.save(os.path.join(save_dir, 'labels', new_label_name))
        new_semantic_img.save(os.path.join(save_dir, 'semantic', new_semantic_name))
        original_img.save(os.path.join(save_dir, 'original', new_original_name))
        old_semantic_img.save(os.path.join(save_dir, 'old_semantic', old_semantic_name))
    else:
        new_semantic_img = Image.fromarray(new_semantic_map)

        # Correct labels to train ID for new semantic
        semantic_copy = new_semantic_map.copy()
        for k, v in id_to_trainid.items():
            semantic_copy[new_semantic_map == k] = v
        new_semantic_map =semantic_copy.astype(np.uint8)
        new_semantic_train_img = Image.fromarray(new_semantic_map)

        # save images
        mask_img.save(os.path.join(save_dir, 'labels', new_label_name))
        original_img.save(os.path.join(save_dir, 'original', new_original_name))
        new_semantic_img.save(os.path.join(save_dir, 'semantic_labelId', new_semantic_name))
        new_semantic_train_img.save(os.path.join(save_dir, 'semantic', new_semantic_train_name))

        # save images for synthesis
        original_img.save(os.path.join(save_dir, 'temp', 'leftImg8bit', 'val', new_original_name))
        new_semantic_img.save(os.path.join(save_dir, 'temp', 'gtFine', 'val', new_semantic_name))

        instance_img = Image.open(instance)
        instance_img.save(os.path.join(save_dir, 'temp', 'gtFine', 'val', os.path.basename(instance)))

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

synthesis_paths = [os.path.join(synthesis_fdr, image)
                                    for image in os.listdir(os.path.join(synthesis_fdr))]
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
