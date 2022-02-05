# Pixel-wise Anomaly Detection in Complex Driving Scenes
This repository is PyTorch Lightning implementation of the paper, ["Pixel-wise Anomaly Detection in Complex Driving Scenes"](https://arxiv.org/abs/2103.05445). It is well documented version of the original repository with the code flow available [here](https://github.com/giandbt/synboost). The paper address the problem of anomaly segmentation.

![Alt text](display_images/methodology.png?raw=true "Methodology")

### Requirements

1) To install the requirements:
```pip install -r requirements_demo.txt```

2) Wandb has been used to log files and data, so it is necessary to have a wandb account (https://wandb.ai/site), details of which need to be added in the config file (`image_dissimilarity/configs/train/default_configuration.yaml`). 


### Pretrained Weights

1) Dissimilarity Module : [Link(dissimilarity models)](https://drive.google.com/drive/folders/16ELWb4Qu0AZ5dolf1vT5SoIkpdNR59DR?usp=sharing)


### Datasets 

- This is the link to dataset of full framework (provided in the original author's repo) : [Link(original)](http://robotics.ethz.ch/~asl-datasets/Dissimilarity/data_processed.tar)
- Link to dataset of light data : [Link(light data)](https://www.kaggle.com/mlrc2021anonymous/synboost-light-data)
- Link to dataset of w/o data generator: [Link(without datagenerator)](https://www.kaggle.com/mlrc2021anonymous/synboost-without-data-generator)


### Training 
In order to train the dissimilarity network, we have to do the following:

1) Modify the necessary parameters in the configuration file `configs/train/default_configuration.yaml`. 
More importanly, modify the folder paths for each dataset to your local path. In order to get the required data for training, also add the deatils of wandb in the configuration file.
2) ```
   python train_lightning.py --config configs/train/default_configuration.yaml
   ```
3)The following file can be run to train the model in kaggle : [Link(kaggle notebook)](https://www.kaggle.com/mlrc2021anonymous/synboost-pytorch)

### Evaluation
To Run ensemble(with grid search):
1) For FS Static: 
```
python test_lightning_ensemble.py --config configs/test/fs_static_configuration.yaml 
```

2) For Lost and Found : 
```
python test_lightning_ensemble.py --config configs/test/fs_lost_found_configuration.yaml 
```

To Run testing directly with fixed weights(this also saves the final prediction images where the anomaly has been segmented, for this the model path base in wandb must be provided):
1) For FS Static: 
```

python test_lightning.py --config configs/test/fs_static_configuration.yaml 
``` 
2) For Lost and Found : 
```

python test_lightning.py --config configs/test/fs_lost_found_configuration.yaml 
```
   
### Results

![alt text](https://github.com/manideep1108/synboost/blob/master/display_images/Comapring%20oututs%20of%20ours%20and%20authors.jpeg?raw=true)
The above image compares author's final predictions (2nd column from right) with our predictions(last column).


<p align="center">
  <img src="https://github.com/manideep1108/synboost/blob/master/display_images/main%20results.png?raw=true" alt="Sublime's custom image"/>
</p>
<p align="center">
  The above table shows the results on the the Fishyscapes private test data
</p>

<p align="center">
  <img src="https://github.com/manideep1108/synboost/blob/master/display_images/table%202.png?raw=true" alt="Sublime's custom image"/>
</p>
<p align="center">
  The above table shows the results on the Fishyscapes Validation datasets
</p>

<p align="center">
  <img src="https://github.com/manideep1108/synboost/blob/master/display_images/light.png?raw=true" alt="Sublime's custom image"/>
</p>
<p align="center">
  The above table shows the results of Light version 
</p>



## References
[1] Learning to Predict Layout-to-image Conditional Convolutions for Semantic Image Synthesis.
Xihui Liu, Guojun Yin, Jing Shao, Xiaogang Wang and Hongsheng Li.

[2] Improving Semantic Segmentation via Video Propagation and Label Relaxation
Yi Zhu1, Karan Sapra, Fitsum A. Reda, Kevin J. Shih, Shawn Newsam, Andrew Tao, Bryan Catanzaro.

[3] https://github.com/lxtGH/Fast_Seg

[4] High-resolution image synthesis and semantic manipulation with conditional gans.
Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. 

[5] The cityscapes dataset for semantic urban scene understanding. 
Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele
