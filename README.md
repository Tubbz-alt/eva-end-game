# Mask and Dense Depth Estimation Using Single Network

## Objective
* Given background and actual image, network have to predict mask and dense depth of the objects
* We need to use only a single network, which takes in the two image inputs and gives two image output

![objective](https://github.com/santhiya-v/eva-end-game/blob/master/results/objective.png?raw=true)

## Dataset Preparation

* Dataset preparation itself is a big journey! 
* Details of how it is prepared can be found here : https://github.com/santhiya-v/EVA/tree/master/S15/A

## Model selection 
* Both the tasks in hand, depth estimation and mask prediction, needed encoder-decoder kind of architecture
* Went through few encoder-decoder architecture and chose U2Net since this had few advantages
  * The architecture of U2-Net is a two-level nested U-structure. 
  * It is able to capture more contextual information from different scales 
  * It increases the depth of the whole architecture without significantly increasing the computational cost
  ### Model Parameters
  Total params: 1,769,306
  
Each task was first individually experimented, before combining them into one network

## Mask / Image Segmentation
Used the model as in the https://github.com/NathanUA/U-2-Net for mask prediction

## Loss functions
* Used Multi BCE loss which was suggested in U2Net implementation

## Accuracy
* Calculated IOU for measuring accuracy of Image segmentation 

## Result of Mask prediction
![Mask Prediction](https://github.com/santhiya-v/eva-end-game/blob/master/results/mask_prediction.png?raw=true)

## Dense Depth
## Loss functions
* Used same BCE loss function since that was giving better results for dense depth as well
* Explored on SSIM, and other losses

## Accuracy
* Calculated RMSE for measuring accuracy of Dense depth estimation

## Result of Mask prediction
![Dense depth](https://github.com/santhiya-v/eva-end-game/blob/master/results/dense_depth_prediction.png?raw=true)

## Things considered for better training

## Model Training

## Time 

## Results






