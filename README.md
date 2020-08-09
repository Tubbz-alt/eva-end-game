# Mask and Dense Depth Estimation Using Single Network

## Objective
* Given background and actual image, network have to predict mask and dense depth of the objects
* We need to use only a single network, which takes in the two image inputs and gives two image output

![objective](https://github.com/santhiya-v/eva-end-game/blob/master/results/objective.png?raw=true)

## Dataset Preparation

* Dataset preparation itself is a big journey! 
* Details of how it is prepared can be found here : https://github.com/santhiya-v/EVA/tree/master/S15/A
* So now we have dataset prepared and few samples of inputs and ground truth can be seen below :
![Dataset](https://github.com/santhiya-v/eva-end-game/blob/master/results/dataset.png?raw=true)


## Model selection 
* Both the tasks in hand, depth estimation and mask prediction, needed encoder-decoder kind of architecture
* Went through few encoder-decoder architecture and chose U2Net since this had few advantages
  * The architecture of U2-Net is a two-level nested U-structure. 
  * It is able to capture more contextual information from different scales 
  * It increases the depth of the whole architecture without significantly increasing the computational cost
  ### Model Parameters
  Total params: 1,769,306
  
## Approach
* A small dataset of 12k was taken to carry out experiments
* Each task was first individually experimented, before combining them into one network
* Experimented with varying batch size, workers, loss functions to arrive at optimum methods
* Arrived at a single network to train both dense depth and mask in single network
* Started with smaller image size and gradually increased the image size for best accuracy

## Mask / Image Segmentation
Used the model as in the https://github.com/NathanUA/U-2-Net for mask prediction

### Loss functions
* Used Multi BCE loss which was suggested in U2Net implementation, which calculates Binary Cross Entropy loss for all the sides output

### Accuracy
* Calculated IOU for measuring accuracy of Image segmentation 

### Result of Mask prediction (after 2 epochs)
![Mask Prediction](https://github.com/santhiya-v/eva-end-game/blob/master/results/mask_prediction.png?raw=true)

## Dense Depth
Used same U2Net model for dense depth estimation

### Loss functions
* Used multi BCE loss function since that was giving better results for dense depth as well
* Explored on SSIM loss but BCE was giving better result 

### Accuracy
* Calculated RMSE for measuring accuracy of Dense depth estimation

### Result of Dense depth prediction (after 2 epochs)
![Dense depth](https://github.com/santhiya-v/eva-end-game/blob/master/results/dense_depth_prediction.png?raw=true)

## Things considered for better training
* Fg Bg Mask and Dense depth was prepared as 1 channel image for memory efficiency
* Dataset paths are pickled and saved. Dataloader loads the pickled data everytime. This reduces timing on framing dataset paths
* Training from drive was huge time consuming and hence moved the dataset to colab for faster running

## Model Training
* Model was initally trained on smaller size and gradually increased the size
* Model is saved after every epoch
* On increasing the image size, previous model state was loaded and training is continued from there
* Details of no of epochs, batch size, image size, accuracy, loss is given below:

No Of Epochs | Batch Size | Image size | Time Taken | Mask Accuracy (IOU) | Dense Depth Accuracy (RMSE) | Loss |
------ | ----- | ---------- | ---------- | -------------- | --------- | -------------- |
10 | 128 | 64*64 | 5.8 hrs | 0.9493 | 0.0666 | 4.2672 |
2 | 128 | 96*96 | 1.5 hrs | 0.9501 | 0.0689 | 4.0669 |
2 | 128 | 112*112 | 1.9 hrs |  0.9501 | 0.0689 | 4.0669 |
1 | 32 | 224*224 | 3.1 hrs | 0.9685 | 0.0801 | 3.988 |

## Time 
Task | Time |
---- | ---- |
Average train time for 1 Epoch | 36 Mins |
Average test time for 1 Epoch  | 7 Mins |
Time to load dataset to colab  | 4.9 Mins |

## Results
### Results after 10 Epochs (Trained on 64*64)
![prediction_64](https://github.com/santhiya-v/eva-end-game/blob/master/results/prediction_aft_10_epochs.png?raw=true)

### Results after 12 Epochs (Trained on 96*96)
![prediction_96](https://github.com/santhiya-v/eva-end-game/blob/master/results/prediction_aft_12_epochs_96.png?raw=true)

### Results after 14 Epochs (Trained on 112*112)
![prediction_112](https://github.com/santhiya-v/eva-end-game/blob/master/results/prediction_aft_12_epochs_96.png?raw=true)

### Results after 16 Epochs (Trained on 224*224)
![prediction_224](https://github.com/santhiya-v/eva-end-game/blob/master/results/prediction_aft_12_epochs_96.png?raw=true)

## References
* https://github.com/NathanUA/U-2-Net
* https://arxiv.org/pdf/2005.09007.pdf
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7038358/
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6514714/






