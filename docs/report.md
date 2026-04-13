# BiteNet: food classification

## Group
- Year: 2025/2026
- Agata Benvegna, 1000069182

## Abstract
The goal of this project is to build a machine learning model for food classification.
The dataset used to achive this purpose is a subset of the "Food-101" open source dataset.
Three models have been trained on this dataset and their performances have been compared to choose the best one.
A brief description of the used methods in the following:
- **BiteNetV1**: the first model used, it has a simple architecture, basically built to familiarize with the training procedure and environment (Google Colab). As expected, its performaces are not good;
- **BiteNetV2**: this model is an AlexNet network pre-trained on the IMAGENET dataset, properly modified to be adapted to the used dataset and re-trained for the food classification task. The model has shown overfitting during the first training experiment, so it was trained agane using data augmentation as regularizzation method;
- **BiteNetV3**: this is a ResNet18 network pre-trained on the IMAGENET dataset, properly adapted and re-trained using data augmentation strategy.

In the following table main results are presented:

| **Model**     | **Test accuracy**| **Epochs**      |
| :-----------: | :--------------: | :-------------: |
| BiteNetV1     | 0.298            | 40              |
| BiteNetV2     | 0.537            | 20              |
| BiteNetV3     | 0.808            | 20              |

## Introduction
The food classification is an example of image classification task: given in input a food image, we ask the model to guess the type of food depicted.
The models presented in this project are more or less able to distinguish between 27 different dishes: needless to say, they solve a quite specific problem in the framework of food classification task. 

The dataset choose to achieve the above mentioned task is the open source dataset "Food-101", a collection of images belonging to 101 different cathegories of dishes. To be precise, a subset of 27 classes has been selected, in order to reduce the amount of data to manage.

All the models presented in this project have been trained and tested on this subset, running notebooks inside the Google Colab environment.
The first model is BiteNetV1, a model with a simple architecture, by which great results were not expected by principle. The main purpose of this model was to familiarize with the Colab environment.

Both the other two models have been built with the transfer learning strategy, using the IMAGENET dataset in the pre-training phase. 
BiteNetV2 is an AlexNet network, while BiteNetV3 is a ResNet18 network.
In a first stage, BiteNetV2 has been trained for 20 epochs, after which the model was in overfitting; then data augmentation was adopted as regularizzation strategy to remove the overfitting effect and the model was trained agane for 20 epochs. This choice has mostly removed the overfitting.

According to BiteNetV2 results, BiteNetV3 was trained for 20 epochs directly applying data augmentation. 

## Dataset
"Food-101" dataset is a collection of 101,000 images belonging to 101 different cathegories, with a total dimention of 5 GB.
For this project, a sample of 27 classes (1.42 GB) of the original dataset has been choosen. 
The main purpose of this choice was the limited memory available and also the need for the training process of not being too slow.

The dataset contains jpg images of different dimensions, each strored inside a directory named as the corresponding class. Each directory contains the same number of images. 

It is possible to download the entier "Food-101" dataset from the following link: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/.


The "Food-101" dataset has been downloaded from the link above; then 27 of the 101 classes have been randomly selected. A directory "images" has been created to store the subset; then the "image" directory has been zipped and put on Google Drive.


The dataset has been randomly splitted in train, test and validation subsets with the following percentages:

| **Training**  | **Validation**  | **Test**  |
| :-----------: | :-------------: | :-------: |
| 60%           | 10%             | 30%       |

Then mean and standard deviation of the training set have been calculated and stored in the file "normalizzation.yaml".

All the preprocessing procedure can be reproduced inside the "preprocessing.ipynb" notebook. In this notebook it can be seen that the image "/content/images/caprese_salad/1987239 (1).jpg" has been removed from the dataset. This is a 0 byte image put inside the test batch after the splitting of the dataset.
In fact, the first running of "bitenetv1_test.ipynb" gave an error due to this image and the cell in which this image is removed has been added to the preprocessing notebook in a second time.
Fortunately, this image was not inside the training batch, so there was no need neither of calculating agane normalizzation parameters nor restart the training of the model.


The composition of training, validation and test sets in the following.

*Training set composition*
![Training set composition](/media/training_set.png)

*Validation set composition*
![Validation set composition](/media/validation_set.png)

*Test set composition*
![Test set composition](/media/test_set.png)


## Metodology
In this project, three models have been built:
1) **BiteNetV1**: this is a really simple model, built up with 3 convolutional layers and 1 layer fully connected. The architecture of the model can be seen in the module "bitenet_v1.py". As already specified, the main purpose of this model was the one of familiarize with the Google Colab environment and check the correct funtionality of built functions.

The strategy of transfer learning has been choosen for the other two models, both pre-trained on the IMAGENET dataset. The structure and wheigts of the models have been downloaded from the "torchvision.model" module. Each of those models has built-in transformation of the data that is necessary to use the model in inference.

2) **BiteNetV2**: this is an "AlexNet" network. The last layer has been changed in order to have in output a tensor of size 27. After that, two training experiments have been done: in the first experiment, the data have been transformed using the alexnet built-in trasformation ; with this strategy, the model had shown overfitting, so the second experiment has been done applying regularizzation. In particular, data have been transformed in order to apply data augmentation.

3) **BiteNetV3**: the last model is the most performative. This is a ResNet18 network, also in this case modified to give in output a tensor of dimension 27.
This model has been trained directly applying data augmentation, because of the results observed with the model BiteNetV2.

In the following tabel, all the training parameters choosen for all the experiments:

| **Learining rate** | **momentum** |
| :----------------: | :----------: |
| 0.01               | 0.99         |

## Esperiments
"BiteNetV1" is the first model implemented. As alredy mentioned, it is a simple and unpreticious model, built with the intension of setting the training strategy and testing some of the built modules.
For this model, 3 experiments have been done. In the first experiment, the model has been trained for 20 epochs, at the end of which the loss trend was still undergoing and, for this reason, a second training cycle of 10 other epocs was done. A decrease of the 3.41% of the validation loss and an increase of 10.39% of the validation accurancy were observed between the first and the second experiment. The increase in the validation accurancy was the reason why a third experiment was started, training the model for 10 other epochs.

The last 10 epocs have not improved in a significant way the model: the validation accuracy increases of 1.04%, while the validation loss increases of 0.78%.
Because of the results presented above, it was decided to ignore the last 10 training epochs and to take into account the weights of the second experiment during the test phase. The improvement in the accuracy of the model does not make a too relevant difference to ignore the increase in the validation loss, even if small.
The results discussed above are in the following tables and figures.

*Training*
| **Esperiment** | **Loss**  | **ΔLoss (%)** | **Accuracy** | **ΔAcc(%)** |
| :------------: | :-------: | :----------:  | :----------: | :---------: |
| 1              | 2.36      | —             | 0.32         | —           |
| 2              | 2.15      | -8.91%        | 0.37         | +15.99%     |
| 3              | 1.92      | -10.67%       | 0.43         | +15.73%     |

*Validation*
| **Esperiment** | **Loss**  | **ΔLoss (%)**| **Accuracy** | **ΔAcc(%)** |
| :------------: | :-------: | :----------: | :----------: | :---------: |
| 1              | 2.52      | —            | 0.28         | —           |
| 2              | 2.44      | -3.41%       | 0.31         | +10.39%     |
| 3              | 2.46      | +0.78%       | 0.31         | +1.04%      |


*Training accuracy*
![Training accuracy](/media/accuracy_train_1.png)

*Training loss*
![Training loss](/media/loss_train_1.png)

*Validation accuracy*
![Validation accuracy](/media/accuracy_test_1.png)

*Validation loss*
![Test set composition](/media/loss_test_1.png)

The orange curves are for experiment 1, the blue ones for the second experiment and the red ones for the third one.

"BiteNetV2" is the second built model. This is an "AlexNet" network, pre-trained on the IMAGENET dataset and modified in order to have an output of 27 elements (the output of the original AlexNet has 101 elements, corresponding to the number of classes between which the model can choose).
The results of the experiment are shown in the figures below: the orange curves refere to the first experiment, while the blu ones to the second one. 
In both the experiments, the model has been trained for 20 epochs.

*Training accuracy*
![Training accuracy](/media/accuracy_train_2.png)

*Training loss*
![Training loss](/media/loss_train_2.png)

*Validation accuracy*
![Validation accuracy](/media/accuracy_test_2.png)

*Validation loss*
![Test set composition](/media/loss_test_2.png)


In the first training experiment, the images of the dataset have been transformed under the transformation bult-in the alexnet network. This choice was taken because the "AlexNet" network has two dropout level, so in a first moment it was thought to not apply data augmentation to the dataset, because the already present regularizzation strategy.
Unfortunatelly, this was not sufficient to delete the overfitting, clearly visible in the trend of the test loss: the loss curve reaches a minimum and then it starts growing, while the training loss still goes down.
In the second training experiment, it was decided to apply data augmentation. Looking at the test-loss trend, one can see that the overfitting is greatly reduced; nevertheless, the loss-test has not a descendent trend, but it remains more or less constant, actually if one considers the first and the last value, it globally grows.

In the tables below, a comparison between accuracy and loss values of experiment 1 and 2. One can notice the big increase in the training loss (+112%) and the decrease of the training accuracy (+10%) going from the first to the second experiment; on the contrary, the test loss decrease of the 17.13% and the test accuracy grows of the 2.32%. 
Those results are reasonable and show an immprovement of the model: the training parameters of the second experiment are worse than the ones of the first because in the first experiment the model is overfitted, so it goes better than the second one on the training data; the test parameters, instead, are better in the second experiment, because the model is less overfitted than the first one and it can generalize more.

*Training*
| **Esperiment** | **Loss** | **ΔLoss (%)** | **Accuracy** | **ΔAcc(%)** |
| :------------: | :------: | :-----------: | :----------: | :---------: |
| 1              | 0.28     | —             | 0.91         | —           |
| 2              | 0.59     | +112.90%      | 0.81         | -10.74%     |

*Test*
| **Esperiment** | **Loss** | **ΔLoss (%)** | **Accuracy** | **ΔAcc(%)** |
| :------------: | :------: | :-----------: | :----------: | :---------: |
| 1              | 1.69     | —             | 0.63         | —           |
| 2              | 1.40     | -17.13%       | 0.64         | +2.32%      |

There is a clear improvement from experiment 1 to experiment 2; the only problem of the experiment 2 is the constant trend of the test loss. In instance, it was though to use batch normalizzation, in order to make the test loss decreasing, but it is not possible to insert a new layer in a model with a fixed structure: the simplest solution was to change the model structure.
In fact, "BiteNetV3" is a "ResNet18" network, in which there are no dropout layers but batch normalizzation is used.
This model has been trained directly using data augmentation: without it, the model whould have being surelly overfitted.

The trend of training and test parameters is shown below.

*Training accuracy*
![Training accuracy](/media/accuracy_train_3.png)

*Training loss*
![Training loss](/media/loss_train_3.png)

*Validation accuracy*
![Validation accuracy](/media/accuracy_test_3.png)

*Validation loss*
![Test set composition](/media/loss_test_3.png)

|           | **Loss** | **Accuracy** |
| :-------: | :------: | :----------: |
| Train     | 0.08     | 0.97         |
| Test      | 1.03     | 0.79         |


After the training phase, the model has been tested on the test set. The accuracy of the models on the test batch are shown in the table below:

| **BiteNetV1** | **BiteNetV2** | **BiteNetV3** |
| :-----------: | :-----------: | :-----------: |
| 0.30          | 0.54          | 0.81          |

It is clear that the performace of "BiteNetV3" are better than the ones of "BiteNetV2": it is sufficient looking at the test-accuracy.

## Demo
The Demo of the "BiteNetV3" model is the notebook "Demo.ipynb".
In this notebook an instance of the model is created and the weights of the model are loaded from the path 'results/model_name/model/model_name.pth'.

The model is a class that comes with the method "predict()"; this function takes in input the path of the image the user wants to infer on, so the user only needs to change the image path variable to apply the model on the image he wants. 

## Conclusions
The test-accuracy values of each model are reported in the following table, already seen above:

| **BiteNetV1** | **BiteNetV2** | **BiteNetV3** |
| :-----------: | :-----------: | :-----------: |
| 0.30          | 0.54          | 0.81          |

It is clear that the performances of "BiteNetV3" are better than those of the other two models. The gap between "BiteNetV3" and "BiteNetV1" its the biggest and it was expected because the first version of the model is really simple, so the relevant comparison is the one between "BiteNetV2" and "BiteNetV3".

"BiteNetV3" works better than "BiteNetV2" and this is because "ResNet18" is a more complex and deep network than "AlexNet"; it is better also from the point of view of the overfitting.

Each model has been used on a micro-batch of photos to try them in inference mode. The images are inside the directory "data/samples" and the behaviour of the models can be seen inside the notebooks "Example - *.ipynb".
As expected, "BiteNetV3" works better than the other two models also on those images, but it fails in recognizing a photographed from above cheescacke, classified as "cannoli". Instead, the same image has been correctly classified by "BiteNetV2". The image about which all the models agree is the one of edamame, probably because it is a very clear image, without a lot of disturbing noise.

The models presented are very simple ones and they do not work as well as already existing models specialized in the food recognition task. Moreover, as already said, the faced problem is like a drop in the ocean of the food classification task.
One could obtain better results training more "BiteNetV3", but then probably the problem of overfitting could rase. Also changing the training parameters cold be intresting, in order to see if there is a better combination that optimizes the model much better. 


## Riferimenti
- [Torch library](https://pytorch.org)
- [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [Google Colab](https://colab.research.google.com)

