# BiteNet: food classification

## Group
- Year: 2025/2026
- Agata Benvegna, 1000069182

## Abstract
The goal of this project is to build a machine learning model for food classification. The dataset used to achieve this purpose is a subset of the "Food-101" open-source dataset. Three models have been trained on this dataset and their performances have been compared to choose the best one.
- **BiteNetV1**: a simple custom architecture designed to establish the training pipeline and familiarize the workflow with Google Colab;
- **BiteNetV2**: based on the AlexNet architecture pre-trained on ImageNet. It was adapted for this specific dataset and trained using data augmentation to mitigate overfitting;
- **BiteNetV3**: based on the ResNet18 architecture pre-trained on ImageNet, utilizing data augmentation and batch normalization for superior performance.

In the following table main results are presented:

| **Model**     | **Test accuracy**| **Epochs**      |
| :-----------: | :--------------: | :-------------: |
| BiteNetV1     | 0.298            | 40              |
| BiteNetV2     | 0.537            | 20              |
| BiteNetV3     | 0.808            | 20              |

## Introduction
Food classification is an example of an image classification task: given an input food image, we ask the model to guess the type of food depicted. The models presented in this project are more or less able to distinguish between 27 different dishes; naturally, they solve a quite specific problem within the framework of the food classification task.

The dataset chosen to achieve the above-mentioned task is the open-source dataset "Food-101," a collection of images belonging to 101 different categories of dishes. To be precise, a subset of 27 classes has been selected to reduce the amount of data to manage.

All models presented in this project have been trained and tested on this subset using notebooks inside the Google Colab environment. The first model, BiteNetV1, has a simple architecture from which great results were not expected; its main purpose was to familiarize with the Colab environment.

Both subsequent models were built using a transfer learning strategy, using the ImageNet dataset in the pre-training phase. BiteNetV2 has an AlexNet architecture, while BiteNetV3 is based on ResNet18. In a first stage, BiteNetV2 was trained for 20 epochs, after which the model was overfitting; then data augmentation was adopted as a regularization strategy to remove the overfitting effect, and the model was trained again for 20 epochs. This choice mostly removed the overfitting. Based on the BiteNetV2 results, BiteNetV3 was trained for 20 epochs directly applying data augmentation.

## Dataset
The "Food-101" dataset is a collection of 101,000 images belonging to 101 different categories, with a total dimension of 5 GB. For this project, a sample of 27 classes (1.42 GB) of the original dataset was chosen. The main purpose of this choice was the limited memory available and the need to prevent the training process from being too slow.

The dataset contains JPEG images of different dimensions, each stored inside a directory named after the corresponding class. Each directory contains the same number of images. The full dataset can be downloaded at: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/.

After downloading, 27 classes were randomly selected. A directory named "images" was created to store the subset; the directory was then zipped and uploaded to Google Drive. The dataset was randomly split into training, validation, and test subsets with the following percentages:

| **Training**  | **Validation**  | **Test**  |
| :-----------: | :-------------: | :-------: |
| 60%           | 10%             | 30%       |

The mean and standard deviation of the training set were calculated and stored in `normalization.yaml`. All preprocessing can be reproduced in `preprocessing.ipynb`. In this notebook, the image `/content/images/caprese_salad/1987239 (1).jpg` was removed: this was a 0-byte image found in the test batch after splitting. The first run of `bitenetv1_test.ipynb` failed due to this image, so the removal cell was added to the preprocessing notebook later. Fortunately, this image was not in the training batch, so there was no need to recalculate normalization parameters or restart training.


The composition of training, validation and test sets in the following.

*Training set composition*
![Training set composition](/media/training_set.png)

*Validation set composition*
![Validation set composition](/media/validation_set.png)

*Test set composition*
![Test set composition](/media/test_set.png)


## Metodology
In this project, three models have been built:
1) **BiteNetV1**: a simple model with 3 convolutional layers and 1 fully connected layer (see `bitenet_v1.py`). Its purpose was environment testing and checking the functionality of built modules.

The strategy of transfer learning has been chosen for the other two models, both pre-trained on the ImageNet dataset. The architectures and weights of these models were loaded from the `torchvision.models module`. Each of those models has built-in transformation of the data that is necessary to use the model in inference.

2) **BiteNetV2**: an AlexNet network using transfer learning (pre-trained on ImageNet). We modified the last layer for a size 27 output. Two experiments were conducted: one using built-in AlexNet transformations (resulting in overfitting) and a second applying data augmentation as regularization.

3) **BiteNetV3**: a ResNet18 network, modified for 27 outputs and trained directly with data augmentation, based on observations from BiteNetV2.

In the following table, all the training parameters choosen for all the experiments:

| **Learining rate** | **momentum** |
| :----------------: | :----------: |
| 0.01               | 0.99         |

## Esperiments
BiteNetV1 is the first model implemented. As already mentioned, it is a simple and unpretentious model, designed with the intention of defining the training strategy and testing some of the implemented modules.
For this model, three experiments were conducted. In the first experiment, the model was trained for 20 epochs. At the end of this phase, the loss was still decreasing, therefore a second training phase of 10 additional epochs was performed. Between the first and the second experiment, a decrease of 3.41% in validation loss and an increase of 10.39% in validation accuracy were observed. This improvement in validation accuracy motivated the start of a third experiment, consisting of an additional 10 training epochs.
However, the final 10 epochs did not significantly improve the model: the validation accuracy increased by only 1.04%, while the validation loss increased by 0.78%.
Based on these results, it was decided to discard the last 10 training epochs and to use the weights obtained from the second experiment during the test phase. The improvement in validation accuracy is not substantial enough to justify the increase in validation loss, even if the latter is relatively small.
The results discussed above are reported in the following tables and figures.

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

The orange, blue, and red curves correspond to Experiments 1, 2, and 3, respectively.


BiteNetV2 is the second model developed. It is based on the AlexNet architecture, pre-trained on the ImageNet dataset and modified to produce an output of 27 elements. The original AlexNet model has 1000 output units, corresponding to the number of classes in the ImageNet dataset.

The results of the experiments are shown in the figures below. The orange curves correspond to the first experiment, while the blue curves correspond to the second one.
In both experiments, the model was trained for 20 epochs

*Training accuracy*
![Training accuracy](/media/accuracy_train_2.png)

*Training loss*
![Training loss](/media/loss_train_2.png)

*Validation accuracy*
![Validation accuracy](/media/accuracy_test_2.png)

*Validation loss*
![Test set composition](/media/loss_test_2.png)


In the first experiment, we used standard AlexNet transformations, assuming the two dropout layers would provide enough regularization. However, overfitting occurred: the validation loss reached a minimum and then began to grow while training loss continued to drop.

In the second experiment, we applied data augmentation. Overfitting was greatly reduced; although the validation loss trend became somewhat constant (globally growing if comparing the first and last values), the results were better. Comparing Exp 1 to Exp 2: training loss increased by 112% and training accuracy dropped by 10.74%, but validation loss decreased by 17.13% and validation accuracy grew by 2.32%. This shows the model generalized much better.
The values already mentioned are in the tables below.

*Training*
| **Esperiment** | **Loss** | **ΔLoss (%)** | **Accuracy** | **ΔAcc(%)** |
| :------------: | :------: | :-----------: | :----------: | :---------: |
| 1              | 0.28     | —             | 0.91         | —           |
| 2              | 0.59     | +112.90%      | 0.81         | -10.74%     |

*Validation*
| **Esperiment** | **Loss** | **ΔLoss (%)** | **Accuracy** | **ΔAcc(%)** |
| :------------: | :------: | :-----------: | :----------: | :---------: |
| 1              | 1.69     | —             | 0.63         | —           |
| 2              | 1.40     | -17.13%       | 0.64         | +2.32%      |

There is a clear improvement from Experiment 1 to Experiment 2; however, the main issue with the second experiment is the nearly constant trend of the validation loss.
In this context, the use of batch normalization was considered in order to trigger a decrease in the validation loss. However, it is not straightforward to insert additional layers into a model with a fixed architecture; therefore, the simplest solution was to modify the model structure.
In fact, “BiteNetV3” is based on the ResNet-18 architecture, which does not rely on dropout layers but instead makes use of batch normalization.
This model was trained using data augmentation from the beginning; without it, the model would have likely overfitted.
The trends of the training and test metrics are shown below.

*Training accuracy*
![Training accuracy](/media/accuracy_train_3.png)

*Training loss*
![Training loss](/media/loss_train_3.png)

*Validation accuracy*
![Validation accuracy](/media/accuracy_test_3.png)

*Validation loss*
![Test set composition](/media/loss_test_3.png)

|             | **Loss** | **Accuracy** |
| :---------: | :------: | :----------: |
| Train       | 0.08     | 0.97         |
| Validation  | 1.03     | 0.79         |


After the training phase, each model was evaluated on the test set. The accuracies of the models are shown in the table below.

| **BiteNetV1** | **BiteNetV2** | **BiteNetV3** |
| :-----------: | :-----------: | :-----------: |
| 0.30          | 0.54          | 0.81          |

The performance of BiteNetV3 is clearly superior to that of BiteNetV2, as evidenced by the test accuracy.

## Demo
A demo of BiteNetV3 is available in `Demo.ipynb`. The model class includes a `predict()` method that requires only the image path as input. To use the model, the user simply needs to modify the image path variable to apply the model to the desired image.

## Conclusions
The test-accuracy values of each model are reported in the following table, already seen above:

| **BiteNetV1** | **BiteNetV2** | **BiteNetV3** |
| :-----------: | :-----------: | :-----------: |
| 0.30          | 0.54          | 0.81          |

It is clear that the performance of BiteNetV3 is superior to that of the other two models. The gap between BiteNetV3 and BiteNetV1 is the largest, which is expected given that the first version of the model is intentionally simple. Therefore, the most meaningful comparison is between BiteNetV2 and BiteNetV3.
BiteNetV3 outperforms BiteNetV2, mainly because it is based on the ResNet-18 architecture, which is deeper and more complex than AlexNet. It also shows better behavior with respect to overfitting.
Each model was also evaluated on a small batch of sample images to assess its performance in inference mode. The images are located in the `data/samples` directory, and the models’ behavior can be observed in the notebook `Example_BiteNet.ipynb`.
As expected, BiteNetV3 performs better than the other models on these images. However, it fails to correctly classify a cheesecake photographed from above, incorrectly labeling it as “cannoli”. Interestingly, the same image is correctly classified by BiteNetV2. On the other hand, all models correctly classify the image of edamame, likely because it is a clear example with minimal noise.
The models presented are relatively simple and do not achieve the performance of state-of-the-art models specifically designed for food recognition. Moreover, the problem addressed here represents only a small subset of the broader food classification task.
Further improvements could be achieved by training BiteNetV3 for a longer time; however, this may increase the risk of overfitting. Additionally, exploring different training hyperparameters could be beneficial in identifying a more optimal configuration for the model.


## Riferimenti
- [Torch library](https://pytorch.org)
- [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [Google Colab](https://colab.research.google.com)

