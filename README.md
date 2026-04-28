# BiteNet

In this project, three models are developed for a food classification task and compared with each other to determine the best-performing one.

* **BiteNetV1**: a simple custom architecture designed to establish the training pipeline and become familiar with the workflow in Google Colab.
* **BiteNetV2**: based on the AlexNet architecture pre-trained on ImageNet. It is adapted for this specific dataset and trained using data augmentation to mitigate overfitting.
* **BiteNetV3**: based on the ResNet18 architecture pre-trained on ImageNet, using data augmentation and batch normalization to achieve superior performance.

---

## Project Structure

```
bitenet/
├── .github/
│   └── workflows/
├── data/
├── docs/
├── media/
├── notebooks/
├── project_ideas/
├── results/
├── src/
├── tests/
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Usage

### 1. Dataset Preparation

To ensure reproducibility, download the Food-101 dataset:
https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

Then:

* Select **27 classes**
* Place them into a directory named `image`
* Zip the directory
* Upload it to your Google Drive

---

### 2. Training and Evaluation (Google Colab)

The preprocessing, training, and testing procedures can be reproduced by opening the following notebooks in Google Colab:

* `preprocessing.ipynb`
* `BiteNetV*_training.ipynb`
* `BiteNetV*_test.ipynb`

---

### 3. Local Demo (Jupyter Lab)

The notebooks:

* `Example_BiteNet.ipynb`
* `Demo.ipynb`

can be run locally using Jupyter Lab.

#### Setup virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install jupyterlab
```

---

### 4. Running Tests

To run the test suite:

```bash
python -m unittest discover
```

