# Diabetic Retinopathy Detection

## Introduction
- **Background**: Diabetic retinopathy is an eye disease that can lead to blindness and vision loss in people who have diabetes. In its early stages, diabetic retinopathy often shows no symptoms. However, as it progresses, it can result in a gradual decline in visual sharpness, potentially leading to complete blindness. According to the World Health Organization (WHO), 4.8% of the 37 million cases of blindness worldwide are attributed to diabetic retinopathy. This percentage is constantly rising, underscoring the critical need for timely and accurate diagnosis of diabetic retinopathy. Early detection is essential in preventing the progression of the disease and reducing the risk of severe vision loss, making it a significant public health priority.

## Data
- **Data Description**: The dataset utilized for this project is the EyePACS dataset, sourced from the [Diabetic Retinopathy Detection Challenge](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data) on Kaggle. It consists of 35.126 high resolution retinal images.
- **Data Preprocessing**: We employed two ways of processing the original images in order to get them in a reasonable size of 512x512.
  - Resizing and Cropping ([View Code](https://github.com/Stefanstud/diabetic-retinopathy-detection/blob/main/src/preprocessing/preprocessing_1.py))
  - Trimming and Aspect Ratio Maintenance ([View Code](https://github.com/Stefanstud/diabetic-retinopathy-detection/blob/main/src/preprocessing/preprocessing_2.py))

## Models
- **Overview of Models**: In this project, we have employed various neural network architectures to address the challenge of DR grading. Their implementation can be found below:
  1. **BiraNet**: Modified BiRA-Net with EfficientNetb3 backbone. Source: ([View Code](https://github.com/Stefanstud/diabetic-retinopathy-detection/blob/main/notebooks/bira_net.ipynb))
  2. **Siamese Network**: Combine left and right eye information. ([View Code](https://github.com/Stefanstud/diabetic-retinopathy-detection/blob/main/notebooks/siamese_net.ipynb))
  3. **Simple CNN**: Custom CNN for baseline comparisons. ([View Code](https://github.com/Stefanstud/diabetic-retinopathy-detection/blob/main/notebooks/simple_cnn.ipynb))
  4. **EfficientNet**: Pretrained EfficientNet model, weights from ImageNet. ([View Notebook](https://github.com/Stefanstud/diabetic-retinopathy-detection/blob/main/notebooks/efficient_net.ipynb))
  5. **Inception v3**: Implementation of Inception v3, weights from ImageNet. ([View Notebook](https://github.com/Stefanstud/diabetic-retinopathy-detection/blob/main/notebooks/inception_v3.ipynb))
  6. **ResNet**: Application of ResNet model, weights from ImageNet. ([View Notebook](https://github.com/Stefanstud/diabetic-retinopathy-detection/blob/main/notebooks/resnet.ipynb))
  7. **Vision Transformer (ViT)**: Utilizing ViT for image classification, weights from ImageNet. ([View Notebook](https://github.com/Stefanstud/diabetic-retinopathy-detection/blob/main/notebooks/vit.ipynb))
- **Visualization**: Heatmaps of our final model for multiple examples of profilerative DR retinas, using Grad-CAM. ([View Notebook](https://github.com/Stefanstud/diabetic-retinopathy-detection/blob/main/notebooks/grad-CAM.ipynb))
## Usage

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Stefanstud/CS502-diabetic-retinopathy-detection.git
   cd CS502-diabetic-retinopathy-detection
2. Create an environment using Python 3.8.18
   ```bash
   conda create --name dr_grading python==3.8.18
4. Activate the environment
   ```bash
   conda activate dr_grading
6. Install the required packages:
   ``` bash
   pip install -r requirements.txt
   
### Download Data
1. In section 1 of each notebook, there is a code for downloading the data used in this project. There are two directories, `images` and `images_keep_ar`. Using either of them is fine, however the second one `images_keep_ar` yielded better results. The pre-processed test data also comes with this folder.
2. After downloading the data, the folder should be organized in the following way:
``` bash
├── data
│   ├── images
│   ├── images_keep_ar
│   ├── labels
│   │   └── trainLabels.csv
│   └── test
├── notebooks
│   ├── bira_net.ipynb
│   ├── efficient_net.ipynb
│   ├── inception_v3.ipynb
│   ├── resnet.ipynb
│   ├── siamese_net.ipynb
│   ├── simple_cnn.ipynb
│   └── vit.ipynb
├── requirements.txt
├── results
│   ├── figures
│   └── models     
├── src
│   ├── loading.py
│   ├── models
│   │   ├── bira_net.py
│   │   ├── siamese_net.py
│   │   └── simple_cnn.py
│   ├── preprocessing
│   │   ├── preprocessing_1.py
│   │   └── preprocessing_2.py
│   ├── train.py
│   └── utils.py
└── submission.csv
```
### Reproducibility

To reproduce the best performing model, follow the detailed steps outlined in the project report. We provide three model files for flexibility and further research:

1. **eff_net_400x400.pt**: An EfficientNet model trained through multiple steps, achieving a 0.733 score on the private Kaggle test set.
2. **siamese_net_400x400_2.pt**: Our best performing model in terms of Quadratic Weighted Kappa, with a score of 0.764 on the private Kaggle test set.
3. **siamese_net_400x400_3.pt**: A model that utilizes penalty weights. While it has a slightly lower kappa score, it demonstrates a better confusion matrix.

For quick reproduction, you may use these pre-trained models. Alternatively, for potential customization, you can train the model from scratch following the guidelines in Table II of the report.

We also include checkpoints in the repository intended for those who wish to continue refining and improving the model. These checkpoints serve as a starting point for further training, allowing you to build upon the existing work without starting from the beginning.

Additionally, to generate the best submission file, you can run the `siamese_net.ipynb` notebook provided in this repository, specifically sections 0, 1, 2 and 4, without training a model. This notebook is specifically set up to work with the provided model files, which is designed for generating a submission file using our best performing model.

