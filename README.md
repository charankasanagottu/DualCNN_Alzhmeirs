# Alzheimer's Disease MRI Classification using Dual Convolutional Neural Network

## Overview

Alzheimer's disease is a progressive neurological disorder that leads to
the degeneration of brain cells and the eventual loss of memory and
cognitive functions. Early detection and classification into stages can
greatly improve treatment and care.

This project aims to classify MRI scans of patients into four distinct
stages of Alzheimer's disease using a **Dual Convolutional Neural
Network (CNN)** architecture, leveraging both pre-trained models and a
customized CNN for enhanced accuracy and precision.

## Project Highlights

-   **MRI Classification**: The model classifies patient MRI scans into
    four stages of Alzheimer's disease.
-   **Dual CNN Architecture**: Combines powerful pre-trained models
    (VGG16, ResNet50V2, DenseNet169) with a customized CNN.
-   **High Performance**: Achieved **98% accuracy** and **95%+
    precision** in all classes.

------------------------------------------------------------------------

## Project Workflow

### 1. Data Preprocessing

-   **Image Data Generator (IDG)**: Used for preprocessing MRI images to
    improve model generalization by applying transformations like
    rescaling, rotation, zoom, and more.
-   **SMOTE Analysis**: Employed **Synthetic Minority Oversampling
    Technique (SMOTE)** to address class imbalance, ensuring equal
    representation across all classes.

### 2. Model Architecture

-   **Dual Convolutional Neural Network (Dual-CNN)**:
    -   Combines a **pre-trained CNN** (VGG16, ResNet50V2, DenseNet169)
        with a **custom CNN**.
    -   Both models process the input MRI data in parallel.
    -   The outputs from both networks are concatenated and passed
        through fully connected layers before classification with a
        **SoftMax** layer.

### 3. Training and Optimization

-   **Transfer Learning**: Utilized pre-trained weights from the
    ImageNet dataset to transfer knowledge to the Alzheimer's
    classification task.
-   **Customized CNN**: Built a lightweight CNN architecture with
    additional convolutional and fully connected layers to complement
    the pre-trained models.
-   **Model Training**: Trained the Dual CNN model using the combined
    architecture, with fine-tuning for optimal performance.
-   **Evaluation**: Achieved high accuracy and precision across all
    Alzheimer's stages with the **VGG16 + customized CNN combination**.

### 4. Performance Metrics

-   **Accuracy**: 98%
-   **Precision**: Over 95% across all four stages.

------------------------------------------------------------------------

## Dual CNN Architecture

The Dual Convolutional Neural Network (CNN) model used in this project
processes the input data through two parallel networks: a pre-trained
CNN (like VGG16) and a custom CNN. The outputs of these networks are
concatenated and processed through fully connected layers before final
classification.

**Pictorial Representation:**

Below is a simplified illustration of the Dual CNN architecture (Add an
image here if possible):

``` plaintext
                Input MRI
                    |
      +----------------------------+
      |                            |
  Pre-trained CNN (VGG16)       Customized CNN
      |                            |
  Feature Map 1                Feature Map 2
      |                            |
      +------------+---------------+
                   |
              Concatenation
                   |
              Fully Connected Layers
                   |
                SoftMax Output
```

------------------------------------------------------------------------

## Installation & Requirements

### Prerequisites

-   **Python 3.7+**
-   **TensorFlow 2.0+**
-   **Keras**
-   **SMOTE** (from `imbalanced-learn`)
-   **Matplotlib, NumPy, Pandas**

### Steps to Run

1.  **Clone the repository**:

    ``` bash
    git clone https://github.com/charankasanagottu/DualCNN_Alzhmeirs.git
    cd Alzheimers-Classification
    ```

2.  **Install dependencies**:

    ``` bash
    pip install -r requirements.txt
    ```

3.  **Prepare Data**:

    -   Place your MRI data in the `data/` directory.
    -   Ensure the data is structured according to the four stages of
        Alzheimer's.

4.  **Train the Model**:

    ``` bash
    python train_model.py
    ```

5.  **Evaluate the Model**:

    -   After training, the results will be saved, and you can run the
        evaluation script to check the performance.

------------------------------------------------------------------------

## Results

-   **Accuracy**: 98%
-   **Precision**: Over 95% for all stages.
-   **Best Model**: VGG16 + Customized CNN yielded the best results for
    classification accuracy and precision.

------------------------------------------------------------------------

## Future Work

-   Working on Interactive Website using Flask, Streamlit using python.
-   Enhance model generalization by experimenting with additional
    architectures or hybrid methods.
-   Explore real-time MRI classification using this model in clinical
    applications.

------------------------------------------------------------------------

## License

This project is licensed under the MIT License - see the LICENSE.md file
for details.
