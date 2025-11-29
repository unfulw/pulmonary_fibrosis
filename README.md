# Baseline CT - Augmented FVC Forecasting to Project Pulmonary Fibrosis Progression, Aiding Medical Decision-Making and Prognosis

## Dataset Description

The dataset contains baseline lung CT scans in the format of DICOM (.dcm) file (which includes grayscale pixel data of scans and metadata), anonymized patient demographics and the weekly forced vital capacity (FVC) measurement of pulmonary fibrosis patients, a crucial and effective lung function indicator. Data is collected by the Open Source Image Consortium, a "not-for-profit cooperative effort between academia, industry and patient advocacy groups" that focuses on respiratory disease, providing highly reliable data provided by experts and leading organizations.

**Data Components:**
- **DICOM CT Scans**: Baseline lung CT images stored as .dcm files containing grayscale pixel data and metadata
- **Tabular Data**: Patient demographics (Age, Sex, SmokingStatus) and weekly FVC measurements
- **Target Variable**: FVC (Forced Vital Capacity) in milliliters
- **Temporal Features**: Weeks relative to baseline CT scan

## Motivation

Pulmonary fibrosis (PF) is a progressive chronic lung disease that scars the tissue around the alveoli, leading to irreversible decline in lung function. Doctors have a hard time determining how rapidly patients' conditions will decline, putting patients under extreme anxiety. As no curative treatment currently exists, patients often undergo frequent monitoring and long-term therapy, which together create a substantial financial and healthcare burden. 

This project proposes a predictive model that leverages longitudinal baseline lung CT scans alongside patient demographic and clinical information to forecast individual disease trajectories. By providing accurate forecasts, the model aims to support timely treatment adjustments while reducing the intensity and frequency of monitoring, ultimately helping to ease both the clinical and financial strain of PF management.

## Project Overview

This project implements and compares multiple approaches for predicting FVC decline in pulmonary fibrosis patients, organized through the main analysis notebook (`main.ipynb`). The workflow follows this structure:

### 1. Exploratory Data Analysis (EDA)

**Tabular Data Analysis:**
- Visualization of FVC decline trajectories over time across patients
- Investigation of demographic factors (Age, Sex, SmokingStatus) and their relationship with disease progression
- Statistical analysis showing:
  - Overall declining trend in percent-predicted FVC over weeks
  - Sex-based differences in baseline FVC (males > females) but similar decline rates
  - Smoking status correlation with FVC variability

**Key Findings:**
- Disease progression shows high inter-patient variability
- FVC decline is not strongly correlated with age alone
- Male patients exhibit higher baseline FVC but similar decline rates to females
- Ex-smokers show wider distribution with more extreme low values

### 2. Baseline Unimodal Models

#### Tabular Data Models:

**XGBoost Baseline:**
- Collapses longitudinal data into single-row representations
- Computes engineered features: Δt, ΔFVC, next_time
- Assumes linear progression between measurements
- Provides quick baseline performance metrics

**Gaussian Process Regression:**
- Models temporal correlation and inter-patient covariance
- Utilizes patient ID embeddings for individual trajectories
- Incorporates mixed kernels for time-series (Matérn) and demographic features (RBF, Index)
- Trained with 5 random restarts for optimization stability
- Provides probabilistic predictions with uncertainty quantification (95% confidence intervals)

#### Image Data Models:

**CNN (Convolutional Neural Network):**
- Processes DICOM CT scan slices
- Extracts spatial features from lung images
- Combined with fully connected layers for FVC prediction
- *(Implementation details in `models/cnn.py`)*

5 Layer CNN + 3 Layer FC to produce prediction.
Previous work on problem utilized pretrained EfficietNet models, but used plain CNN layers to focus on improving preprocessing and training logic for extracting lung features.

### 3. Multimodal Fusion Approaches

#### Approach 1: Late Fusion
Combines predictions from independently trained unimodal models:

**Architecture:**
- **Tabular Branch**: Gaussian Process regression on demographic + temporal data
- **Image Branch**: CNN on CT scan slices
- **Fusion**: Linear regression combines both model predictions with learned weights
- **Output**: Final FVC prediction with optimized combination

**Advantages:**
- Leverages complementary information from both modalities
- Each modality can be optimized independently
- Simple fusion mechanism allows interpretable weighting

**Process:**
1. Train GP model on tabular features
2. Train CNN model on CT scan images
3. Stack predictions from both models
4. Learn optimal linear combination weights
5. Evaluate on validation set with MAE and R² metrics

#### Approach 2: Early Fusion
Integrates features at the representation level before prediction:

**Architecture:**
- **Image Encoder**: ResNet3D CNN for 3D CT volume feature extraction
- **Tabular Encoder**: LSTM for time-series demographic feature extraction
- **Fusion Module**: RNN-based fusion combining both feature representations
- **Output**: Joint prediction with uncertainty (mean ± 1.96σ for 95% CI)

**Advantages:**
- Joint feature learning allows cross-modal interactions
- Better captures relationships between imaging and clinical data
- Provides probabilistic predictions with confidence intervals

**Process:**
1. Extract deep features from CT scans using ResNet3D
2. Extract temporal features from patient data using LSTM
3. Concatenate feature representations
4. Feed into RNN fusion model for final prediction
5. Evaluate with MAE, R², and Laplace Log-Likelihood

### 4. Evaluation Metrics

**Performance Measures:**
- **MAE (Mean Absolute Error)**: Average prediction error in ml
- **R² Score**: Proportion of variance explained by the model
- **RMSE (Root Mean Squared Error)**: Sensitivity to large errors
- **Laplace Log-Likelihood**: Probabilistic model quality (for GP and Early Fusion)

**Visualization:**
- Predicted vs. Actual scatter plots with perfect prediction line
- Residual distribution histograms
- Per-patient trajectory plots with confidence intervals
- Model comparison bar charts

## Project Structure

```
pulmonary_fibrosis/
├── data/                          # Dataset files
│   ├── train.csv                  # Patient demographics & FVC measurements
│   ├── test.csv                   # Test set
│   └── sample_submission.csv      # Submission format
├── preprocessing/                 # Data preprocessing modules
│   ├── tabular_preprocessing.py   # Tabular data preprocessing
│   ├── image_preprocessing.py     # DICOM image preprocessing
│   ├── pid_split.py              # Patient ID train/val split
│   └── scan/                      # CT scan preprocessing utilities
├── models/                        # Model implementations
│   ├── baseline.py               # XGBoost baseline model
│   ├── gaussian_process.py       # Gaussian Process regression
│   ├── cnn.py                    # CNN for image data
│   ├── joint_fusion_process.py   # Early fusion (ResNet3D + LSTM + RNN)
│   └── checkpoints/              # Saved model weights
├── evaluation/                    # Evaluation utilities
│   └── eval_template.py          # Metrics calculation functions
├── Tabular_EDA/                   # Exploratory analysis scripts
├── main.ipynb                     # Main analysis notebook
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/unfulw/pulmonary_fibrosis.git
cd pulmonary_fibrosis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from [OSIC Pulmonary Fibrosis Progression](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression) and place in the `data/` directory.

## Usage

**Run the main analysis:**
```bash
jupyter notebook main.ipynb
```

The notebook walks through:
1. Data loading and EDA
2. Baseline model training (XGBoost, Gaussian Process)
3. CNN model evaluation
4. Late fusion implementation
5. Early fusion evaluation
6. Model comparison and visualization

**Train individual models:**
- XGBoost: `python models/baseline.py`
- Gaussian Process: `python models/gaussian_process.py`
- CNN: `python models/cnn.py`
- Early Fusion: `python models/joint_fusion_process.py`

## Results Summary

*(To be completed with final model performance metrics)*

**Baseline Models:**
- XGBoost: MAE = [TBD] ml, R² = [TBD]
- Gaussian Process: MAE = [TBD] ml, R² = [TBD]
- CNN: MAE = [TBD] ml, R² = [TBD]

**Fusion Models:**
- Late Fusion (GP + CNN): MAE = [TBD] ml, R² = [TBD]
- Early Fusion (ResNet3D + LSTM + RNN): MAE = [TBD] ml, R² = [TBD]

## Future Work

- Complete comprehensive model comparison across all approaches
- Implement attention mechanisms for better feature fusion
- Explore transformer-based architectures for temporal modeling
- Add cross-validation for robust performance estimation
- Optimize hyperparameters for each model
- Implement ensemble methods combining multiple fusion strategies
- Deploy model for clinical trial testing

## Contributors

- Kim Minjun
- Nam Sangjun
- Jeon Jinkyung
- Angel Bu Tong Mei
- Keven Wong
- Kaveri Patil

## License

This project is for academic purposes as part of CS3244 project.

## Acknowledgments / Citations

- Open Source Image Consortium (OSIC) for providing the dataset
- CS3244 course staff for guidance and support
- Research Paper on Significance of CT feature for fibrosis longevity. 
  https://journal.chestnet.org/article/S0012-3692(15)46366-4/fulltext
- Understanding CT of honeycombed lungs
  https://radiopaedia.org/articles/honeycombing-lungs
- Approach from Kaggle Competition winner for inspiration
  https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression/writeups/art-1st-place-mostly-unpredictable-solution

---

*Note: This README is a work in progress and will be updated as the project advances.*
