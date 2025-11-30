# Baseline CT - Augmented FVC Forecasting to Project Pulmonary Fibrosis Progression, Aiding Medical Decision-Making and Prognosis

## Dataset Description

The dataset contains baseline lung CT scans in DICOM (.dcm) format (which includes grayscale pixel data and metadata), anonymized patient demographics, and weekly forced vital capacity (FVC) measurements of pulmonary fibrosis patients—a crucial lung function indicator.

**Data Source:** Open Source Imaging Consortium (OSIC) Pulmonary Fibrosis Progression - Kaggle Competition (2020)

The OSIC is a "not-for-profit cooperative effort between academia, industry and patient advocacy groups" that focuses on respiratory disease, providing highly reliable data from experts and leading organizations including George Mason University, MedQIA, Harvard Medical School, and Boehringer Ingelheim.

**Data Components:**
- **DICOM CT Scans**: Baseline lung CT images stored as .dcm files containing grayscale pixel data and metadata
- **Tabular Data**: Patient demographics (Age, Sex, SmokingStatus) and weekly FVC measurements  
- **Target Variable**: FVC (Forced Vital Capacity) in milliliters
- **Temporal Features**: Weeks relative to baseline CT scan
- **Dataset Size**: 176 training patients with longitudinal measurements

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

**Image Data Preprocessing:**
- Conversion of DICOM files to Hounsfield Units (HU) for standardization
- Lung windowing (center=-600, width=1500) to focus on lung tissue
- Morphological operations for lung region isolation
- Resampling and normalization for consistent input dimensions

### 2. Baseline Unimodal Models

#### Tabular Data Models:

**XGBoost Baseline:**
- Collapses longitudinal data into single-row representations
- Computes engineered features: Δt, ΔFVC, next_time
- Assumes linear progression between measurements
- **Performance**: MAE = 524.07 ml, R² = 0.238

**GRU-T (Gated Recurrent Unit - Time-aware):**
- Handles variable-length sequential patient data
- Processes temporal dependencies in FVC measurements
- Encodes demographics and baseline features
- **Performance**: MAE = 143.41 ml, R² = 0.894, RMSE = 213.89 ml

**Gaussian Process Regression (M2 with Patient Embeddings):**
- Models temporal correlation and inter-patient covariance using mixed kernels (Matérn, RBF, Index)
- Utilizes patient ID embeddings (dimension=4) for individual trajectories
- Captures population-level trends while accounting for patient-specific effects
- Trained with 5 random restarts for optimization stability
- Provides probabilistic predictions with uncertainty quantification (95% confidence intervals)

#### Image Data Models:

**Manual Feature Extraction**
- Makes uses of traditional CV filters to extract features from preprocessed scans
1. Edge Detection (Sobel)
2. Corner Detection (Harris Corner Detection)
3. Blob Detection (DoG)
4. Local Binary Pattern
5. Gabor Filter

Primary axis is analyzed for each feature using PCA
With reduced dimensions, RandomForestRegressor of each feature makes prediction

**CNN (Convolutional Neural Network):**
- Processes preprocessed scans
- Extracts spatial features from lung images
- Project tabular features to image feature's dimensions
- Concatenated and fed into fully connected layers for FVC prediction
- *(Implementation details in `models/cnn.py`)*

5 Layer CNN + 3 Layer FC to produce prediction.
Previous work on problem utilized pretrained EfficietNet models, but used plain CNN layers to focus on improving preprocessing and training logic for extracting lung features.
Model Architecture: TD-CNN-BiLSTM

**Hybrid Spatiotemporal Regression Network** 

Spatial Feature Extraction (Time-Distributed CNN):
- Applies a custom CNN to each CT slice independently.
- Uses GroupNormalization (Groups=4) instead of BatchNormalization. This stabilizes training dynamics when using a Batch Size of 1 (required for memory constraints).
- Uses L2 Regularization to penalize complex weights and prevent overfitting.

Temporal Aggregation (Bidirectional LSTM):
- A Bidirectional LSTM processes the sequence of slice features from Top-to-Bottom and Bottom-to-Top (lower 55% area).
- This captures the 3D structural progression of the disease across the lung's depth.

Regression Head:
- A Dense layer with Huber Loss combines the spatiotemporal features to predict the final Slope (ml/week).

##### Other Approaches to Vision Analysis
All approaches to improve the model is implemented in 'models/cnn.ipynb':

1. Baseline Simple CNN (5 Convolutional Layers)
Implementation: Basic CNN with 5 conv layers (32→64→128→256→512 channels), BatchNorm, MaxPooling, and Global Average Pooling

Architecture: Progressive channel expansion with 2x2 pooling after each conv block
FC Layer: Takes averaged scan features + tabular data (weeks, initial_FVC, initial_weeks)

Objective: Establish baseline performance for CT scan feature extraction combined with tabular temporal data

Key Features:
Per-patient feature aggregation via mean pooling across slices
Tabular feature expansion from 3→30 dimensions


2. Flattened Features vs Global Pooling

Implementation: Changed from averaging features across slices to flattening all slice features

Architecture Change: Minimize information loss using features.flatten() instead of torch.mean(features, dim=0)
FC Input Dimension: 512 * num_slices instead of 512

Objective: Test if preserving per-slice information improves predictions vs aggregated representation

Additional Changes:
Switched from BatchNorm to GroupNorm
Added dropout layers (0.2-0.3) for regularization
Used AdamW optimizer

Result: Similar performance between two methods, finding mean over slices doesn't lead to significant information loss.

3. ImageNet-Inspired Deep Architecture (ResNet34-like)

Implementation: Deeper network with repeated conv blocks mimicking ImageNet architectures

Architecture:
Conv1: 7x7 kernel, stride 2, followed by 3x3 maxpool
Conv2-5: Multiple 3x3 convolutions with GroupNorm
6 blocks in conv2, 11 in conv3, 5 in conv4, 2 in conv5

Objective: Test if deeper architecture captures more complex lung tissue patterns

Result: Had hard time converging, potentially extracting noise irrelevant to prediction


4. Random Scan Selection Within Partitions

Intent: Instead of deterministic scan selection, randomly sample within each partition to make full use of the available scans in the dataset. 

Implementation: Divide the available scans into partitions, randomly sample a scan from each partition by scan_index = random.randint(curr, next_index-1) within each partition

Objective: Increase training data diversity and prevent overfitting to specific slice positions

Benefit: Better utilization of all available CT scan slices during training

Result: Non-deterministic aspect hindered model's training progress, doesn't seem to help with overfitting.

5. Patient-Agnostic Training Loop
Implementation: Changed from patient-batched training to individual datapoint shuffling

Training Strategy:
Create flat list of (patient_id, x, y) tuples
Shuffle all training points across patients
Gradient accumulation every 4 steps

Objective: Improve generalization by breaking patient-specific correlations. During gradient accumulation, provide model with samples from different patient to make a more accurate estimation of the dataset-wide gradient. 

Downside: Repeatedly loads same patient scans (inefficient)


6. ResNet34 with Residual Connections

Implementation: Added skip connections (residual blocks) to enable training deeper networks

Architecture:
ResidualBlock: two 3x3 convs with identity shortcut when stride=1
3, 4, 6, 3 blocks in conv2-5 respectively
GroupNorm for normalization

Objective: Address vanishing gradient problem and enable learning of residual functions

Result: Did not meaningfully improve model's performance, meaning information lost during the long CNN layers is not the central issue.


7. Bottleneck Blocks (ResNet-101/152 Architecture)
Implementation: Deeper architecture using 1x1→3x3→1x1 bottleneck blocks

Architecture:
3 blocks in conv2, 4 in conv3, 23 in conv4, 3 in conv5 (ResNet-101 config)
Bottleneck reduces computation: 256→64→64→256 instead of 256→256→256

Output channels: 4x the intermediate channels

Objective: Create much deeper model (101-152 layers) while maintaining computational efficiency

Tradeoff: Higher capacity but longer training time (6s/it vs 1-2s/it)

Result: Deeper network doesn't improve the model, signaling issues with data input


8. Top/Bottom Lung Partition Analysis

Implementation: Divide scans into spatial partitions (top/bottom half of lungs), and train two separate models specializing on each halves of the lung.

Implementation:
Extract features from each partition separately
Concatenate partition features before FC layers
Use dedicated model to analyze top/bottom half

Objective: Test hypothesis that different lung regions show different fibrosis patterns

Result: No meaningful improvement in performance, much longer training time. Potentially due to available dataset reducing into half, not enough to train two models fully.

9. Loss Function Experiments

MSE Loss: Standard squared error - sensitive to outliers

Huber Loss (delta=1.0): Combines MSE (small errors) and MAE (large errors)

More robust to outliers

Objective: Improve robustness to extreme FVC values and measurement noise


10. Overfitting Capacity Test
Implementation: Single-patient overfitting test to verify model has sufficient capacity. Train on one patient for 1000 epochs with detailed diagnostics

Metrics Tracked:
Per-sample loss progression
Gradient norms (CNN vs FC)
Feature statistics (mean, std)

Objective: Diagnose if poor performance is due to insufficient model capacity vs other issues

Findings: Model struggles to overfit even single patient. This means each datapoint that belongs under the same patient is so noisy, that the model is having a hard time optimizing function that predicts all points nicely. 

### 3. Multimodal Fusion Approaches

#### Approach 1: Late Fusion
#### Approach 1: Late Fusion (GP + CNN)
Combines predictions from independently trained unimodal models:

**Architecture:**
- **Tabular Branch**: Gaussian Process (M2) regression on demographic + temporal data
- **Image Branch**: CNN on CT scan slices with FC layers
- **Fusion**: Linear regression combines both model predictions with learned weights
- **Output**: Final FVC prediction with optimized combination

**Advantages:**
- Leverages complementary information from both modalities
- Each modality can be optimized independently
- Simple fusion mechanism allows interpretable weighting

**Performance:**
- **Late Fusion**: MAE = 138.99 ml, R² = 0.966
- GP-only: MAE = 146.37 ml, R² = 0.961
- CNN-only: MAE = 174.07 ml, R² = 0.947
- **Result**: Late fusion improves upon individual models by optimally weighting their predictions

**Process:**
1. Train GP model on tabular features
2. Train CNN model on CT scan images
3. Stack predictions from both models
4. Learn optimal linear combination weights (GP weight: 0.7663, CNN weight: 0.2336)
5. Evaluate on validation set with MAE and R² metrics

#### Approach 2: Early Fusion (ResNet3D + LSTM + RNN)
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

**Performance:**
- **Early Fusion**: MAE = 256.17 ml, R² = 0.870, Laplace Log-Likelihood = -1071.39

**Analysis:**
The early fusion approach underperformed compared to late fusion, suggesting that direct joint feature learning may face challenges with:
- Limited image diversity in the dataset
- Difficulty in learning effective cross-modal representations
- Potential overfitting with the more complex architecture

**Process:**
1. Extract deep features from CT scans using ResNet3D
2. Extract temporal features from patient data using LSTM
3. Concatenate feature representations
4. Feed into RNN fusion model for final prediction
5. Evaluate with MAE, R², and Laplace Log-Likelihoodhic feature extraction
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
## Project Structure

```
pulmonary_fibrosis/
├── data/                              # Dataset files
│   ├── train.csv                      # Patient demographics & FVC measurements
│   ├── test.csv                       # Test set
│   └── sample_submission.csv          # Submission format
├── preprocessing/                     # Data preprocessing modules
│   ├── tabular_preprocessing.py       # Tabular data preprocessing & train/val split
│   ├── image_preprocessing.py         # DICOM image preprocessing
│   ├── pid_split.py                   # Patient ID train/val split
│   ├── preprocess.py                  # General preprocessing utilities
│   ├── EDA_scans_processing.py        # Scan exploration utilities
│   └── scan/                          # CT scan preprocessing utilities
│       ├── preprocess.py              # DICOM to HU conversion, windowing, masking
│       ├── cnn.ipynb                  # Scan preprocessing exploration
│       └── dicom.ipynb                # DICOM format exploration
├── models/                            # Model implementations
│   ├── baseline.py                    # XGBoost baseline model
│   ├── gaussian_process.py            # Gaussian Process regression (M2)
│   ├── gaussian_process.ipynb         # GP development notebook
│   ├── gru.ipynb                      # GRU model development
│   ├── gru_t.ipynb                    # GRU temporal variant
│   ├── cnn.py                         # CNN for image data
│   ├── cnn.ipynb                      # CNN development notebook
│   ├── joint_fusion_process.py        # Early fusion (ResNet3D + LSTM + RNN)
│   ├── joint_fusion_process.ipynb     # Early fusion development
│   ├── manual_feat_ext.ipynb          # Manual feature extraction experiments
│   ├── final_jointFusion_model.pth    # Trained early fusion model weights
│   └── checkpoints/                   # Saved model checkpoints
│       └── cnn_model_*.pth            # CNN model checkpoints
├── evaluation/                        # Evaluation utilities
│   └── eval_template.py               # Metrics calculation (MAE, R², LLL)
├── Tabular_EDA/                       # Exploratory analysis
│   ├── Tabular_EDA.ipynb              # Comprehensive EDA notebook
│   ├── dicom_csv_check.py             # DICOM metadata validation
│   ├── LinearRegression.py            # Simple baseline experiments
│   └── dicom_parsed_sample*.csv       # Sample parsed DICOM data
├── scan/                              # Additional scan utilities
│   └── cnn.ipynb                      # Scan processing experiments
├── documents/                         # Project documentation
├── main.ipynb                         # **Main analysis notebook**
├── requirements.txt                   # Python dependencies
├── __init__.py                        # Package initialization
└── README.md                          # This file
``` ├── image_preprocessing.py     # DICOM image preprocessing
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
**Train individual models:**
```bash
python models/baseline.py              # XGBoost
python models/gaussian_process.py      # Gaussian Process
python models/cnn.py                   # CNN
python models/joint_fusion_process.py  # Early Fusion
```

## Results Summary

### Model Performance Comparison

| Model | MAE (ml) | R² | RMSE (ml) | Notes |
|-------|----------|-----|-----------|-------|
| **XGBoost Baseline** | 524.07 | 0.238 | 653.17 | Simple baseline with row coercion |
| **GRU-T** | 143.41 | 0.894 | 213.89 | Best tabular-only model |
| **Gaussian Process (M2)** | 0.1714* | 0.927* | - | *On scaled values; includes uncertainty |
| **CNN** | 174.07 | 0.947 | - | Evaluated within late fusion |
| **Late Fusion (GP+CNN)** | **138.99** | **0.966** | - | **Best overall performance** |
| **Early Fusion (ResNet3D+LSTM)** | 256.17 | 0.870 | - | Joint learning approach |

*Best performer: **Late Fusion** achieves lowest MAE and highest R²*

### Key Insights

**1. Late Fusion Success:**
- Achieves 5.0% improvement over GP-only (146.37 → 138.99 ml MAE)
- Optimal weighting: 76.6% GP, 23.4% CNN
- Demonstrates complementary value of imaging and clinical data

**2. Tabular Models Strong Performance:**
- GRU-T and GP both achieve R² > 0.89, showing effectiveness on longitudinal clinical data
- GP provides valuable uncertainty quantification with 95% confidence intervals

**3. Early Fusion Challenges:**
- Higher MAE (256.17 ml) suggests difficulty in joint representation learning
- May require larger dataset or different architecture choices
- Trade-off between accuracy and uncertainty calibration observed

**4. Baseline Comparison:**
- XGBoost's poor performance (MAE = 524.07 ml) validates need for specialized temporal modeling
- Simple row coercion loses critical longitudinal information

## Future Work

### Model Improvements
- Implement attention mechanisms for better feature fusion in early fusion architecture
- Explore transformer-based architectures for temporal modeling
- Add cross-validation for robust performance estimation across multiple folds
- Optimize hyperparameters using systematic grid/random search
- Investigate intermediate fusion strategies between early and late approaches

### Data and Features
- Incorporate additional clinical biomarkers if available
- Experiment with different image preprocessing techniques (e.g., different windowing)
- Explore data augmentation for CT scans to improve robustness
- Analyze feature importance and attention weights for interpretability

### Deployment and Clinical Application
- Develop uncertainty-aware prediction intervals for clinical decision support
- Validate on external datasets for generalizability assessment
- Create user-friendly interface for clinical deployment
- Conduct prospective validation in clinical settings

## References

1. **Artkulak. (2020).** GitHub - artkulak/osic-pulmonary-fibrosis-progression: 1st place solution of OSIC Pulmonary Fibrosis Progression competition at Kaggle. GitHub. https://github.com/artkulak/osic-pulmonary-fibrosis-progression

2. **Fang, C., Bai, S., Chen, Q., Zhou, Y., Xia, L., Qin, L., Gong, S., Xie, X., Zhou, C., Tu, D., Zhang, C., Liu, X., Chen, W., Bai, X., & Torr, P. H. (2021).** Deep learning for predicting COVID-19 malignant progression. Medical Image Analysis, 72, 102096. https://doi.org/10.1016/j.media.2021.102096

3. **Glotov, A., & Lyakhov, P. (2021).** Pulmonary fibrosis progression prognosis using Machine Learning. 2021 Ural Symposium on Biomedical Engineering, Radioelectronics and Information Technology (USBEREIT), 0327–0329. https://doi.org/10.1109/usbereit51232.2021.9455070

4. **Liu, N., Gou, S., Gao, R., Su, B., Liu, W., Park, C. K. S., Xing, S., Yuan, J., & Fenster, A. (2025).** GRU-TV: Time- and velocity-aware gated recurrent unit for patient representation. Journal of Biomedical Informatics. Advance online publication. https://doi.org/10.1016/j.jbi.2025.104667

5. **Open Source Imaging Consortium (OSIC). (2020).** "OSIC Pulmonary Fibrosis Progression." *Kaggle Competition*. https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression

6. **Schulam, P., & Saria, S. (2015).** "A Framework for Individualizing Predictions of Disease Trajectories by Exploiting Multi-Resolution Structure." *Advances in Neural Information Processing Systems*, 28, 748-756. [arXiv:1511.08950](https://arxiv.org/abs/1511.08950)

## Contributors

**Team Pg34 - CS3244 Machine Learning Project:**
- Kim Minjun
- Nam Sangjun
- Jeon Jinkyung
- Angel Bu Tong Mei
- Keven Wong
- Kaveri Patil

## License

This project is for academic purposes as part of CS3244 Machine Learning coursework at the National University of Singapore.

## Acknowledgments / Citations

- Open Source Image Consortium (OSIC) for providing the dataset
- CS3244 course staff for guidance and support
- Research Paper on Significance of CT feature for fibrosis longevity. 
  https://journal.chestnet.org/article/S0012-3692(15)46366-4/fulltext
- Understanding CT of honeycombed lungs
  https://radiopaedia.org/articles/honeycombing-lungs
- Approach from Kaggle Competition winner for inspiration
  https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression/writeups/art-1st-place-mostly-unpredictable-solution
- Open Source Imaging Consortium (OSIC) for providing the high-quality dataset
- CS3244 course staff for guidance and support throughout the project
- National University of Singapore, School of Computing

---

**Project Status:** Completed (November 2025)  
**Contact:** For questions or collaborations, please reach out through the course instructors.
