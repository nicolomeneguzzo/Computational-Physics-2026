# Astrophysical Object Classification using Machine Learning

Automatic classification of Stars, Galaxies, and Quasars using Machine Learning algorithms applied to photometric data from the Sloan Digital Sky Survey (SDSS DR17).

## Overview

Contemporary astronomy has entered a data-driven era. Large-scale surveys like SDSS collect hundreds of terabytes of data, making human visual inspection of astronomical sources entirely impractical. 

This project aims to develop a robust Machine Learning framework capable of accurately distinguishing between three major classes of astronomical objects: **Stars, Galaxies, and Quasars (QSOs)**.

To make the classification problem more challenging and astrophysically meaningful, the **redshift feature was intentionally excluded** from the analysis. Since redshift is strongly correlated with object distance and type, its inclusion would artificially simplify the classification task. Instead, our models rely strictly on the photometric information typically available in any broad-band survey:
- **Broadband magnitudes:** $u, g, r, i, z$
- **Derived color indices:** $u-g, g-r, r-i, i-z$

### Project Goals

1. **Systematic Model Comparison:** Evaluate and compare tree-based ensemble methods (Random Forest, Extra Trees, XGBoost, LightGBM), ensemble strategies (Voting, Stacking), and fully connected Feed-Forward Neural Networks.

2. **Feature Interpretability:** Identify the physical drivers behind the classification process through Feature Importance and Feature Ablation analyses.

3. **Astrophysical Insights:** Link the models' behavior and misclassifications to the actual physical properties of the objects, identifying photometric regimes where the distinction between classes becomes intrinsically degenerate.

## Dataset

The data used for this project is extracted from the **Sloan Digital Sky Survey (SDSS) Data Release 17 (DR17)**. The initial dataset contains 100,000 labeled astronomical sources described by 22 attributes.

## Dataset Features

The dataset contains 100,000 astronomical objects described by 22 attributes, including positional, photometric, spectroscopic, and engineered features.

### Target Variable

- **`class`**: The object category to be predicted (Star, Galaxy, or Quasar).


### Positional Features

- **alpha** – Right Ascension (RA) in the J2000 coordinate system.

- **delta** – Declination (DEC) in the J2000 coordinate system.

### Photometric Features

- **u** – Ultraviolet band magnitude.

- **g** – Green band magnitude.

- **r** – Red band magnitude.

- **i** – Near-infrared band magnitude.

- **z** – Infrared band magnitude.

### Spectroscopic Features

- **redshift** – Measured redshift of the object.

### Survey and Observation Metadata

- **run_ID** – Observation run identifier.

- **rerun_ID** – Data processing rerun identifier.

- **cam_col** – Camera column identifier.

- **field_ID** – Field identifier within the survey.

- **plate** – Spectroscopic plate identifier.

- **MJD** – Modified Julian Date of observation.

- **fiber_ID** – Fiber identifier used for spectroscopy.

### Identifiers

- **obj_ID** – Unique object identifier.

- **spec_obj_ID** – Unique spectroscopic object identifier.

### Engineered Features (Color Indices)
- **u_g** = u − g
- **g_r** = g − r
- **r_i** = r − i
- **i_z** = i − z


### Features Used for Training

To force the models to learn the intrinsic Spectral Energy Distribution (SED) shape rather than relying on distance metrics or coordinates, only the following 9 features were retained:

- **Photometric Magnitudes:** `u`, `g`, `r`, `i`, `z`

- **Engineered Color Indices:** `u_g`, `g_r`, `r_i`, `i_z`

### Dropped Features

The following features were intentionally excluded from the training process:
- **Trivial Predictors:** `redshift` (excluded to evaluate purely photometric classification capabilities).

- **Positional Coordinates:** `alpha` (Right Ascension), `delta` (Declination).

- **Survey Metadata & Identifiers:** `obj_ID`, `spec_obj_ID`, `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `plate`, `MJD`, `fiber_ID`.


## Methodology


### Data Preprocessing

1. **Exploratory Data Analysis (EDA) & Data Cleaning:** 

    - Initial examination of feature distributions and physical relevance.

   - Removal of outliers and unphysical photometric measurements to ensure a clean signal.
   
   - Exclusion of non-informative metadata and the `redshift` feature.
   
2. **Data Splitting:**  The dataset was divided into **Training (60%)**, **Validation (20%)**, and **Test (20%)** subsets to allow for unbiased model evaluation and hyperparameter tuning.

3. **Feature Scaling:** Standardization (Z-score normalization) was applied to all numerical features to ensure that scale-sensitive models (like Neural Networks) converge optimally.

4. **Class Balancing:** The **Synthetic Minority Oversampling Technique (SMOTE)** was applied *exclusively* to the training set to address class imbalance, ensuring that the models do not become biased toward the majority classes (Galaxies and Stars) at the expense of the minority class (Quasars).


### Machine Learning Models

A systematic comparison of different machine learning paradigms was conducted to identify the best-performing architecture for photometric classification. The following models were trained and evaluated:

### Tree-Based Ensemble Methods

- **Random Forest (RF)**

- **Extra Trees (ERT)**

- **XGBoost** (Extreme Gradient Boosting)

- **LightGBM** (Light Gradient Boosting Machine)

### Ensemble Strategies

- **Hard Voting Classifier** (aggregating predictions from the tree-based models)

- **Stacking Classifier** (using a meta-model to combine base model predictions)

### Deep Learning

- **Fully Connected Feed-Forward Neural Network (NN)** (investigating the capacity of deep learning to capture non-linear multidimensional correlations without orthogonal decision boundaries).

Hyperparameter optimization was performed for tree-based models using Randomized Search and Cross-Validation.

## Repository Structure


```text
astrobject_classification/
│
├── data/
│   ├── __init__.py
│   └── preprocessing.py          # Data cleaning, outlier removal, train/val/test splits
│
├── experiments/
│   ├── feature_ablation.py       # Feature ablation for NN
│   ├── nn_runner.py              # Hyperparameters tuning for NN
│   └── pca_ablation.py           # Dimensionality reduction and PCA experiments
│  
├── inference/
│   ├── __init__.py
│   └── predictor.py              # Model evaluation, predictions generation, and testing
│
├── models/
│   ├── __init__.py
│   ├── network.py                # Base Neural Network architecture definition
│   ├── trees.py                  # Tree-based model configurations (XGBoost, RF)
│   └── nn_variants.py            # Variations of NN architectures 
│
├── training/
│   └── __init__.py               # Training modules and utilities
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py                # Performance metrics calculation (F1, precision, recall)
│   ├── seeding.py                # Global random seed configuration for reproducibility for nn 
│   └── wrapped.py                # Adaptor PyTorch → scikit-learn for using utility sklearn.
│
├── visualization/
│   ├── __init__.py
│   ├── model_selection.py        # ML plots (Confusion Matrix, Feature Importance, Learning Curves)
│   └── nn_plots.py               # NN plots 
│
├── trainer.py                    # Main pipeline to train models 
├── requirements.txt              # Project dependencies (libraries and versions)
└── README.md                     
```

## Installation

Clone the repository:

```bash
git clone https://github.com/nicolomeneguzzo/Computational-Physics-2026/tree/main/Astrophysical-Objects-Recognition/astrobject_classification
cd astrobject_classification

pip install -r requirements.txt
```

## Results

The models were evaluated on the held-out Test set. Since the original dataset exhibited class imbalance, the **Macro F1-score** was established as the primary evaluation metric alongside Accuracy, Precision, and Recall.

| Model | Test Accuracy (%) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | **89.18** | **0.8598** | **0.8692** | **0.8641** |
| **Neural Network** | 89.22 | 0.8613 | 0.8653 | 0.8632 |
| **Voting Classifier** | 89.03 | 0.8592 | 0.8649 | 0.8614 |
| **LightGBM** | 88.89 | 0.8555 | 0.8678 | 0.8612 |
| **Extra Trees** | 88.95 | 0.8592 | 0.8633 | 0.8603 |
| **Stacking Classifier** | 88.95 | 0.8574 | 0.8625 | 0.8597 |
| **Random Forest** | 88.79 | 0.8553 | 0.8623 | 0.8584 |

*Note: XGBoost achieved the highest F1-score, striking the best balance between precision and recall, and demonstrating excellent generalization with minimal overfitting compared to deeper tree ensembles.*

### Visual Diagnostics 

- **Confusion Matrix:** `results_plot/confusion_matrix_xgboost.png`

- **Feature Importance:** `results_plot/feature_importance_comparison.png`

- **Learning Curve:** `results_plot/learning_curve_xgboost.png`

- **Stacked Error histogram:** `results_plot/stacked_error_histogram_xgboost.png`



## Conclusions

1. **Model Performance:** Both **XGBoost** and the **Neural Network** successfully solved the photometric classification task, achieving F1-scores above 0.86. XGBoost was selected as the primary model due to its slight edge in F1-score and its intrinsic tree-based interpretability.

2. **The Power of Color Indices:** Feature importance analysis revealed that derived color indices (specifically **$r-i$** and **$u-g$**) overwhelmingly dominate the classification process. 

3. **The Redshift/Distance Degeneracy:** The models correctly learned to ignore raw photometric magnitudes ($u, g, r, i, z$). Without redshift information, apparent magnitudes are heavily degenerate. By relying entirely on colors, the algorithms effectively learned to classify objects based on the intrinsic shape of their Spectral Energy Distribution (SED).

4. **Information Redundancy:** Feature ablation studies demonstrated a strict performance plateau after introducing the top 5 features. Adding raw magnitudes introduced zero discriminative power for tree-based models and slight statistical noise for the Neural Network.



## References
- [Sloan Digital Sky Survey (SDSS DR17)](https://www.sdss.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-Learn Machine Learning in Python](https://scikit-learn.org/)