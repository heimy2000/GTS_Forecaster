
# GTS_Forecaster User Manual V1.0

## 1. Introduction

GTS_Forecaster is a geodetic time series preprocessing predicting and toolbox developed in Python that supports both Windows and Ubuntu operating systems to implement functional modules.

GTS_Forecaster is dedicated to time series interpolation and forecasting with deep learning algorithms on geodetic time series (e.g., GNSS displacements, Tide gauge sea level time series and satellite altimetry sea surface height time series). The main features of GTS_Forecaster incorporates:

### 1.1 Overview of System Architecture

#### 1.1.1 Technical Architecture Diagram

**Dual-Mode Collaborative Workflow**

**Core Engine**

**Data Preprocessing**:

- Sliding window generation

- Normalization (Min-Max scaling)

- Missing value interpolation (linear/spline)

**Deep Learning Models**:

- LSTM, BiLSTM, GRU, TCN, Transformer, TimeGNN, etc. (17+ variants)

- Custom architectures (e.g., ConvLSTM, Attention-enhanced models)

**Multi-Mode Support**:

- CLI for batch prediction

- GUI for interactive operation

**Result Output**:

- Visualization: Time series prediction plots

- Structured data: CSV/Parquet files

- Model metrics: RMSE, MAE, R² in model_scores.csv

#### 1.1.2 Interactive Frontend

**Tkinter GUI Framework**:

- Parameter configuration panel

- Real-time training monitoring (loss curves, metrics)

- Result visualization dashboard

**CLI Integration**:

- Converts GUI inputs to command-line arguments

- Executes core engine via subprocess calls

### 1.2 Data Specifications

#### 1.2.1 Input File Format

File Type: CSV (UTF-8)

**Mandatory Columns**:

| Column Name | Type   | Description                     | Example Value |
|-------------|--------|---------------------------------|---------------|
| time        | float  | MJD-format timestamp            | 60325.5       |
| value       | float  | Primary observation (e.g., elevation) | 58.2          |

**Optional Columns**: Auxiliary numerical features (e.g., temperature, pressure)

Preprocessing Requirement: Multi-dimensional convolution for feature fusion (enabled via `--enable_md_conv` flag).

#### 1.2.2 Sample Document (csv)

```
time,value,temperature,pressure  
60325.5,12.345,25.3,1012.7  
60325.504,12.356,25.1,1012.5  
```

## 2. User Manual (Command Line Version)

### Basic Parameters

| Parameter          | Required | Description                     | Example Value |
|--------------------|----------|---------------------------------|---------------|
| `--path`           | ✅       | Data file path                  | data/gnss.csv |
| `--input_features` | ✅       | Input feature columns (comma-separated) | time,value,temp |
| `--output_features`| ✅       | Output target columns (comma-separated) | value |
| `--model_name`     | ✅       | Model name (case-sensitive)    | TimeGNN       |
| `--window_size`    | ✅       | Sliding window size (≥10)      | 30            |

### Training Control

| Parameter              | Required | Description                     | Example Value |
|------------------------|----------|---------------------------------|---------------|
| `--batch_size`         | ✅       | Training batch size             | 32            |
| `--num_epochs`         | ✅       | Training epochs (≥100)          | 10000         |
| `--lr`                 | ✅       | Learning rate (suggested range: 1e-5~1e-3) | 0.001 |
| `--use_early_stopping` | ❌       | Enable early stopping mechanism | No value needed, add flag directly |

### Prediction Control

| Parameter          | Required | Description                     | Example Value |
|--------------------|----------|---------------------------------|---------------|
| `--predict_start`  | ❌       | Custom prediction start index (≥ window size) | 400 |
| `--predict_end`    | ❌       | Custom prediction end index (≤ dataset length) | 1655 |

### Model Parameter Reference Table

| Algorithms          | Prediction Models | Features of Available models                                                                 |
|---------------------|-------------------|-----------------------------------------------------------------------------------------------|
| **Prototype Algorithms** | LSTM             | Solves long-term dependency issues via gating mechanisms. High computational complexity and low parallelism. |
|                      | GRU              | Parameter efficiency and training speed. Weaker long-sequence modeling capability compared to LSTM. |
|                      | TCN              | Parallel computation and stable gradients. Sensitivity to input length and limited local feature capture. |
| **Modified Algorithms**  | BiLSTM           | Combines forward and backward LSTMs to capture bidirectional dependencies in time series.                |
|                      | Transformer      | Global dependency modeling and parallelism. Quadratic complexity and reliance on positional encoding.   |
|                      | Informer         | ProbSparse attention improves efficiency on long sequences. Complex hyperparameter tuning.               |
|                      | ConvGRU          | Spatiotemporal feature fusion. Convolutional kernels limit long-range dependency capture.               |
|                      | TimeGNN          | Dynamic spatiotemporal graph modeling. Limited causal interpretability and high preprocessing complexity. |

## 3. Innovative Algorithms and Theories

### GRUGNN

TimeGNN is a time series forecasting method based on graph neural networks (GNNs). It aims to capture the spatiotemporal dependencies in multivariate time series through dynamic graph modeling. Combining TimeGNN with GNN-GRU (Graph Neural Network - Gated Recurrent Unit) creates a hybrid model that integrates graph structure and temporal dynamics.

### Kolmogorov-Arnold Networks (KAN)

The core theory of KAN is based on the Kolmogorov-Arnold Representation Theorem, which states that any multivariate continuous function can be decomposed into a finite combination of univariate functions. Unlike MLPs that rely on multi-layer nonlinear transformations to approximate complex functions, KAN explicitly learns combinations of univariate functions.

**Advantages**:

- Parameter Efficiency: KAN concentrates parameters on fitting univariate functions (e.g., splines or polynomials), whereas MLPs require massive parameters to learn high-dimensional linear combinations.

- Dimensionality Mitigation: By decomposing high-dimensional functions into univariate functions, KAN alleviates the parameter explosion problem in MLPs caused by high-dimensional inputs.

**Architectural Design**: Learnable Activation Functions on Edges

**Spline Functions on Edges**:

Each weight parameter is replaced with a univariate spline function (e.g., B-spline), composed of a linear combination of local basis functions (e.g., piecewise polynomials). The formula for a B-spline is:

**Hierarchical Network Expansion**: By stacking KAN layers (each layer being a matrix of univariate functions), a deep network is constructed, breaking through the limitation of the two-layer network in the original theorem.

**Model Comparison**:

| Model                | Multi-Layer Perception (MLP) | Kolmogorov-Arnold Network (KAN) |
|----------------------|------------------------------|---------------------------------|
| **The**orem          | Universal Approximation Theorem | Kolmogorov-Arnold Representation Theorem |
| **Model Depth**      | Shallow                      | Shallow                         |
| **Formula (Shallow)**| Formula                      | Formula                         |

## 4. Graphical User Interface (GUI) Guide

### 4.1 Main Interface Functional Zones

| Zone | Functionality             | Key Elements                     |
|------|---------------------------|----------------------------------|
| A    | Data Management           | File path configuration, Feature column selection |
| B    | Parameter Configuration   | Model selection, Core parameter setup |
| C    | Advanced Control          | Training optimization, Prediction customization |
| D    | Runtime Monitoring        | Training process tracking, System diagnostics |

### 4.2 Workflow Demo: GNSS Elevation Prediction

| Step | Operation                               |
|------|-----------------------------------------|
| 1    | Data Loading:<br>- Click Browse to select gnss.csv<br>- Set input features: [time, value]<br>- Set output target: [value] |
| 2    | Model Setup:<br>- Select LSTM model<br>- Configure window_size=30<br>- Set batch_size=64 |

## 5. Dual-Mode Comparison Guide

### 5.1 Feature Matrix

| Feature            | CLI Version          | GUI Version          |
|--------------------|----------------------|----------------------|
| Batch Prediction   | ✅ Supports parameter files | ❌ Requires manual input |
| Real-time Monitoring | ❌ Text-only output   | ✅ Visual progress bars |
| Result Export      | ✅ CSV + static charts | ✅ Auto-generated interactive plots |

## 6. Quick Start Examples

### 1. Command-Line Mode

```bash
python main.py \  
  --path data/gnss.csv \  
  --input_features "time,value,temperature" \  
  --output_features "value" \  
  --model_name TimeGNN \  
  --window_size 30 \  
  --seq_len 30 \  
  --batch_size 16 \  
```

### 2. GUI Workflow

| Step | Action                               | Technical Details |
|------|--------------------------------------|-------------------|
| 1    | Data Loading:<br>- Click Browse to select CSV file<br>- Set input/output feature columns | Supported formats: CSV/JSON (UTF-8 encoded)<br>Auto-detects timestamp formats (ISO 8601 preferred) |
| 2    | Model Configuration:<br>- Select model type (e.g., TimeGNN)<br>- Set window size & batch size | Model-specific constraints enforced:<br>✓ LSTM: Requires seq_len parameter<br>✓ TimeGNN: Auto-calculates graph hops |
| 3    | Advanced Parameters:<br>- Click Advanced Settings<br>- Add model-specific parameters | Example configuration:<br>`python { "--seq_len": 30, # Temporal sequence length<br> "--dropout_rate": 0.2 # Regularization strength<br>}` |
| 4    | Execution Monitoring:<br>- Click Start Training<br>- Monitor real-time metrics | Key observables:<br>- Loss curve (MAE/RMSE) |
# GTS_Forecaster
