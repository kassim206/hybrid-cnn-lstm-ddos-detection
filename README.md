# Hybrid CNN-LSTM-Based Real-Time DDoS Detection and Mitigation

<p align="center">
  <b>Final Master Thesis Project 2026</b><br>
  <b>Hybrid Machine Learning-Based Real-Time DDoS Detection and Mitigation for Web Applications</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue" />
  <img src="https://img.shields.io/badge/Machine%20Learning-DDoS%20Detection-green" />
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN--LSTM-orange" />
  <img src="https://img.shields.io/badge/Thesis-2026-purple" />
  <img src="https://img.shields.io/badge/Status-Final%20Experiments%20Completed-brightgreen" />
</p>

<p align="center">
  <a href="https://kassim206.github.io/hybrid-cnn-lstm-ddos-detection/">
    <b>Project Website</b>
  </a>
  |
  <a href="https://github.com/kassim206/hybrid-cnn-lstm-ddos-detection/releases/tag/v1.1-thesis-experiments">
    <b>Final Experimental Release</b>
  </a>
</p>

---

## Author

**Mohammed Kassim Cherukodan**  
Master Thesis, Applied Informatics  
Vytautas Magnus University  
Faculty of Informatics  
Supervisor: **Prof. Dr. Audrius Zajančkauskas**  
Year: **2026**

---

## Thesis Title

**Hybrid Machine Learning-Based Real-Time DDoS Detection and Mitigation for Web Applications**

---

## Project Overview

Distributed Denial of Service attacks are a serious threat to modern web applications, especially Layer 7 attacks such as HTTP flood and WebDDoS attacks. These attacks are difficult to detect because they can imitate legitimate user traffic.

This project implements a hybrid deep learning framework based on **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks for real-time DDoS detection and mitigation.

The CNN component extracts spatial traffic patterns, while the LSTM component captures temporal dependencies across network flow sequences. The proposed CNN-LSTM model is compared with traditional machine learning baseline models, including **Random Forest** and **Support Vector Machine**.

The project also includes a proof-of-concept mitigation module for dynamic rate limiting, traffic filtering, and IP blocking based on model prediction confidence.

---

## Research Objectives

The main objectives of this project are:

- To preprocess and analyze the CIC-DDoS2019 dataset.
- To perform feature selection using correlation analysis and Random Forest feature importance.
- To implement Random Forest and SVM baseline models.
- To design and train a hybrid CNN-LSTM deep learning model.
- To evaluate model performance using accuracy, precision, recall, F1-score, ROC-AUC, and inference latency.
- To implement a proof-of-concept real-time DDoS mitigation module.
- To provide a reproducible research repository for the final master thesis project.

---

## Thesis Figures

### Feature Correlation Heatmap

<p align="center">
  <img src="thesis_figures/figure_4.1_correlation_heatmap.png" width="700">
</p>

<p align="center">
  <i>Figure 4.1: Feature correlation heatmap after feature selection.</i>
</p>

### CNN-LSTM Training Curves

<p align="center">
  <img src="thesis_figures/figure_4.2_training_curves.png" width="700">
</p>

<p align="center">
  <i>Figure 4.2: CNN-LSTM training loss and accuracy curves.</i>
</p>

### Model Performance Comparison

<p align="center">
  <img src="thesis_figures/figure_4.3_performance_comparison.png" width="700">
</p>

<p align="center">
  <i>Figure 4.3: Comparative performance of Random Forest, SVM, and CNN-LSTM models.</i>
</p>

### Inference Latency Comparison

<p align="center">
  <img src="thesis_figures/figure_4.4_latency_comparison.png" width="700">
</p>

<p align="center">
  <i>Figure 4.4: Inference latency comparison for all evaluated models.</i>
</p>

### Dataset Class Distribution

<p align="center">
  <img src="thesis_figures/figure_4.5_class_distribution.png" width="700">
</p>

<p align="center">
  <i>Figure 4.5: Dataset class distribution after preprocessing.</i>
</p>

---

## Additional Figures

### Chapter 2 Correlation Heatmap

<p align="center">
  <img src="thesis_figures/figure_2.1_correlation_heatmap.png" width="700">
</p>

### Chapter 2 Attack Distribution

<p align="center">
  <img src="thesis_figures/figure_2.2_attack_distribution.png" width="700">
</p>

### Chapter 3 Correlation Heatmap

<p align="center">
  <img src="thesis_figures/figure_3.1_correlation_heatmap.png" width="700">
</p>

### Chapter 3 Training Curves

<p align="center">
  <img src="thesis_figures/figure_3.2_training_curves.png" width="700">
</p>

### Chapter 3 Performance Comparison

<p align="center">
  <img src="thesis_figures/figure_3.3_performance_comparison.png" width="700">
</p>

### Chapter 3 Latency Comparison

<p align="center">
  <img src="thesis_figures/figure_3.4_latency_comparison.png" width="700">
</p>

---

## Project Structure

```text
hybrid-cnn-lstm-ddos-detection/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── train_random_forest.py
│   ├── train_svm.py
│   ├── train_cnn_lstm.py
│   ├── evaluate_models.py
│   ├── mitigation.py
│   └── main.py
│
├── data/
│   └── README.md
│
├── models/
│   ├── README.md
│   └── .gitkeep
│
├── results/
│   ├── figures/
│   ├── metrics/
│   ├── tables/
│   │   ├── sample_attack_prediction_test.csv
│   │   └── sample_attack_prediction_summary.csv
│   └── README.md
│
├── thesis_figures/
│   ├── figure_2.1_correlation_heatmap.png
│   ├── figure_2.2_attack_distribution.png
│   ├── figure_3.1_correlation_heatmap.png
│   ├── figure_3.2_training_curves.png
│   ├── figure_3.3_performance_comparison.png
│   ├── figure_3.4_latency_comparison.png
│   ├── figure_4.1_correlation_heatmap.png
│   ├── figure_4.2_training_curves.png
│   ├── figure_4.3_performance_comparison.png
│   ├── figure_4.4_latency_comparison.png
│   └── figure_4.5_class_distribution.png
│
├── scripts/
│   ├── 01_download_dataset.py
│   ├── 02_preprocess_data.py
│   ├── 03_train_baselines.py
│   ├── 04_train_cnn_lstm.py
│   ├── 05_evaluate_models.py
│   ├── 06_test_mitigation.py
│   ├── 07_generate_chapter2_graphs.py
│   ├── 08_test_sample_predictions.py
│   ├── traffic_generation/
│   │   ├── README.md
│   │   └── hping3_connectivity_test.sh
│   └── visualizations/
│       ├── chapter2_visuals.py
│       ├── chapter3_visuals.py
│       └── chapter4_visuals.py
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── docker-compose-lab.yml
│   ├── README.md
│   └── kali/
│       └── Dockerfile
│
├── docs/
│   ├── index.html
│   ├── style.css
│   └── network_topology.md
│
├── notebooks/
│   └── README.md
│
├── configs/
│   └── config.yaml
│
├── thesis/
│   └── thesis_summary.md
│
├── README.md
├── requirements.txt
├── environment_info.txt
├── LICENSE
└── CITATION.cff
```

---

## Technologies Used

- Python 3.9
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Joblib
- Docker
- Kali Linux
- hping3
- CIC-DDoS2019 dataset

---

## Installation

Clone the repository:

```powershell
git clone https://github.com/kassim206/hybrid-cnn-lstm-ddos-detection.git
cd hybrid-cnn-lstm-ddos-detection
```

Create a virtual environment:

```powershell
py -3.9 -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Virtual Environment Configuration

The Python virtual environment configuration is documented using:

```text
requirements.txt
environment_info.txt
```

The project was executed using Python 3.9 inside a local virtual environment. Dependencies can be installed using:

```powershell
pip install -r requirements.txt
```

---

## Running the Project

Run the main project pipeline:

```powershell
python src\main.py
```

Run the experiment runner:

```powershell
python scripts\run_all_old.py
```

Run individual experiment scripts:

```powershell
python scripts\01_download_dataset.py
python scripts\02_preprocess_data.py
python scripts\03_train_baselines.py
python scripts\04_train_cnn_lstm.py
python scripts\05_evaluate_models.py
python scripts\06_test_mitigation.py
python scripts\07_generate_chapter2_graphs.py
python scripts\08_test_sample_predictions.py
```

Run visualization scripts:

```powershell
python scripts\visualizations\chapter2_visuals.py
python scripts\visualizations\chapter3_visuals.py
python scripts\visualizations\chapter4_visuals.py
```

---

## Dataset

This project uses the **CIC-DDoS2019** dataset.

The dataset is not included in this repository because of its large size. Download the dataset separately and place the required CSV files inside:

```text
data/raw/
```

Processed files generated by the preprocessing pipeline should be saved inside:

```text
data/processed/
```

Expected local dataset structure:

```text
data/
├── raw/
│   └── CIC-DDoS2019 CSV files
└── processed/
    └── cleaned and preprocessed files
```

---

## Methodology

The research workflow follows these main stages:

1. Dataset collection and preparation
2. Data cleaning and preprocessing
3. Feature selection and engineering
4. Baseline model training using Random Forest and SVM
5. Hybrid CNN-LSTM model training
6. Model evaluation and comparison
7. Sample attack prediction testing
8. Real-time mitigation prototype testing
9. Visualization of experimental results

---

## Evaluation Metrics

The models are evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall correct classification rate |
| Precision | Measures correctness of predicted attack samples |
| Recall | Measures ability to detect actual attack samples |
| F1-score | Balances precision and recall |
| ROC-AUC | Measures classification separability using predicted probability scores |
| Inference Latency | Measures real-time suitability |
| CPU/RAM Usage | Measures resource overhead |

ROC-AUC was calculated using predicted probability scores. For Random Forest and SVM, probability estimates were obtained using `predict_proba()`. For CNN-LSTM, sigmoid output probabilities were used and evaluated against binary ground-truth labels.

---

## Model Comparison

The project compares three main models:

| Model | Type | Purpose |
|---|---|---|
| Random Forest | Traditional machine learning | Baseline comparison |
| Support Vector Machine | Traditional machine learning | Baseline comparison |
| CNN-LSTM | Hybrid deep learning | Proposed detection model |

---

## Experimental Results

The models were evaluated using the processed CIC-DDoS2019 sample dataset. The comparison includes traditional machine learning baselines and the proposed hybrid CNN-LSTM model.

### Model Performance Summary

| Rank | Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Latency (ms/sample) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Random Forest | **0.999833** | **0.999964** | **0.999856** | **0.999910** | **0.999998** | **0.005417** |
| 2 | CNN-LSTM | 0.993465 | 0.998479 | 0.994445 | 0.996458 | 0.999026 | 0.086719 |
| 3 | SVM | 0.990500 | 0.999745 | 0.989974 | 0.994836 | 0.999099 | 0.116428 |

### Key Findings

- **Random Forest** achieved the highest overall accuracy and the lowest inference latency in this experimental run.
- **CNN-LSTM** achieved strong detection performance and is important for the thesis because it captures temporal traffic patterns.
- **SVM** produced good classification performance, but it had higher inference latency compared with Random Forest and CNN-LSTM.
- The results show that both traditional machine learning and hybrid deep learning methods can detect DDoS traffic effectively on the processed CIC-DDoS2019 sample dataset.

### Result Files

The generated model evaluation outputs are stored in:

```text
results/metrics/model_comparison.csv
results/tables/model_comparison.csv
results/figures/model_comparison.png
results/figures/latency_comparison.png
```

---

## Sample Attack Prediction Test

A sample-level prediction test was performed using processed CIC-DDoS2019 test records. The test selected **100 BENIGN samples** and **100 ATTACK samples** from the processed test set.

The trained Random Forest and SVM models were used to classify these samples. This test verifies that the trained models can process individual benign and attack examples from the prepared dataset.

### Sample Test Summary

| Model | Sample Size | BENIGN Samples | ATTACK Samples | False Positives | False Negatives | True Positives |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 200 | 100 | 100 | 0 | 1 | 99 |
| SVM | 200 | 100 | 100 | 0 | 2 | 98 |

<p align="center">
  <img src="results/figures/sample_attack_prediction_results.png" width="700">
</p>

<p align="center">
  <i>Sample attack prediction results for Random Forest and SVM.</i>
</p>

The generated sample prediction outputs are stored in:

```text
results/tables/sample_attack_prediction_test.csv
results/tables/sample_attack_prediction_summary.csv
```

This is a dataset-based attack sample prediction test. It does not perform live public-network traffic generation.

---

## Real-Time Mitigation Module

The mitigation module is designed as a proof-of-concept component that can respond to detected DDoS activity.

Planned mitigation actions include:

- Dynamic rate limiting
- Suspicious traffic filtering
- IP blocking
- Risk-based response using model confidence scores

The mitigation logic is implemented in:

```text
src/mitigation.py
```

---

## Mitigation Simulation Results

A safe local mitigation simulation was performed using four traffic scenarios: normal traffic, moderate suspicious traffic, high DDoS traffic, and critical DDoS traffic.

The mitigation module applies different actions based on the model confidence score:

| Confidence Score | Mitigation Action |
|---:|---|
| < 0.50 | Allow |
| 0.50 – 0.70 | Monitor |
| 0.70 – 0.90 | Rate Limit |
| > 0.90 | Block |

Generated mitigation outputs are stored in:

```text
results/metrics/mitigation_results.csv
results/tables/mitigation_results.csv
results/figures/mitigation_response.png
results/figures/resource_overhead.png
```

---

## Supplementary Docker/Kali/hping3 Lab

A supplementary controlled Docker laboratory setup is provided for local reproducibility.

The lab includes:

- Nginx victim web server
- Kali Linux container
- hping3 installed inside the Kali container
- Isolated Docker bridge network
- Network topology documentation
- Safe local connectivity test script

The configuration is available in:

```text
docker/docker-compose-lab.yml
docker/kali/Dockerfile
docs/network_topology.md
scripts/traffic_generation/
```

This setup is intended only for local academic testing. The main model experiments use the CIC-DDoS2019 dataset, while the Docker/Kali/hping3 configuration is provided as supplementary controlled laboratory material.

### Docker Lab Startup

From the repository root:

```powershell
docker compose -f docker/docker-compose-lab.yml up --build
```

Open another terminal:

```powershell
docker exec -it kali_hping3_lab bash
```

Inside the Kali container:

```bash
bash /app/scripts/traffic_generation/hping3_connectivity_test.sh
```

---

## Visualization Scripts

The repository includes thesis visualization scripts inside:

```text
scripts/visualizations/
```

Current visualization files:

```text
chapter2_visuals.py
chapter3_visuals.py
chapter4_visuals.py
```

These scripts generate graphs and figures used in the thesis documentation.

---

## Configuration

Project settings are stored in:

```text
configs/config.yaml
```

The configuration file includes:

- Dataset paths
- Model paths
- Training parameters
- CNN-LSTM hyperparameters
- Evaluation settings
- Mitigation thresholds

---

## Experimental Scope Clarification

The main model experiments in this repository use the CIC-DDoS2019 dataset and generated evaluation outputs. The Docker/Kali/hping3 laboratory setup is provided as supplementary controlled reproducibility material.

The hping3 script included in this repository is a small local connectivity test for the isolated Docker network. It is not intended for public-network traffic generation, unauthorized testing, or real attack execution.

The sample attack prediction test uses processed CIC-DDoS2019 dataset records. It verifies model predictions on dataset-based benign and attack samples, not live attack traffic.

---

## Ethical Use Notice

This repository is intended strictly for academic research, defensive cybersecurity education, and controlled laboratory experimentation.

Any attack simulation scripts or traffic generation tools must only be used in isolated environments owned or explicitly authorized by the user.

Unauthorized testing against public systems, third-party networks, or real web services is not permitted.

The author does not support or encourage malicious use of this project.

---

## Citation

If you use this repository, please cite:

```text
Cherukodan, M. K. (2026). Hybrid Machine Learning-Based Real-Time DDoS Detection and Mitigation for Web Applications. Master Thesis, Vytautas Magnus University.
```

---

## Repository Status

This repository is under active development as part of a final master thesis project for 2026.

Current status:

```text
Project structure: Completed
Documentation: Completed
Visualization scripts: Added
Thesis figures: Added
Baseline experiments: Completed
CNN-LSTM experiment: Completed
Evaluation comparison: Completed
Sample attack prediction test: Completed
Mitigation simulation: Completed
Docker configuration: Added
Supplementary Kali/hping3 lab: Added
Network topology documentation: Added
Virtual environment configuration: Added
```

---

## Author Contact

**Mohammed Kassim Cherukodan**  
GitHub: [kassim206](https://github.com/kassim206)

Project Website:  
https://kassim206.github.io/hybrid-cnn-lstm-ddos-detection/

Source Code Repository:  
https://github.com/kassim206/hybrid-cnn-lstm-ddos-detection

Final Experimental Release:  
https://github.com/kassim206/hybrid-cnn-lstm-ddos-detection/releases/tag/v1.1-thesis-experiments