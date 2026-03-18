# Intrusion Detection System using Deep Learning

## Project Overview

This project presents a **Network Intrusion Detection System (IDS)** using machine learning and deep learning techniques. The system is designed to detect anomalies and classify different types of network attacks using the **KDDCUP99 dataset**.

The solution combines:

- **Autoencoder (Unsupervised Learning)** for anomaly detection  
- **LSTM (Deep Learning)** for multiclass classification of attack types  

A **Flask-based web application** is developed to allow users to upload network traffic data and visualise results interactively.

## Features

- Detects **anomalies (normal vs attack traffic)**
- Classifies attacks into:
  - DOS
  - Probe
  - R2L
  - U2R
  - Normal
- Upload CSV data for real-time analysis
- Visualisation using charts and tables
- Web-based interface using Flask

## Dataset

- **KDDCUP99 Dataset**
- Widely used benchmark dataset for intrusion detection
- Contains labelled network traffic data with multiple attack types

## Models Used

### 1. Autoencoder (Anomaly Detection)
- Detects abnormal network behaviour
- Uses reconstruction error (MSE) to identify anomalies
- Threshold-based classification

### 2. LSTM (Multiclass Classification)
- Classifies detected traffic into attack categories
- Handles sequential data representation
- Outputs attack type predictions

## Workflow

1. Upload CSV file containing network data
2. Preprocessing using saved pipeline (`preprocessor.joblib`)
3. Autoencoder detects anomalies
4. Results stored and visualised
5. LSTM model classifies attack types
6. Outputs displayed via web interface

## Tech Stack

- Python
- Flask
- Keras / TensorFlow
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- HTML, CSS

## Project Structure

Intrusion-Detection-System/
│
├── Intrusiondetection_project/
│   │
│   ├── run4.py                         # Main Flask application
│   ├── autoencoder.h5                 # Trained Autoencoder model
│   ├── intrusion_detection_LSTM_model_6.h5   # Trained LSTM model
│   ├── preprocessor.joblib            # Preprocessing pipeline
│   ├── KDDCUP99_Xtest.csv             # Sample dataset
│   ├── Anomaly_output.csv             # Generated anomaly results
│   │
│   ├── assets/
│   │   ├── count_plot.png             # Anomaly detection visualisation
│   │   ├── count_plot_2.png           # Attack classification visualisation
│   │
│   ├── templates/
│   │   ├── index.html                 # Upload interface
│   │   ├── anomaly_detection.html     # Anomaly results page
│   │   ├── attack_types.html          # Attack classification page

## Results

- Successfully detects anomalies using Autoencoder
- LSTM model classifies attack types effectively
- Visualisations help interpret results
- Demonstrates real-world cybersecurity application of ML

## Key Learnings

- Application of deep learning in cybersecurity
- Anomaly detection using reconstruction error
- Multiclass classification using LSTM
- Data preprocessing for real-world datasets
- Building ML-powered web applications

## Future Improvements

- Real-time streaming data detection
- Deployment on cloud (AWS/GCP)
- Improved model performance using advanced architectures
- Integration with SIEM systems

## How to Run

1. Clone the repository
2. Install dependencies
3. python run4.py
4. Open in browser
