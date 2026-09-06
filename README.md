# Pattern Classification of Stock Price Movement

This repository contains the implementation for my MSc Data Science dissertation,
"Pattern Classification of Stock Price Movement".

The project investigates whether meaningful long-term stock price movement patterns
can be identified among FTSE 100 companies using unsupervised clustering.

## Project Overview

The analysis uses adjusted closing price data for FTSE 100 companies from
2021–2025.

The project compares:

### Data Representations
- Raw adjusted closing prices
- Normalised prices
- Daily percentage returns
- Log returns

### Similarity Measures
- Euclidean distance
- Correlation distance
- Dynamic Time Warping (DTW)

### Clustering Methods
- K-Means
- K-Shape
- Hierarchical clustering
- DBSCAN

The final analysis identifies three broad long-term movement patterns:
relatively stable behaviour, sustained growth, and exceptional growth.

## Repository Structure

- `download_data.py` – Downloads the stock price data used in the analysis.
- `explore_data.py` – Performs exploratory data analysis on the stock price dataset.
- `data_representation.py` – Generates and compares the different stock price representations used for clustering.

### Similarity Measures
- `euclidean_distance.py` – Calculates and analyses Euclidean distances between stock price trajectories.
- `correlation_distance.py` – Calculates and analyses correlation-based distances between stock price trajectories.
- `dtw_distance.py` – Calculates and analyses Dynamic Time Warping (DTW) distances.

### K-Means
- `kmeans_evaluation.py` – Evaluates K-Means across different numbers of clusters.
- `kmeans_final.py` – Performs the final selected K-Means clustering.
- `kmeans_further_k_analysis.py` – Examines additional K-Means solutions at different values of k.
- `kmeans_sensitivity_rr_removed.py` – Performs sensitivity analysis after removing Rolls-Royce (RR.L).

### K-Shape
- `kshape_evaluation.py` – Evaluates K-Shape clustering across different numbers of clusters.
- `kshape_final.py` – Performs the final selected K-Shape clustering.

### Hierarchical Clustering
- `hierarchical_evaluation.py` – Evaluates different hierarchical clustering configurations.
- `hierarchical_final.py` – Performs the final selected hierarchical clustering.

### DBSCAN
- `dbscan_evaluation.py` – Evaluates DBSCAN parameter configurations.
- `dbscan_final.py` – Performs the final selected DBSCAN clustering.

### Cluster Analysis
- `cluster_profiling.py` – Profiles and summarises the characteristics of the identified clusters.

## Running the Project

1. Clone the repository:

https://github.com/yusurhrm/Pattern-Classification-of-Stock-Price-Moving-.git

2. Install the required dependencies:

pip install -r required_download.txt

3. Run the analysis scripts as described in the repository structure above.

## Data

Stock price data was obtained using the Yahoo Finance `yfinance` Python package.
The analysis covers the period from 1 January 2021 to 31 December 2025.

## Requirements

The project was implemented in Python.

Main libraries include:
- pandas
- numpy
- matplotlib
- scikit-learn
- scipy
- yfinance
- tslearn

See `required_download.txt` for the complete list of dependencies.

## Author

Yusur Taha  
MSc Data Science, School of Social Sciences
University of Manchester  
Student ID: 10970776
2026