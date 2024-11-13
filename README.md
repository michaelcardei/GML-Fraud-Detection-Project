# GML-Fraud-Detection-Project

## Fraud Detection Using Graph Machine Learning
This repository provides implementations for fraud detection using various Graph Neural Networks (GNNs), specifically tailored for the IEEE-CIS Fraud Detection dataset ([source](https://www.kaggle.com/competitions/ieee-fraud-detection)). This work uses three graph-based methods to identify fraudulent transactions based on various user and transaction attributes, and compares the efficacy of each one to determine which is best suited for the task.

## Project Overview
The project aims to predict fraudulent transactions in a transaction network by modeling the transactions as a graph: where nodes represent transactions, and edges denote relationships between these transactions based off of several features. This repository includes the entirety of our code, including how to preprocess the data, creating a graph representation, building the architectures of the models, as well as training and evaluation.

### Dataset

The dataset, found on Kaggle, consists of:
1. **Transactions Table**: Contains details like transaction ID, payment type, timestamp, country, and `isFraud` label.
2. **Identity Table**: Contains device and browser information, operating system, and the transaction ID. 

**Note:** Several features from each dataset were anonymized by the creator due to privacy concerns.

## Setup

### Requirements
- **Python** >= 3.7
- **PyTorch** >= 2.5.0
- **PyTorch Geometric (PyG)** >= 2.6.1
- **scikit-learn** >= 1.5.2
- **pandas** >= 2.2.3 
- **numpy** >= 2.1.2


## Contact
For questions or issues, contact Michael Cardei or Spencer Hernández (slh3mm@virginia.edu).
