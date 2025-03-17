Project Overview

This project focuses on predicting stock prices using Long Short-Term Memory (LSTM) neural networks. It involves data preprocessing, feature engineering, and model training to forecast future stock prices.

Dataset

The dataset includes historical stock prices with features such as Open, High, Low, Close, Adj Close, and Volume.

Key Steps:

Data Cleaning & Preparation
Handled missing values
Scaled and normalized data
Engineered time-series features (Day, Month, Year, Previous Close values)
Feature Engineering
Created lag variables for improved forecasting
Converted categorical time features into numerical format
Model Building & Training
Implemented LSTM for time-series forecasting
Tuned hyperparameters for better accuracy
Used loss functions and optimizers to reduce forecasting error
Evaluation & Performance Metrics
Compared LSTM model results with baseline models
Used RMSE, MAE, and MAPE for performance evaluation
Results & Findings

The LSTM model effectively captured stock price trends.
Feature engineering improved model performance significantly.
Technologies Used:

Python: Pandas, NumPy, Matplotlib, Seaborn
Machine Learning: Scikit-learn, TensorFlow, Keras
Time-Series Forecasting: LSTM, ARIMA (if applicable)
Visualization: Tableau Dashboard for stock trends
