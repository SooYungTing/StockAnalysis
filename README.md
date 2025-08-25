# StockAnalysis

Nvidia Stock Analysis and Price Prediction

## Project Overview

This project shows a high level of forecasting the daily closing stock prices of NVIDIA based on Long Short-Term Memory (LSTM) neural networks. The workflow includes scouring the historical stock data using Kaggle, preprocessing, feature, engineering, training the model using time-series cross validation, and complete evaluation

## Table Of Content

1. [Installation](#installation)
2. [LSTM?](#why-lstm)
3. [Logic](#logic)

## Installation

```bash
    git clone https://github.com/SooYungTing/StockAnalysis.git

    cd StockAnalysis
    python3 -m venv stock
    source stock/bin/activate #for Mac/Linux
    stock\Scripts\activate #for Windows

    pip3 install ipykernel jupyter matplotlib numpy pandas scikit-learn tensorflow kagglehub
```

If you want to run the code local on your device:

- Create a `.streamlit/secrets.toml` file
- Within `secrets.toml` add this

```toml
ARTIFACT_BASE_URL = "link/to/your/github/model/folder"
```

## Why LSTM?

- **Catches Temporal Patterns**: LSTMs are also capable of memorizing long-term dependencies in series data, which is why they would be suitable in stock-prices series.

- **Handles Nonlinear Dynamics**: Stock markets are nonlinear and simple linear models cannot explain the behavior easily.

- **Avoids Vanishing Gradient Problem**: The LSTM cell design enables training on long sequences with preservation of the gradients over many time steps.

## Logic

1. **Data Acquisition**: retrieve historical price data (Open, High, Low, Close, Volume), using a simple API call, found on Kaggle.

2. **Feature Scaling**: maintain input variables in [0,1] range through MinMax scaling. This makes all features have an equal part and accelerates the training in the neural network.

3. **Preparation of sequences**: Transform the data to the 3D tensors the LSTM layers require: (samples, timesteps, features). We apply here one feature/per day as a single timestep on a time series.

4. **Model definition**: Create a Sequential model having:

   - **LSTM Layer**: 128 units to learn temporal dependencies.

   - **Dense Output**: One neuron as a predictor of the next day closing price.

5. **Training Strategy**:

   - **Train/Test Split**: Split time series with an expanding window to validate with unseen data, but no shuffle.

   - **Early Stopping**: Instead of overfitting, training can be stopped by early stopping boxed and box. The learning is started and continued until validation performance stops improving.

6. **Prediction & Inversion**:

   - **Predict**: The output falls on the normalised range.

   - **Inverse Scale**: Revert scaled predictions to actual price units by means of the same scaler.

7. **Test/evaluation**: Calculate RMSE and MAE on the testing data to measure the forecast.
