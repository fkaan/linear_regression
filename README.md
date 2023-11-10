"# linear_regression" 
# Linear Regression Model for Predicting Car Prices

## Overview

This repository contains Python code for a linear regression model to predict car prices based on various features. The model is implemented using gradient descent for optimization.

## Features

- **Scaling**: The code includes a min-max scaling function for numerical features.
- **One-Hot Encoding**: Categorical features like fuel type, seller type, transmission, and owner are one-hot encoded for model compatibility.
- **Gradient Descent**: The linear regression model is trained using gradient descent with different learning rates (alpha values).

## Dependencies

- Python 3.x
- pandas
- matplotlib

## Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/fkaan/linear_regression.git
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the script:**

    ```bash
    python linear_regression_model.py
    ```

## Dataset

The code uses a CSV file (`trainData.csv`) as the training dataset. Replace it with your dataset if needed.

## Configuration

Adjust hyperparameters like learning rate (`alpha`) and the number of iterations in the script according to your requirements.

## Results

The script generates a plot showing the cost values over iterations for different learning rates.


## Author

- [fkaan](https://github.com/fkaan)

Feel free to contribute by opening issues
