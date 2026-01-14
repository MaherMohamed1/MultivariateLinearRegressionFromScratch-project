# Linear Regression & Gradient Descent Implementation

A comprehensive implementation of Linear Regression using multiple solution methods, including Normal Equations, Gradient Descent, and Scikit-learn comparison.

## Overview

This project demonstrates three different approaches to solving linear regression problems:
1. **Normal Equations** - Analytical closed-form solution
2. **Gradient Descent** - Iterative optimization algorithm
3. **Scikit-learn** - Library-based implementation for comparison

## Features

- Custom implementation of gradient descent with configurable parameters
- Normal equations solution for exact analytical results
- Cost function tracking and visualization
- Data preprocessing with MinMax scaling
- Comprehensive visualization including:
  - Pair plots for feature-target relationships
  - Correlation heatmap
  - Cost function convergence plot

## Requirements

The following Python packages are required:

```
numpy
pandas
seaborn
matplotlib
scikit-learn
```

Install dependencies using:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

## Dataset

The script expects a CSV file at:
```
C:\Users\Maher\Desktop\dataset_200x4_regression.csv
```

The dataset should have:
- 3 feature columns (Feat1, Feat2, Feat3)
- 1 target column (Target)
- 200 rows

**Note:** You can also use the built-in diabetes dataset by uncommenting lines 13-14 in the code.

## Usage

Simply run the script:
```bash
python LR_GD.py
```

## Implementation Details

### Preprocessing (`preprocessing()`)
- Loads data from CSV file
- Extracts features (first 3 columns) and target (last column)
- Applies MinMaxScaler for feature normalization
- Adds bias term (column of ones) to feature matrix

### Normal Equations Solution (`normal_equations_solution()`)
Computes the optimal weights using the analytical formula:
```
w = (X^T * X)^(-1) * X^T * y
```

### Gradient Descent (`gradient_descent_linear_regression()`)
- **Initial weights**: [1.0, 1.0, 1.0, 1.0]
- **Default step size**: 0.5
- **Default precision**: 0.00001
- **Default max iterations**: 500
- Tracks cost function history for visualization

### Cost Function (`cost_f()`)
Implements Mean Squared Error (MSE) divided by 2:
```
J(w) = (1/2N) * ||Xw - t||²
```

### Gradient Computation (`f_deriv()`)
Computes the gradient of the cost function:
```
∇J(w) = (1/N) * X^T * (Xw - t)
```

### Scikit-learn Comparison (`LR_using_scikit()`)
Uses scikit-learn's LinearRegression for baseline comparison and validation.

## Output

The script will display:
1. Initial random cost
2. Optimal error after gradient descent convergence
3. Scikit-learn parameters and error
4. Normal equations solution weights
5. Three visualization plots:
   - Pair plots showing feature-target relationships
   - Correlation heatmap
   - Cost function convergence over iterations

## Customization

### Adjust Gradient Descent Parameters
Modify the function call in `__main__`:
```python
optimal_weights, cost_history = gradient_descent_linear_regression(
    X, t, 
    step_size=0.1,      # Learning rate
    precision=0.0001,   # Convergence threshold
    max_iter=1000       # Maximum iterations
)
```

### Use Different Dataset
1. Update the file path in `preprocessing()` function (line 15)
2. Adjust column indices if needed (line 17)
3. Or uncomment lines 13-14 to use sklearn's diabetes dataset

## File Structure

```
1- LR&GD/
├── LR_GD.py          # Main implementation file
├── README.md         # This file
├── cost_function.py
├── generalisation_of_GD.py
└── gradient_descent.py
```

## Notes

- The implementation uses vectorized operations for efficiency
- All features are normalized to [0, 1] range using MinMaxScaler
- The bias term is automatically added to the feature matrix
- Cost function history is tracked for convergence analysis

## Author

Maher

## License

This project is for educational purposes.
