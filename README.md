# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize parameters (slope m, intercept b) and choose a learning rate.
2. Compute predicted values using:
3. Calculate error and gradients, then update m and b.
4. Repeat until the error is minimized (convergence).

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: LOGESHWARI N
RegisterNumber:  212225040206
*/

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Dataset (X = input, y = output)
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Step 2: Initialize parameters
m = 0   # slope
b = 0   # intercept
learning_rate = 0.01
epochs = 1000
n = len(X)

# Step 3: Gradient Descent
for i in range(epochs):
    y_pred = m * X + b
    
    # Calculate gradients
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

# Step 4: Final model
print("Slope (m):", m)
print("Intercept (b):", b)

# Step 5: Predictions
y_pred = m * X + b

# Step 6: Plot
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

## Output:
<img width="852" height="553" alt="Screenshot 2026-04-28 090417" src="https://github.com/user-attachments/assets/ae89c0fb-6b07-4d4a-949e-b9bbf7ab3bfa" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
