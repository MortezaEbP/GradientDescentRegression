
# Gradient Descent Regression
### The GradientDescentRegression class is a simple implementation of linear regression using the gradient descent optimization algorithm. Linear regression is a fundamental supervised learning technique used to model the relationship between a dependent variable and one or more independent variables. It finds the best-fitting straight line through the data points to make predictions.

  </a>
  <!-- scikit-learn -->
  <a href="https://scikit-learn.org/stable/" target="_blank" rel="noreferrer">
    <img src="https://blog.paperspace.com/content/images/2018/05/convex_cost_function.jpg" />
  </a>


# Mathematical Formulation:
### In a simple linear regression with one independent variable, the relationship between the dependent variable y and the independent variable x is represented as:

$\huge{\mathbf{y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 = \sum ( \beta_i X_i)}}$


## where:

#### • y is the target (dependent) variable.
#### • x is the input (independent) variable.
#### • β₀ (intercept) is the value of y when x is 0.
#### • β (coefficient) is the slope of the line, representing the change in y for a unit change in x.



### The objective of linear regression is to find the optimal values for β₀ and β that minimize the difference between the predicted values and the actual target values. This is typically achieved by minimizing the mean squared error (MSE) or the sum of squared residuals.



# Gradient Descent and Vectorized Operations:
### In the given implementation, the model utilizes vectorized operations for efficient computation. Vectorization is a technique that enables performing mathematical operations on entire arrays or matrices at once, rather than element-wise computation. NumPy, a popular numerical library in Python, supports vectorized operations, making it suitable for this implementation.

### The update equations for gradient descent can be expressed using vectorized operations, which can significantly improve the computational efficiency. Instead of looping over individual data points, the entire dataset can be processed in one go.

## Update Equations (Vectorized):
### 1) Update β₀ (intercept):
##### $\huge{\beta_0 \leftarrow \beta_0 - \alpha \frac{1}{m} \sum (y - \hat{y})}$

### This equation can be vectorized as follows:
# $\beta_0 \leftarrow \beta_0 - \alpha \frac{1}{m} (\mathbf{y} - \mathbf{X} \mathbf{\beta})$

### 2) Update β (coefficient):
# $\beta \leftarrow \beta - \alpha \frac{1}{m} \sum ((y - \hat{y}) \cdot x)$
### This equation can be vectorized as follows:
$\Huge{\mathbf{\beta}} \leftarrow \Huge{\mathbf{\beta}} - \Huge{\alpha} \frac{\Huge{1}}{\Huge{m}} (\Huge{\mathbf{x}}^T (\Huge{\mathbf{y}} - \Huge{\mathbf{X}} \Huge{\mathbf{\beta}}))$


### where:

#### • m is the number of samples in the training data.
#### • y is the actual target values.
#### • x is the input features.
#### • α (learning rate) is a hyperparameter that controls the step size in each iteration.
#### • X is the matrix of input features with shape (n_samples, n_features).
#### • β is the vector of model parameters, [β₀, β], which includes the intercept and coefficient.

## How to Use:
### To use this implementation, follow these steps:

## 1) Instantiate the Model:
#### Create an instance of the GradientDescentRegression class.

## 2) Fit the Model:
#### Train the model on your training data using the fit method. Provide the training features and target values along with optional parameters like learning rate (lr) and maximum iterations (max_iteration).

## 3) Make Predictions:
#### Use the predict method to make predictions on new data. Pass the input data with shape (n_samples, n_features), and the method will return the predicted target values.

## 4) Evaluate the Model:
#### Assess the performance of the model using the score method, which calculates the coefficient of determination R_Score.
#### The R_Score indicates the goodness of fit of the model by measuring how well the predicted values match the actual target values.



