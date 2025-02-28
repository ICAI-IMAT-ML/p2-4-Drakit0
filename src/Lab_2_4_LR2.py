import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000)-> tuple[list[float], list[np.ndarray], list[np.ndarray], list[np.float16]] | None:
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            gradient (list(floar)): Gradient for the weights and each coefficient.
            weights (list(np.ndarray)): Weight on each iteration.
            biases (list(np.ndarray)): Bias on each iteration.
            losses (list(np.float16)): Loss on each iteration.
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X:np.ndarray = X.reshape(-1, 1)
        
        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            gradients, weights, biases, losses= self.fit_gradient_descent(X, y, learning_rate, iterations)

            return gradients, weights, biases, losses

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        X_ones_transpose:np.ndarray = np.transpose(X)
        X_ones_transpose_X_ones:np.ndarray = np.dot(X_ones_transpose, X)

        X_ones_transpose_X_ones_inv:np.ndarray = np.linalg.inv(X_ones_transpose_X_ones)
        coefficients_matrix:np.ndarray = np.dot(np.dot(X_ones_transpose_X_ones_inv, X_ones_transpose), y)
            
        self.intercept:np.ndarray = coefficients_matrix[0]
        self.coefficients:np.ndarray = coefficients_matrix[1:]

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000)-> tuple[list[float], list[np.ndarray], list[np.ndarray], list[np.float16]]:
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            gradient (list(floar)): Gradient for the weights and each coefficient.
            weights (list(np.ndarray)): Weight on each iteration.
            biases (list(np.ndarray)): Bias on each iteration.
            losses (list(np.float16)): Loss on each iteration.
        """

        # Initialize the parameters to very small values (close to 0)
        m:int = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1]) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01
        
        gradients:list[float] = []
        weights:list[np.ndarray] = []
        biases:list[np.ndarray] = []
        losses:list[np.float16] = []
    
        # Implement gradient descent 
        for epoch in range(iterations):
            predictions:np.ndarray = self.predict(X)
            error:np.ndarray = predictions - y

            # Write the gradient values and the updates for the paramenters
            gradient:np.ndarray = X.T.dot(error) / m
            self.intercept -= learning_rate * np.mean(error)
            self.coefficients -= learning_rate * gradient

            # Calculate and print the loss every 10 epochs
            if epoch % 100 == 0:
                mse:np.float16 = np.mean(error ** 2)
                print(f"Epoch {epoch}: MSE = {mse}")
                
            gradients.append(gradient)
            weights.append(self.coefficients)
            biases.append(self.intercept)
            losses.append(mse)
            
        return gradients, weights, biases, losses

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            predictions:np.ndarray = self.intercept + self.coefficients * X
            
        else:
            predictions:np.ndarray = np.dot(X, self.coefficients) + self.intercept
            
        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    # R^2 Score
    r_squared:float = 1 - sum((y_true - y_pred) ** 2) / sum((y_true - np.mean(y_true)) ** 2)

    # Root Mean Squared Error
    rmse:np.ndarray = np.sqrt(sum((y_true - y_pred) ** 2) / len(y_true))

    # Mean Absolute Error
    mae:float = sum(abs(y_true - y_pred)) / len(y_true)

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()
    for index in sorted(categorical_indices, reverse=True):
        # Extract the categorical column
        categorical_column:np.ndarray = X_transformed[:, index]

        # Find the unique categories (works with strings)
        unique_values:np.ndarray = np.unique(categorical_column)

        # Create a one-hot encoded matrix (np.array) for the current categorical column
        one_hot:np.ndarray = np.zeros((len(categorical_column), len(unique_values)))
        for i, value in enumerate(unique_values):
            one_hot[:, i] = (categorical_column == value).astype(int)

        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]

        # TODO: Delete the original categorical column from X_transformed and insert new one-hot encoded columns
        X_transformed:np.ndarray = np.delete(X_transformed, index, axis=1)
        for i in range(one_hot.shape[1]):
            X_transformed = np.insert(X_transformed, index + i, one_hot[:, i], axis=1)

    return X_transformed

