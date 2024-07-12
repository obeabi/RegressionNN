# This is a sample Python script.
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import RandomizedSearchCV
from NeuralNetRegressor import ANNRegressor


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_test = np.random.rand(20, 10)
    y_test = np.random.rand(20)
    # Initialize the regressor
    regressor = ANNRegressor(input_dim=10, hidden_layers=[64, 64, 32])
    regressor.train(X_train, y_train, epochs=50, batch_size=32)
    # Evaluate the model
    loss = regressor.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')
    # Predict
    predictions = regressor.predict(X_test)
    print(predictions)

    # Plot the loss history and save it as PNG
    regressor.plot_loss_history('loss_history.png')

    # Print the regressor's string representation
    print(regressor)

