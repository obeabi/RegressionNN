# This is a sample Python script.
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import RandomizedSearchCV
from NNRegressor import ANNRegressor
from ExploratoryAnalysis import extensive_eda
from FeatureEngineering import clean
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

df = pd.read_csv('Admission_Prediction.csv')
df = df.drop(df.columns[0], axis=1)
X = df.iloc[:, 0:-1]
#x = df.drop(columns=['Chance of Admit'])
y = df.iloc[:, -1]
#y = df['Chance of Admit']
# Perform EDA using extensive_eda class
#eda = extensive_eda()
#eda.save_eda_html(df)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
     # Create object
    model = clean()
    print(model)
    #best_columns, _ = model.correlation_multicollinearity(X)
    best_columns, _ = model.vif_multicollinearity(X)
    print(f"\nImportant columns after performing the mult-collinearity test using VIF approach are :{best_columns}\n")
    numerical_cols, category_cols = model.find_numericategory_columns(X)
    print("\nThe numeric columns are :", numerical_cols)
    print("\nThe categorical columns are :", category_cols)
    X_best = X[best_columns].copy()
    X_train, X_test, y_train, y_test = model.split_train_test(X_best, y, test_size=0.2)
    model.preprocessor_fit(X_train, one_hot_encode_cols=category_cols, label_encode_cols=None)
    # Transform X_train
    X_train_transformed = model.preprocessor_transform(X_train).values
    print(X_train_transformed)

    # #perform one hot encoding
    # ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    # X = np.array(ct.fit_transform(X))
    # Create and tune the model
    regressor = ANNRegressor(input_dim=X_train.shape[1], hidden_layers=[64, 32, 16, 8])
    regressor.train(X_train_transformed, y_train, epochs=300, batch_size=16)
    regressor.plot_loss_history()
    # Hyperparameter tuning
    tuner = regressor.tune(X_train_transformed, y_train, max_trials=10, executions_per_trial=1, directory='Admissions')
    print(f'Best hyperparameters: {tuner.get_best_hyperparameters(num_trials=1)[0].values}')
    #
    # Retrain the model with the best hyperparameters on the entire training set
    regressor.build_model()  # Rebuild the model with the best hyperparameters
    regressor.model.fit(X_train_transformed, y_train, epochs=100, batch_size=32, verbose=1)
    # Evaluate the model with the best parameters on the test set
    X_test_transformed = model.preprocessor_transform(X_test).values
    test_loss = regressor.evaluate(X_test_transformed, y_test)
    print(f'Test Loss with best hyperparameters: {test_loss}')
    regressor.model_summary()
    y_pred = regressor.predict(X_test_transformed)
    print("Root mean squared error is :", root_mean_squared_error(y_pred, y_test))
    print("R2 score is :", r2_score(y_pred, y_test))

