import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from Logger import CustomLogger

logs = CustomLogger()


class clean:
    """
    This class object handles regression problems
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor = None
        self.scaler = StandardScaler()
        self.max_scaler = MinMaxScaler()
        self.le_encoder = LabelEncoder()

    def find_numericategory_columns (self, X):
        """
        Find numerical and categorical columns from dataset
        :param X:
        :return: numeric and categorical column names
        """
        try:
            numeric_columns = X.select_dtypes(include=[float, int]).columns.tolist()
            categorical_columns = X.select_dtypes(exclude=[float, int]).columns.tolist()
            return numeric_columns, categorical_columns
        except Exception as e:
            raise ValueError(f"Something went wrong while finding the numeric and categorical columns: {e}")
            logs.log("Something went wrong while finding the numeric and categorical columns", level='ERROR')

    def standard_scale_features(self, X):
        """
        Perform standardization of the dataset using Standard Scaler
        :param X:
        :return: standard scaled features
        """
        try:
            return self.scaler.fit_transform(X)
        except Exception as e:
            raise ValueError(f"Error in scaling features: {e}")
            logs.log("Something went wrong while scaling features", level='ERROR')

    def minmax_scale_features(self, X):
        """
        Perform standardization of the dataset using MinMax Scaler
        :param X:
        :return: min-max scaled features
        """
        try:
            return self.max_scaler.fit_transform(X)
        except Exception as e:
            raise ValueError(f"Error in scaling features: {e}")
            logs.log("Something went wrong while scaling features", level='ERROR')

    def labelEncode_feature(self, X):
        """
        Perform encoding of the column using Label Encoder
        :param X:
        :return: label-encoded feature
        """
        try:
            return self.le_encoder.fit_transform(X)
        except Exception as e:
            raise ValueError(f"Error in encoding features: {e}")
            logs.log("Something went wrong while encoding features", level='ERROR')

    def pre_process(self, X):
        """
        Create data pre-processing pipeline
        :param X:
        :return:
        """
        try:
            numeric_features, categorical_features = self.find_numericategory_columns(X)
            if (len(numeric_features) > 0) and (len(categorical_features) > 0):
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())])
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                    #('encoder', LabelEncoder())
                    ])
                self.preprocessor = ColumnTransformer(
                        transformers=[('num', numeric_transformer, numeric_features),
                                      ('cat', categorical_transformer, categorical_features)])

            elif (len(numeric_features) > 0) and (len(categorical_features) == 0):
                numeric_transformer = Pipeline(steps=[
                     ('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler())])
                self.preprocessor = ColumnTransformer(
                        transformers=[('num', numeric_transformer, numeric_features)])

            elif (len(numeric_features) == 0 ) and (len(categorical_features) > 0):
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                   # ('encoder', LabelEncoder())
                ])
                self.preprocessor = ColumnTransformer(
                        transformers=[('cat', categorical_transformer, categorical_features)])

            x_preprocessed = self.preprocessor.fit_transform(X)
            logs.log("Successfully performed the pre-processing step!")
            return x_preprocessed

        except Exception as e:
            raise ValueError(f"Error in preprocessing data: {e}")
            logs.log("Something went wrong while pre-processing the dataset", level='ERROR')

    def split_train_test(self, X, y, test_size=0.2):
        """
        Split dataset into tain and test split using test size of 20%
        :param X:
        :param y:
        :param test_size:
        :return:
        """
        try:
            logs.log("Successfully splitted the dataset to train-test set!")
            return train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        except Exception as e:
            raise ValueError(f"Error in splitting dataset into train-test sets {e}")
            logs.log("Something went wrong while splitting dataset into train-test sets", level='ERROR')

    def vif_multicollinearity(self, X, threshold=10.0):
        """
        Checks for multi-collinearity between features doesn't work well
        :param X:
        :param threshold:
        :return: non-collinear features
        """
        try:
            numeric_features, _ = self.find_numericategory_columns(X)
            x_num = self.pre_process(X[numeric_features])
            vif_data = pd.DataFrame()
            vif_data["feature"] = X[numeric_features].columns
            vif_data["VIF"] = [variance_inflation_factor(x_num, i) for i in range(X[numeric_features].shape[1])]

            # Drop columns with VIF above the threshold
            high_vif_features = vif_data[vif_data["VIF"] > threshold]["feature"].tolist()
            x_dropped = X.drop(columns=high_vif_features)
            logs.log("Successfully performed the multi-collinearity check step!")

            return x_dropped.columns, high_vif_features, vif_data
        except Exception as e:
            raise ValueError(f"Error in checking multi-collinearity: {e}")
            logs.log("Something went wrong while checking multi-collinearity:", level='ERROR')

    def correlation_multicollinearity(self, X, threshold=0.8):
        """
        Checks for multi-collinearity between features using pearson correlation
        :param X:
        :param threshold:
        :return: non-collinear features
        """
        try:
            numeric_features, _ = self.find_numericategory_columns(X)
            x_num = X[numeric_features].dropna()
            correlation_matrix = x_num.corr().abs()
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            high_correlation_pairs = [(column, row) for row in upper_triangle.index for column in upper_triangle.columns if upper_triangle.loc[row, column] > threshold]
            columns_to_drop = {column for column, row in high_correlation_pairs}
            df_reduced = X.drop(columns=columns_to_drop)
            return df_reduced.columns, columns_to_drop
        except Exception as e:
            raise ValueError(f"Error in checking multi-collinearity: {e}")
            logs.log("Something went wrong while checking multi-collinearity:", level='ERROR')



    def __str__(self):
        return "This is my custom feature engineering class object"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Installed successfully!")
