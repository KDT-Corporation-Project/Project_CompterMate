import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    BayesianRidge,
    ElasticNet,
    ARDRegression,
    HuberRegressor,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# Import custom functions
from bayesian_optimizer import bayes_SVR_RFR, bayes_scores


# Define the models and scalers
models = [
    ("LinearRegression", LinearRegression()),
    ("Lasso", Lasso()),
    ("Ridge", Ridge()),
    ("BayesianRidge", BayesianRidge()),
    ("ElasticNet", ElasticNet()),
    ("RandomForestRegressor", RandomForestRegressor()),
    ("GradientBoostingRegressor", GradientBoostingRegressor()),
    ("AdaBoostRegressor", AdaBoostRegressor()),
    ("LGBMRegressor", LGBMRegressor()),
    ("ARDRegression", ARDRegression()),
    ("HuberRegressor", HuberRegressor()),
]

scalers = [
    ("None", None),
    ("StandardScaler", StandardScaler()),
    ("MinMaxScaler", MinMaxScaler()),
    ("RobustScaler", RobustScaler()),
]


class Scale_Prediction:
    def __init__(self, models=models, scalers=scalers):
        self.models = models
        self.scalers = scalers

    @staticmethod
    def preps(data):
        """Data preprocessing
        1) Filter data for scale_pv < 5, E_scr_pv == 8, k_rpm_pv > 50
        2) Split data into train and october data
        3) Drop unnecessary columns
         : E_scr_sv, E_scr_pv, c_temp_sv, s_temp_sv, k_rpm_sv, n_temp_sv, Unnamed: 12
        4) Convert time to datetime
        5) Return train_data and oct_data
        """
        data = data[data["scale_pv"] < 5]
        data = data[data["E_scr_pv"] == 8]
        data = data[data["k_rpm_pv"] > 50]
        data = data[data["c_temp_pv"] > 65]
        data.drop(
            [
                "E_scr_sv",
                "E_scr_pv",
                "c_temp_sv",
                "s_temp_sv",
                "k_rpm_sv",
                "n_temp_sv",
                "Unnamed: 12",
            ],
            axis=1,
            inplace=True,
        )
        data["time"] = pd.to_datetime(data["time"])
        oct_data = data[data["time"].dt.month == 10].drop("time", axis=1)
        train_data = data[data["time"].dt.month != 10].drop("time", axis=1)
        return train_data, oct_data

    def scale_2_4(self, data):
        """Filter data for scale_pv between 2 and 4
        Args:
            data (DataFrame): DataFrame with scale_pv column
        Returns:
            data (DataFrame): DataFrame with scale_pv between 2 and 4
        """
        return data[(data["scale_pv"] < 4) & (data["scale_pv"] > 2)]

    def feature_importance_finder(self, model, data):
        """Find top 3 feature importances of the model
        """
        if hasattr(model, "feature_importances_"):
            top_features = data.columns[np.argsort(model.feature_importances_)[::-1][:3]]
            return ' '.join(top_features.tolist())
        elif hasattr(model, "coef_"):
            top_features = data.columns[np.argsort(model.coef_)[::-1][:3]]
            return ' '.join(top_features.tolist())
        else:
            return "None"

    def basic_modeling(self, X_train, y_train, X_test, y_test):
        """Perform basic modeling with various scalers and models
        1) Loop through models and scalers
        2) Fit the model with training data
        3) Cross-validation with 5 folds
        4) Predict the test data
        5) Calculate the difference between y_test and y_pred
        6) Return the scores in a DataFrame
        
        Args:
            X_train (DataFrame): Training data without the target
            y_train (Series): Training target
            X_test (DataFrame): Test data without the target
            y_test (Series): Test target
            
        Returns:
            scores (DataFrame): DataFrame with columns ['model', 'scaler', 'mae', 'mape', 'mean_diff', 'std_diff', 'min_diff', '25%_diff', '50%_diff', '75%_diff', 'max_diff', 'top_features']
        """
        scores = []
        for name, model in self.models:
            for scaler_name, scaler in self.scalers:
                pipe = Pipeline([("scaler", scaler), ("model", model)])
                print(f"{name} with {scaler_name}.", end=" ")

                # Cross-validation scores
                mae_scores = cross_val_score(
                    pipe,
                    X_train,
                    y_train,
                    cv=5,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                )
                mape_scores = cross_val_score(
                    pipe,
                    X_train,
                    y_train,
                    cv=5,
                    scoring="neg_mean_absolute_percentage_error",
                    n_jobs=-1,
                )

                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                diff = y_test - y_pred

                scores.append(
                    {
                        "model": name,
                        "scaler": scaler_name,
                        "mae": mae_scores.mean() * (-1),
                        "mape": mape_scores.mean() * (-100),
                        "mean_diff": diff.mean(),
                        "std_diff": diff.std(),
                        "min_diff": diff.min(),
                        "25%_diff": diff.quantile(0.25),
                        "50%_diff": diff.quantile(0.5),
                        "75%_diff": diff.quantile(0.75),
                        "max_diff": diff.max(),
                        "top_features": self.feature_importance_finder(model, X_train),
                    }
                )
        return pd.DataFrame(scores)

    def plot_results(self, scores):
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        sns.barplot(data=scores, x="model", y="mae", hue="scaler", ax=ax[0])
        sns.barplot(data=scores, x="model", y="mape", hue="scaler", ax=ax[1])
        sns.barplot(data=scores, x="model", y="r2", hue="scaler", ax=ax[2])
        plt.show()

    def plot_diff(self, scores):
        """Visualize the distribution of diffs"""
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        sns.boxplot(data=scores, x="model", y="mean_diff", ax=ax[0])
        sns.histplot(data=scores, x="mean_diff", kde=True, ax=ax[1])
        sns.scatterplot(data=scores, x="mean_diff", y="std_diff", hue="model", ax=ax[2])
        plt.show()

    def sum_rank(self, data):
        """Calculate rank sum to find the optimal model"""
        data["diff_abs"] = data["max_diff"] - data["min_diff"]
        data["sum_rank"] = (
            data["mae"].rank()
            + data["mape"].rank()
            + data["mean_diff"].rank()
            + data["diff_abs"].rank()
        )
        data.sort_values("sum_rank", inplace=True)
        return data


if __name__ == "__main__":
    print("Start Preprocessing")
    warnings.filterwarnings(action="ignore")

    start = time.time()
    data = pd.read_csv("../DATA/raw_2023051820231018_경대기업맞춤형.csv")

    rpm_modeling = RPMModeling(models, scalers)

    train_data, oct_data = rpm_modeling.preps(data)
    train_data_2_4 = rpm_modeling.scale_2_4(train_data)
    oct_data_2_4 = rpm_modeling.scale_2_4(oct_data)

    X_train = train_data_2_4.drop("scale_pv", axis=1)
    y_train = train_data_2_4["scale_pv"]
    X_test = oct_data_2_4.drop("scale_pv", axis=1)
    y_test = oct_data_2_4["scale_pv"]

    print("Start Modeling")
    scores = rpm_modeling.basic_modeling(X_train, y_train, X_test, y_test)
    print(scores)
    # rpm_modeling.plot_results(scores)
    # rpm_modeling.plot_diff(scores)

    print("End Modeling")
    print("Elapsed Time: ", time.time() - start)
