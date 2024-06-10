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
    # SGDRegressor,
    ARDRegression,
    HuberRegressor,
    # Lars,
    # LassoLars,
    # LassoLarsIC,
    # OrthogonalMatchingPursuit,
    # OrthogonalMatchingPursuitCV,
    # PassiveAggressiveRegressor,
    # Perceptron,
    # RANSACRegressor,
    # TheilSenRegressor,
    # TweedieRegressor,
    # PoissonRegressor,
    # GammaRegressor,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

# .py에서 함수 호출
from bayesian_optimizer import bayes_SVR_RFR, bayes_scores


def preps(data):
    """전처리
    1) scale_pv < 5, E_scr_pv == 8, k_rpm_pv > 50
    2) E_scr_sv, c_temp_sv, n_temp_sv, s_temp_sv, k_rpm_sv, n_temp_sv 제거
    3) time 컬럼을 datetime으로 변환
    4) 10월 데이터와 그 외 데이터로 분리
    5) time 컬럼 제거
    6) 4, 5번 과정을 거친 데이터를 반환 : train_data, oct_data
    """
    # 2) scale_pv < 5, E_scr_pv == 8, k_rpm_pv > 50
    data = data[data["scale_pv"] < 5]
    data = data[data["E_scr_pv"] == 8]
    data = data[data["k_rpm_pv"] > 50]

    # 3) E_scr_sv, c_temp_sv, n_temp_sv, s_temp_sv, k_rpm_sv, n_temp_sv 제거
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
    oct_data = data[data["time"].dt.month == 10]
    oct_data = oct_data.drop("time", axis=1)

    train_data = data[data["time"].dt.month != 10]
    train_data = train_data.drop("time", axis=1)

    return train_data, oct_data


def scale_2_4(data):
    """data[(data["scale_pv"] < 5) & (data["scale_pv"] > 2)]"""
    return data[(data["scale_pv"] < 5) & (data["scale_pv"] > 2)]


def feature_importance_finder(model, data):
    """모델의 최고 3개 feature_importance의 이름을 반환"""
    if hasattr(model, "feature_importances_"):
        top_features = data.columns[np.argsort(model.feature_importances_)[::-1][:3]]
        return top_features.tolist()
    elif hasattr(model, "coef_"):
        top_features = data.columns[np.argsort(model.coef_)[::-1][:3]]
        return top_features.tolist()
    else:
        return "None"


def basic_modeling(X_train, y_train, X_test, y_test):
    """StandardScaler를 통한 linear regression, RandomForest, LGBMRegressor Modeling

    return: DataFrame
    """
    models = [
        ("LinearRegression", LinearRegression()),
        ("RandomForestRegressor", RandomForestRegressor()),
        ("LGBMRegressor", LGBMRegressor()),
    ]

    scalers = [
        ("None", None),
        ("StandardScaler", StandardScaler()),
        ("MinMaxScaler", MinMaxScaler()),
        ("RobustScaler", RobustScaler()),
    ]

    scores = []

    for name, model in models:
        for scaler_name, scaler in scalers:
            pipe = Pipeline([("scaler", scaler), ("model", model)])

            print(f"{name} with {scaler_name}.", end=" ")

            # cross_val_score : mae, mape
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
                    "top_features": feature_importance_finder(model, X_train),
                }
            )

    return pd.DataFrame(scores)

 
def model_selection(X_train, y_train, X_test, y_test):

    # 모델 리스트
    models = [
        ("LinearRegression", LinearRegression()),
        ("Lasso", Lasso()),
        ("Ridge", Ridge()),
        ("BayesianRidge", BayesianRidge()),
        ("ElasticNet", ElasticNet()),
        # ('SGDRegressor', SGDRegressor()),
        ("RandomForestRegressor", RandomForestRegressor()),
        ("GradientBoostingRegressor", GradientBoostingRegressor()),
        ("AdaBoostRegressor", AdaBoostRegressor()),
        ("LGBMRegressor", LGBMRegressor()),
        ("ARDRegression", ARDRegression()),
        ("HuberRegressor", HuberRegressor()),
        # ("Lars", Lars()),
        # ("LassoLars", LassoLars()),
        # ("LassoLarsIC", LassoLarsIC()),
        # ("OrthogonalMatchingPursuit", OrthogonalMatchingPursuit()),
        # ("OrthogonalMatchingPursuitCV", OrthogonalMatchingPursuitCV()),
        # ("PassiveAggressiveRegressor", PassiveAggressiveRegressor()),
        # ("Perceptron", Perceptron()),
        # ("RANSACRegressor", RANSACRegressor()),
        # ("TheilSenRegressor", TheilSenRegressor()),
        # ("TweedieRegressor", TweedieRegressor()),
        # ("PoissonRegressor", PoissonRegressor()),
        # ("GammaRegressor", GammaRegressor()),
    ]

    scalers = [
        ("None", None),
        ("StandardScaler", StandardScaler()),
        ("MinMaxScaler", MinMaxScaler()),
        ("RobustScaler", RobustScaler()),
    ]

    scores = []

    for name, model in models:
        for scaler_name, scaler in scalers:
            pipe = Pipeline([("scaler", scaler), ("model", model)])

            print(f"{name} with {scaler_name}")

            # cross_val_score : mae, mape
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
                    "top_features": feature_importance_finder(model, X_train),
                }
            )

    return pd.DataFrame(scores)


def plot_results(scores):
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    sns.barplot(data=scores, x="model", y="mae", hue="scaler", ax=ax[0])
    sns.barplot(data=scores, x="model", y="mape", hue="scaler", ax=ax[1])
    sns.barplot(data=scores, x="model", y="r2", hue="scaler", ax=ax[2])
    plt.show()


def plot_diff(scores):
    """diff의 mean, std를 통한 diff 분포를 시각화
    1) boxplot
    2) histplot
    3) scatterplot
    """

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    sns.boxplot(data=scores, x="model", y="mean_diff", ax=ax[0])
    sns.histplot(data=scores, x="mean_diff", kde=True, ax=ax[1])
    sns.scatterplot(data=scores, x="mean_diff", y="std_diff", hue="model", ax=ax[2])
    plt.show()


def sum_rank(data):
    """scores의 mae, mape, mean_diff, diff_abs의 순위를 합친 rank 컬럼을 추가, 최적의 모델 탐색"""
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
    # lightgbm의 출력문을 제거
    warnings.filterwarnings(action="ignore")

    start = time.time()
    data = pd.read_csv("../DATA/raw_2023051820231018_경대기업맞춤형.csv")
    train_data, oct_data = preps(data)
    train_data_2_4 = scale_2_4(train_data)
    oct_data_2_4 = scale_2_4(oct_data)

    X_train = train_data_2_4.drop("scale_pv", axis=1)
    y_train = train_data_2_4["scale_pv"]
    X_test = oct_data_2_4.drop("scale_pv", axis=1)
    y_test = oct_data_2_4["scale_pv"]

    print("Start Modeling")
    scores = basic_modeling(X_train, y_train, X_test, y_test)
    print(scores)
    # plot_results(scores)
    # plot_diff(scores)

    # 경과 시간 확인
    print("End Modeling")
    print("Elapsed Time: ", time.time() - start)
