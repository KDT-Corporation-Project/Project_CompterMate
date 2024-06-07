import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge, ElasticNet, SGDRegressor, ARDRegression, HuberRegressor, Lars, LassoLars, LassoLarsIC, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, PassiveAggressiveRegressor, Perceptron, RANSACRegressor, TheilSenRegressor, TweedieRegressor, PoissonRegressor, GammaRegressor, GeneralizedLinearRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score



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
    """data[(data["scale_pv"] < 5) & (data["scale_pv"] > 2)]
    """
    return data[(data["scale_pv"] < 5) & (data["scale_pv"] > 2)]


def feature_importance_finder(model, data):
    # 모델의 최고 3개 feature_importance의 이름을 반환
    if hasattr(model, 'feature_importances_'):
        top_features = data.columns[np.argsort(model.feature_importances_)[::-1][:3]]
        return top_features
    elif hasattr(model, 'coef_'):
        top_features = data.columns[np.argsort(model.coef_)[::-1][:3]]
        return top_features
    else:
        return 'None'


def basic_modeling(X_train, y_train, X_test, y_test):
    """StandardScaler를 통한 linear regression, RandomForest, LGBMRegressor Modeling
    """
    models = [
        ('LinearRegression', LinearRegression()),
        ('RandomForestRegressor', RandomForestRegressor()),
        ('LGBMRegressor', LGBMRegressor())
    ]
    
    scalers = [
        ('StandardScaler', StandardScaler()),
        ('MinMaxScaler', MinMaxScaler()),
        ('RobustScaler', RobustScaler())
    ]
    
    scores = []
    
    for name, model in models:
        for scaler_name, scaler in scalers:
            pipe = Pipeline([
                ('scaler', scaler),
                ('model', model)
            ])
            
            mae_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            mape_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='neg_mean_absolute_percentage_error')
            r2_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2')
            
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            diff = y_test - y_pred
            
            scores.append({
                'model': name,
                'scaler': scaler_name,
                'mae': mae_scores.mean(),
                'mape': mape_scores.mean(),
                'r2': r2_scores.mean(),
                'mean_diff': diff.mean(),
                'std_diff': diff.std(),
                'min_diff': diff.min(),
                '25%_diff': diff.quantile(0.25),
                '50%_diff': diff.quantile(0.5),
                '75%_diff': diff.quantile(0.75),
                'max_diff': diff.max(),
                'top_features': feature_importance_finder(pipe, X_train)
            })


def model_selection(data):
    
    # 모델 리스트
    models = [
        ('LinearRegression', LinearRegression()),
        ('Lasso', Lasso()),
        ('Ridge', Ridge()),
        ('BayesianRidge', BayesianRidge()),
        ('ElasticNet', ElasticNet()),
        # ('SGDRegressor', SGDRegressor()),
        ('RandomForestRegressor', RandomForestRegressor()),
        ('GradientBoostingRegressor', GradientBoostingRegressor()),
        ('AdaBoostRegressor', AdaBoostRegressor()),
        ('LGBMRegressor', LGBMRegressor()),
        ('ARDRegression', ARDRegression()),
        ('HuberRegressor', HuberRegressor()),
        ('Lars', Lars()),
        ('LassoLars', LassoLars()),
        ('LassoLarsIC', LassoLarsIC()),
        ('OrthogonalMatchingPursuit', OrthogonalMatchingPursuit()),
        ('OrthogonalMatchingPursuitCV', OrthogonalMatchingPursuitCV()),
        ('PassiveAggressiveRegressor', PassiveAggressiveRegressor()),
        ('Perceptron', Perceptron()),
        ('RANSACRegressor', RANSACRegressor()),
        ('TheilSenRegressor', TheilSenRegressor()),
        ('TweedieRegressor', TweedieRegressor()),
        ('PoissonRegressor', PoissonRegressor()),
        ('GammaRegressor', GammaRegressor()),
        ('GeneralizedLinearRegressor', GeneralizedLinearRegressor())
    ]
    
    scalers = [
        ('StandardScaler', StandardScaler()),
        ('MinMaxScaler', MinMaxScaler()),
        ('RobustScaler', RobustScaler())
    ]
    
    
    scores = []
    
    for name, model in models:
        # Scaler별 pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
    

        
    
    
    
    





if __name__ == "__main__":
    data = pd.read_csv("../DATA/raw_2023051820231018_경대기업맞춤형.csv")
    train_data, oct_data = preps(data)
    train_data_2_4 = scale_2_4(train_data)
    oct_data_2_4 = scale_2_4(oct_data)
    
    print(data.head())
