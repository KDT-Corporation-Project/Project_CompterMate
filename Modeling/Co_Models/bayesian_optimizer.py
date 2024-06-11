import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

from bayes_opt import BayesianOptimization

def bayes_SVR_RFR(X_train, y_train):
    def svr_cv(C, gamma, data, targets):
        """cross validation for SVR
        """
        estimator = SVR(C=C, gamma=gamma)
        cval = cross_val_score(estimator, data, targets, scoring='neg_mean_absolute_percentage_error', cv=4)
        return cval.mean()

    def rfr_cv(n_estimators, min_samples_split, max_features, data, targets):
        """cross validation for RFR
        """
        estimator = RFR(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=max(min(max_features, 0.999), 1e-3),
            random_state=2
        )
        cval = cross_val_score(estimator, data, targets, scoring='neg_mean_absolute_percentage_error', cv=4)
        return cval.mean()

    def optimize_svr(data, targets):
        """Bayesian optimization for SVR
        Optimize range:
        - C: 10^-3 ~ 10^2
        - gamma: 10^-4 ~ 10^-1
        """
        def svr_crossval(expC, expGamma):
            C = 10 ** expC
            gamma = 10 ** expGamma
            return svr_cv(C=C, gamma=gamma, data=data, targets=targets)

        optimizer = BayesianOptimization(
            f=svr_crossval,
            pbounds={"expC": (-3, 2), "expGamma": (-4, -1)},
            random_state=1234,
            verbose=2
        )
        optimizer.maximize(n_iter=10)
        
        return optimizer.max
    

    def optimize_rfr(data, targets):
        def rfr_crossval(n_estimators, min_samples_split, max_features):
            return rfr_cv(
                n_estimators=int(n_estimators),
                min_samples_split=int(min_samples_split),
                max_features=max(min(max_features, 0.999), 1e-3),
                data=data,
                targets=targets,
            )

        optimizer = BayesianOptimization(
            f=rfr_crossval,
            pbounds={
                "n_estimators": (10, 500),
                "min_samples_split": (2, 25),
                "max_features": (0.1, 0.999),
            },
            random_state=1234,
            verbose=2
        )
        optimizer.maximize(n_iter=10)
        
        return optimizer.max
        
    svr_result = optimize_svr(X_train, y_train)
    rfr_result = optimize_rfr(X_train, y_train)
    
    return svr_result, rfr_result

def bayes_scores(X_train, y_train, X_test, y_test):
    svr_result, rfr_result = bayes_SVR_RFR(X_train, y_train)
    
    scores = []
    
    # SVR : {'target': -0.008756335568622152, 'params': {'expC': -3.0, 'expGamma': -1.9168548948817559}}
    # RFR : {'target': -0.008957846601163784, 'params': {'max_features': 0.21527920967949887, 'min_samples_split': 24.957028076514515, 'n_estimators': 230.47937054929565}}
    
    for name, result in zip(['SVR', 'RFR'], [svr_result, rfr_result]):
        if name == 'SVR':
            model = SVR(C=10 ** result['params']['expC'], gamma=10 ** result['params']['expGamma'])
        else:
            model = RFR(
                n_estimators=int(result['params']['n_estimators']),
                min_samples_split=int(result['params']['min_samples_split']),
                max_features=result['params']['max_features'],
                random_state=2
            )
        
        mae_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        mape_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
        r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        diff = y_test - y_pred
        
        scores.append({
            'model': name,
            'mae': mae_scores.mean(),
            'mape': mape_scores.mean(),
            'r2': r2_scores.mean(),
            'mean_diff': diff.mean(),
            'std_diff': diff.std(),
            'min_diff': diff.min(),
            '25%_diff': diff.quantile(0.25),
            '50%_diff': diff.quantile(0.5),
            '75%_diff': diff.quantile(0.75),
            'max_diff': diff.max()
        })
    
    return pd.DataFrame(scores)