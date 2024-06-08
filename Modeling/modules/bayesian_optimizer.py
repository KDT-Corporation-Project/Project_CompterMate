import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

from bayes_opt import BayesianOptimization

def bayes_SVR_RFR(X_train, y_train):
    """Bayesian optimization for SVR and RFR
    """
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


    def optimize_svr(X_train, y_train):
        """Bayesian optimization for SVR
        Optimize range:
        - C: 10^-3 ~ 10^2
        - gamma: 10^-4 ~ 10^-1
        """
        def svr_crossval(expC, expGamma):
            C = 10 ** expC
            gamma = 10 ** expGamma
            return svr_cv(C=C, gamma=gamma, data=X_train, targets=y_train)

        optimizer = BayesianOptimization(
            f=svr_crossval,
            pbounds={"expC": (-3, 2), "expGamma": (-4, -1)},
            random_state=1234,
            verbose=2
        )
        optimizer.maximize(n_iter=10)
        model = SVR(C=10 ** optimizer.max['params']['expC'], gamma=10 ** optimizer.max['params']['expGamma'])
        
        return model
    
    def optimize_rfr(X_train, y_train):
        def rfr_crossval(n_estimators, min_samples_split, max_features):
            return rfr_cv(
                n_estimators=int(n_estimators),
                min_samples_split=int(min_samples_split),
                max_features=max(min(max_features, 0.999), 1e-3),
                data=X_train,
                targets=y_train,
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
        model = RFR(
            n_estimators=int(optimizer.max['params']['n_estimators']),
            min_samples_split=int(optimizer.max['params']['min_samples_split']),
            max_features=optimizer.max['params']['max_features'],
            random_state=2
        )
        return model
    
    svr_model = optimize_svr(X_train, y_train)
    rfr_model = optimize_rfr(X_train, y_train)
    
    return svr_model, rfr_model



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

def bayes_scores(X_train, y_train, X_test, y_test):
    pass
    # 일단 주석처리: score를 뽑지 않고 model parameter만 전달

    # svr_result, rfr_result = bayes_SVR_RFR(X_train, y_train)
    
    # scores = []
    
    # # SVR : {'target': -0.008756335568622152, 'params': {'expC': -3.0, 'expGamma': -1.9168548948817559}}
    # # RFR : {'target': -0.008957846601163784, 'params': {'max_features': 0.21527920967949887, 'min_samples_split': 24.957028076514515, 'n_estimators': 230.47937054929565}}
    
    # for name, result in zip(['SVR', 'RFR'], [svr_result, rfr_result]):
    #     if name == 'SVR':
    #         model = SVR(C=10 ** result['params']['expC'], gamma=10 ** result['params']['expGamma'])
    #     else:
    #         model = RFR(
    #             n_estimators=int(result['params']['n_estimators']),
    #             min_samples_split=int(result['params']['min_samples_split']),
    #             max_features=result['params']['max_features'],
    #             random_state=2
    #         )
        
    #     mae_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    #     mape_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
        
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
    #     diff = y_test - y_pred
        
    #     scores.append({
    #         'model': name,
    #         'mae': mae_scores.mean() * (-1),
    #         'mean_diff': diff.mean() * (-100),
    #         'std_diff': diff.std(),
    #         'min_diff': diff.min(),
    #         '25%_diff': diff.quantile(0.25),
    #         '50%_diff': diff.quantile(0.5),
    #         '75%_diff': diff.quantile(0.75),
    #         'max_diff': diff.max(),
    #         'top_features': feature_importance_finder(model, X_train)
    #     })
    
    # return pd.DataFrame(scores)

if __name__ == "__main__":
    pass
    # scores = bayes_scores(X_train, y_train, X_test, y_test)
    # print(scores)
    # plot_results(scores)
    # plot_diff(scores)