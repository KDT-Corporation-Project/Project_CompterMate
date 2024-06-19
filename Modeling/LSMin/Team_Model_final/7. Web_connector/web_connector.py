import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def preprocess_data(data):
    """
    데이터 전처리를 수행하는 함수
    :param data: 원본 데이터프레임
    :return: 전처리된 데이터프레임
    """
    data = data.drop(columns=['s_temp_sv', 'c_temp_sv', 'n_temp_sv', 'k_rpm_sv'])
    data['time'] = pd.to_datetime(data['time']).dt.tz_localize(None)
    data = data[data['E_scr_sv'] == 8]
    data = data.drop(columns=['E_scr_sv'])
    
    data = data[data['scale_pv'] < 600]
    data = data[data['scale_pv'] < 4]
    data = data[data['c_temp_pv'] >= 68]
    data = data.drop(columns=['E_scr_pv'])
    data = data[data['k_rpm_pv'] >= 50]
    data = data[data['scale_pv'] > 2]
    
    return data

def load_and_preprocess_data(file_path, start_date, end_date):
    """
    데이터를 로드하고 전처리하는 함수
    :param file_path: 데이터 파일 경로
    :param start_date: 데이터 필터링 시작 날짜
    :param end_date: 데이터 필터링 종료 날짜
    :return: 전처리된 트레이닝 데이터와 테스트 데이터
    """
    data = pd.read_csv(file_path)
    data = data.drop(columns=['Unnamed: 12'])
    data['time'] = pd.to_datetime(data['time']).dt.tz_localize(None)

    selected_data = data[(data['time'] >= start_date) & (data['time'] <= end_date)]
    oct_1 = pd.Timestamp('2023-10-01')
    after_oct_data = data[data['time'] >= oct_1]
    
    train_data = preprocess_data(selected_data)
    test_data = preprocess_data(after_oct_data)
    
    return train_data, test_data

def train_model(train_data):
    """
    모델을 학습하고 검증하는 함수
    :param train_data: 전처리된 트레이닝 데이터
    :return: 학습된 모델들, 피처 스케일러, 타겟 스케일러
    """
    # 중복값 제거
    train_data.drop_duplicates(inplace=True)

    # 피처와 타겟 분리
    X = train_data.drop(columns=['scale_pv', 'time'])  # 학습에 사용할 때 'time' 컬럼 제거
    y = train_data['scale_pv']

    # 학습 데이터와 검증 데이터 분리
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    # 타겟 스케일링
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_valid_scaled = target_scaler.transform(y_valid.values.reshape(-1, 1))

    # 모델 학습 및 평가 함수
    def train_and_evaluate_model(model, X_train, X_valid, y_train_scaled, y_valid_scaled, target_scaler):
        """
        주어진 모델을 학습하고 평가하는 함수
        :param model: 학습할 모델
        :param X_train: 학습 데이터
        :param X_valid: 검증 데이터
        :param y_train_scaled: 스케일링된 학습 타겟
        :param y_valid_scaled: 스케일링된 검증 타겟
        :param target_scaler: 타겟 스케일러
        :return: 학습 및 검증 성능 지표
        """
        model.fit(X_train, y_train_scaled.ravel())
        y_train_pred_scaled = model.predict(X_train)
        y_valid_pred_scaled = model.predict(X_valid)

        # 역스케일링
        y_train_pred = target_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1))
        y_valid_pred = target_scaler.inverse_transform(y_valid_pred_scaled.reshape(-1, 1))
        y_train_original = target_scaler.inverse_transform(y_train_scaled)
        y_valid_original = target_scaler.inverse_transform(y_valid_scaled)

        train_mae = mean_absolute_error(y_train_original, y_train_pred)
        valid_mae = mean_absolute_error(y_valid_original, y_valid_pred)
        train_mape = mean_absolute_percentage_error(y_train_original, y_train_pred)
        valid_mape = mean_absolute_percentage_error(y_valid_original, y_valid_pred)

        return train_mae, valid_mae, train_mape, valid_mape, y_train_pred, y_valid_pred

    # Linear Regression - GridSearchCV
    lr_param_grid = {
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'n_jobs': [None, -1],
        'positive': [True, False]
    }
    lr_grid_search = GridSearchCV(LinearRegression(), lr_param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    lr_grid_search.fit(X_train_scaled, y_train_scaled.ravel())
    lr_best_model = lr_grid_search.best_estimator_

    # # Random Forest - GridSearchCV
    # rf_param_grid = {
    #     'n_estimators': [50, 100, 200],
    #     'max_features': ['sqrt', 'log2', None],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'bootstrap': [True, False]
    # }
    # rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    # rf_grid_search.fit(X_train_scaled, y_train_scaled.ravel())
    # rf_best_model = rf_grid_search.best_estimator_

    # # LightGBM - GridSearchCV
    # lgb_param_grid = {
    #     'num_leaves': [31, 50, 100],
    #     'learning_rate': [0.01, 0.1, 0.3],
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [-1, 10, 20, 30]
    # }
    # lgb_grid_search = GridSearchCV(lgb.LGBMRegressor(random_state=42), lgb_param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    # lgb_grid_search.fit(X_train_scaled, y_train_scaled.ravel())
    # lgb_best_model = lgb_grid_search.best_estimator_

    # 모델 학습 및 평가
    lr_train_mae, lr_valid_mae, lr_train_mape, lr_valid_mape, lr_y_train_pred, lr_y_valid_pred = train_and_evaluate_model(lr_best_model, X_train_scaled, X_valid_scaled, y_train_scaled, y_valid_scaled, target_scaler)
    #rf_train_mae, rf_valid_mae, rf_train_mape, rf_valid_mape, rf_y_train_pred, rf_y_valid_pred = train_and_evaluate_model(rf_best_model, X_train_scaled, X_valid_scaled, y_train_scaled, y_valid_scaled, target_scaler)
    #lgb_train_mae, lgb_valid_mae, lgb_train_mape, lgb_valid_mape, lgb_y_train_pred, lgb_y_valid_pred = train_and_evaluate_model(lgb_best_model, X_train_scaled, X_valid_scaled, y_train_scaled, y_valid_scaled, target_scaler)

    # 결과 출력
    print(f"Linear Regression - Train MAE: {lr_train_mae}, Train MAPE: {lr_train_mape*100}")
    print(f"Linear Regression - Valid MAE: {lr_valid_mae}, Valid MAPE: {lr_valid_mape*100}")
    print()
    # print(f"Random Forest - Train MAE: {rf_train_mae}, Train MAPE: {rf_train_mape*100}")
    # print(f"Random Forest - Valid MAE: {rf_valid_mae}, Valid MAPE: {rf_valid_mape*100}")
    # print()
    # print(f"LightGBM - Train MAE: {lgb_train_mae}, Train MAPE: {lgb_train_mape*100}")
    # print(f"LightGBM - Valid MAE: {lgb_valid_mae}, Valid MAPE: {lgb_valid_mape*100}")

    # 가장 낮은 Valid MAPE 값을 가진 모델 선택
    best_model = min(
        (lr_best_model, lr_valid_mape)#, (rf_best_model, rf_valid_mape), (lgb_best_model, lgb_valid_mape)],
        #key=lambda x: x[1]
    )[0]

    # scaler_data 정의
    scaler_data = {
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'target_scaler': target_scaler
    }

    return lr_best_model, rf_best_model, lgb_best_model, best_model, scaler_data

def final_evaluate_model(model, X_test_scaled, y_test_scaled, target_scaler):
    """
    최종 모델을 평가하는 함수
    :param model: 학습된 모델
    :param X_test_scaled: 스케일링된 테스트 피처
    :param y_test_scaled: 스케일링된 테스트 타겟
    :param target_scaler: 타겟 스케일러
    :return: 테스트 성능 지표
    """
    y_test_pred_scaled = model.predict(X_test_scaled)

    # 역스케일링
    y_test_pred = target_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1))
    y_test_original = target_scaler.inverse_transform(y_test_scaled)

    test_mae = mean_absolute_error(y_test_original, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test_original, y_test_pred)

    return test_mae, test_mape, y_test_pred

def evaluate_test_data(test_data, best_model, scaler, target_scaler):
    """
    테스트 데이터를 평가하는 함수
    :param test_data: 전처리된 테스트 데이터
    :param best_model: 최적의 모델
    :param scaler: 피처 스케일러
    :param target_scaler: 타겟 스케일러
    """
    # 피처와 타겟 분리
    X_test_final = test_data.drop(columns=['scale_pv', 'time'])  # 학습에 사용할 때 'time' 컬럼 제거
    y_test_final = test_data['scale_pv']

    # 테스트 데이터 스케일링
    X_test_final_scaled = scaler.transform(X_test_final)
    y_test_final_scaled = target_scaler.transform(y_test_final.values.reshape(-1, 1))

    # 최종 평가 결과
    test_mae, test_mape, y_test_pred = final_evaluate_model(best_model, X_test_final_scaled, y_test_final_scaled, target_scaler)

    print(f"Final Test - Best Model MAE: {test_mae}, MAPE: {test_mape*100}")
    return test_mape

def predict_first_row(data, model, scaler, feature_names, target_scaler):
    """
    첫 번째 행에 대한 예측을 수행하는 함수
    :param data: 데이터프레임
    :param model: 학습된 모델
    :param scaler: 피처 스케일러
    :param feature_names: 피처 이름 목록
    :param target_scaler: 타겟 스케일러
    :return: 예측된 scale_pv 값
    """
    first_row = data.iloc[0][feature_names]
    features_scaled = scaler.transform(first_row.values.reshape(1, -1))
    predicted_scale_pv_scaled = model.predict(features_scaled)[0]
    predicted_scale_pv = target_scaler.inverse_transform([[predicted_scale_pv_scaled]])[0][0]
    return round(predicted_scale_pv, 2)  # 소수점 아래 둘째 자리로 반올림

def find_optimal_k_rpm_pv(n_temp, s_temp, c_temp, current_k_rpm_pv, model, scaler, feature_names, target_scaler):
    """
    최적의 k_rpm_pv 값을 찾는 함수
    :param n_temp: n_temp 값
    :param s_temp: s_temp 값
    :param c_temp: c_temp 값
    :param current_k_rpm_pv: 현재 k_rpm_pv 값
    :param model: 학습된 모델
    :param scaler: 피처 스케일러
    :param feature_names: 피처 이름 목록
    :param target_scaler: 타겟 스케일러
    :return: 최적의 k_rpm_pv 값
    """
    best_k_rpm_pv = current_k_rpm_pv
    best_scale_pv_diff = float('inf')
    for k_rpm_adjustment in [-1, 0, 1]:  # -1, 0, 1
        k_rpm_pv = current_k_rpm_pv + k_rpm_adjustment
        features = pd.DataFrame([[n_temp, s_temp, c_temp, k_rpm_pv]], columns=feature_names)
        features_scaled = scaler.transform(features)
        predicted_scale_pv_scaled = model.predict(features_scaled)[0]
        predicted_scale_pv = target_scaler.inverse_transform([[predicted_scale_pv_scaled]])[0][0]
        scale_pv_diff = abs(predicted_scale_pv - 3)
        if scale_pv_diff < best_scale_pv_diff:
            best_scale_pv_diff = scale_pv_diff
            best_k_rpm_pv = k_rpm_pv
    return best_k_rpm_pv

def predict_and_optimize(data, model, scaler, target_scaler, feature_names):
    """
    전체 데이터에 대해 예측 및 최적 k_rpm_pv 값을 찾는 함수
    :param data: 데이터프레임
    :param model: 학습된 모델
    :param scaler: 피처 스케일러
    :param target_scaler: 타겟 스케일러
    :param feature_names: 피처 이름 목록
    :return: 예측 및 최적화 결과 데이터프레임
    """
    results = []
    for i in range(len(data)):
        row = data.iloc[i]
        n_temp, s_temp, c_temp, k_rpm_pv = row['n_temp_pv'], row['s_temp_pv'], row['c_temp_pv'], row['k_rpm_pv']
        if i == 0:
            predicted_scale_pv = predict_first_row(pd.DataFrame([row]), model, scaler, feature_names, target_scaler)
        else:
            features = pd.DataFrame([[n_temp, s_temp, c_temp, k_rpm_pv]], columns=feature_names)
            features_scaled = scaler.transform(features)
            predicted_scale_pv_scaled = model.predict(features_scaled)[0]
            predicted_scale_pv = target_scaler.inverse_transform([[predicted_scale_pv_scaled]])[0][0]
            predicted_scale_pv = round(predicted_scale_pv, 2)

        results.append({
            'n_temp_pv': n_temp,
            's_temp_pv': s_temp,
            'c_temp_pv': c_temp,
            'predicted_scale_pv': predicted_scale_pv,
            'k_rpm_pv': k_rpm_pv
        })

        if i < len(data) - 1:
            current_k_rpm_pv = k_rpm_pv
            optimal_k_rpm_pv = find_optimal_k_rpm_pv(n_temp, s_temp, c_temp, current_k_rpm_pv, model, scaler, feature_names, target_scaler)
            data.at[i + 1, 'k_rpm_pv'] = optimal_k_rpm_pv
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    while True:
        start_date_str = input("Enter the start date (YYYY-MM-DD): ")
        end_date_str = input("Enter the end date (YYYY-MM-DD): ")

        try:
            start_date = pd.to_datetime(start_date_str)
            end_date = pd.to_datetime(end_date_str)
            if start_date < pd.Timestamp('2023-05-18') or end_date > pd.Timestamp('2023-09-26'):
                print('Invalid date range. Please select a range between 2023-05-18 and 2023-09-26.')
                continue
            break
        except ValueError:
            print("Invalid date format. Please enter the date in YYYY-MM-DD format.")
            continue

    file_path = 'C:/Users/KDP/Desktop/승민_기업프로젝트/Project_CompterMate/Modeling/LSMin/Team_Model_final/0. Data/1. original/공정데이터.csv'

    # 데이터 로딩 및 전처리
    train_data, test_data = load_and_preprocess_data(file_path, start_date, end_date)

    # 기존 모델의 MAPE 값
    initial_mape = 0.7737
    print(f"Initial Test MAPE: {initial_mape:.4f}")

    # 모델 학습
    lr_best_model, rf_best_model, lgb_best_model, best_model, scaler_data = train_model(train_data)

    # 테스트 데이터 평가
    new_test_mape = evaluate_test_data(test_data, best_model, scaler_data['scaler'], scaler_data['target_scaler'])

    print(f"New Test MAPE: {100*new_test_mape:.4f}")

    # 사용자에게 모델 선택 요청
    user_choice = input(f"Select the model to use (initial or new): ")

    if user_choice.lower() == 'new' and new_test_mape < initial_mape:
        print("Using new model for prediction and optimization.")
        selected_model = best_model
        selected_mape = new_test_mape
    else:
        print("Using initial model for prediction and optimization.")
        selected_model = best_model

    # 예측 및 최적화 실행 (선택된 모델 사용)
    results = predict_and_optimize(test_data, selected_model, scaler_data['scaler'], scaler_data['target_scaler'], scaler_data['feature_names'])

    # DF에 기존 sv값 컬럼 추가
    results[['c_temp_sv', 's_temp_sv', 'n_temp_sv', 'k_rpm_sv', 'E_scr_pv', 'E_scr_sv']] = (70, 70, 70, 180, 8, 8)

    # 새로운 컬럼 추가
    results['c_temp_pv_dif'] = results['c_temp_pv'].diff().fillna(0)
    results['s_temp_pv_dif'] = results['s_temp_pv'].diff().fillna(0)
    results['n_temp_pv_dif'] = results['n_temp_pv'].diff().fillna(0)
    results['scale_dif'] = results['predicted_scale_pv'].diff().fillna(0)
    results['rpm_dif'] = results['k_rpm_pv'].diff().fillna(0)
    results['loss'] = results['scale_dif']

    # 결과를 CSV 파일로 저장
    results.to_csv('C:/Users/KDP/Desktop/승민_기업프로젝트/Project_CompterMate/Modeling/LSMin/Team_Model_final/7. Web_connector/simulation.csv', index=False)

    # 실제 scale_pv와 예측된 predicted_scale_pv를 포함하는 새로운 데이터프레임 생성
    final_results = pd.DataFrame({
        'actual_scale_pv': test_data['scale_pv'],
        'predicted_scale_pv': results['predicted_scale_pv']
    })

    # 결과를 CSV 파일로 저장
    final_results.to_csv('C:/Users/KDP/Desktop/승민_기업프로젝트/Project_CompterMate/Modeling/LSMin/Team_Model_final/7. Web_connector/real&pred_scale.csv', index=False)
