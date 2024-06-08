import pandas as pd
import numpy as np
# warning 제거
import warnings
warnings.filterwarnings(action='ignore')

class RPMCalculator:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    @staticmethod
    def set_dataframe(data):
        """Set the dataframe for RPM calculation
        1) Filter data
        2) Initialize new columns
         - scale_pv, c_temp_pv_dif, s_temp_pv_dif, n_temp_pv_dif, k_rpm_pv
        3) Select relevant columns

        Args:
            data (_type_): _description_

        Returns:
            data (DataFrame): DataFrame with columns ['c_temp_pv', 'k_rpm_pv', 'n_temp_pv', 's_temp_pv', 'scale_pv', 'c_temp_pv_dif', 's_temp_pv_dif', 'n_temp_pv_dif', 'c_temp_sv', 's_temp_sv', 'n_temp_sv', 'k_rpm_sv', 'E_scr_pv', 'E_scr_sv', 'scale_dif', 'rpm_dif', 'loss']
        """
        # Filter data and initialize new columns
        data = data[(data['scale_pv'] > 2) & (data['scale_pv'] < 4)]
        data['scale_pv'] = 0
        data['c_temp_pv_dif'] = data['c_temp_pv'].diff().fillna(0)
        data['s_temp_pv_dif'] = data['s_temp_pv'].diff().fillna(0)
        data['n_temp_pv_dif'] = data['n_temp_pv'].diff().fillna(0)
        data['k_rpm_pv'] = 0
        data.iloc[0, data.columns.get_loc('k_rpm_pv')] = 168
        data['c_temp_sv'] = 70
        data['s_temp_sv'] = 70
        data['n_temp_sv'] = 70
        data['k_rpm_sv'] = 180
        data['E_scr_pv'] = 8
        data['E_scr_sv'] = 8
        data['scale_dif'] = 0
        data['rpm_dif'] = 0
        data['loss'] = 0

        # Select relevant columns
        columns_to_keep = ['c_temp_pv', 'k_rpm_pv', 'n_temp_pv', 's_temp_pv', 'scale_pv', 
                           'c_temp_pv_dif', 's_temp_pv_dif', 'n_temp_pv_dif', 
                           'c_temp_sv', 's_temp_sv', 'n_temp_sv', 
                           'k_rpm_sv', 'E_scr_pv', 'E_scr_sv', 
                           'scale_dif', 'rpm_dif', 'loss']
        return data[columns_to_keep]

    def calculate_rpm(self, data):
        """Calculate RPM for each row in the dataframe
        1) Predict the scale_pv
        2) Calculate scale_dif and rpm_dif
        3) Calculate loss
        4) Calculate next k_rpm_pv
        5) Update dataframe
        6) Repeat for the next row

        Args:
            data (DataFrame): DataFrame with columns ['c_temp_pv', 'k_rpm_pv', 'n_temp_pv', 's_temp_pv', 'scale_pv']
        """
        for row_num in range(len(data)):
            # Predict the scale_pv for one row
            X = data.iloc[row_num, :4].values
            X = self.scaler.transform(X.reshape(1, -1))
            pred_scale = self.model.predict(X)
            
            print(pred_scale)
            break
        
            # Calculate scale_dif and rpm_dif
            scale_dif = pred_scale - 3
            if scale_dif > 0.05:
                rpm_dif = 1
            elif scale_dif < -0.05:
                rpm_dif = -1
            else:
                rpm_dif = 0

            # Calculate loss
            loss = max(0, pred_scale - 3)
            loss = np.round(loss, 3)

            # Calculate next k_rpm_pv
            if data.loc[row_num, 'k_rpm_pv'] > 210:
                next_k_rpm_pv = 210
            elif data.loc[row_num, 'k_rpm_pv'] < 50:
                next_k_rpm_pv = 50
            else:
                next_k_rpm_pv = data.loc[row_num, 'k_rpm_pv'] + rpm_dif

            # Update dataframe
            data.loc[row_num, 'scale_pv'] = pred_scale
            data.loc[row_num, 'scale_dif'] = scale_dif
            data.loc[row_num, 'rpm_dif'] = rpm_dif
            data.loc[row_num, 'loss'] = loss
            if row_num < len(data) - 1:
                data.loc[row_num + 1, 'k_rpm_pv'] = next_k_rpm_pv
                
        return data

if __name__ == '__main__':
    # Example usage
    data = pd.read_csv('../DATA/raw_2023051820231018_경대기업맞춤형.csv')
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from model_selection import Scale_Prediction as sp
    model = RandomForestRegressor()
    scaler = StandardScaler()
    
    train_data, test_data = sp().preps(data)
    X_train, y_train = train_data.drop('scale_pv', axis=1), train_data['scale_pv']
    X_test, y_test = test_data.drop('scale_pv', axis=1), test_data['scale_pv']
    
    X_train = scaler.fit_transform(X_train)
    model.fit(X_train, y_train)

    print("rpm calculator")
    rpm_calculator = RPMCalculator(model, scaler)

    # Preprocess data
    data = rpm_calculator.set_dataframe(test_data).reset_index(drop=True)

    # Execute the RPM calculation function
    print("calculate RPM")
    rpm_calculator.calculate_rpm(data)