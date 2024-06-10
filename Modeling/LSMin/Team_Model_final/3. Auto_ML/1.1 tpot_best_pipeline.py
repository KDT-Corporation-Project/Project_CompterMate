import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('C:/Users/KDP/Desktop/승민_기업프로젝트/Project_CompterMate/Modeling/LSMin/Team_Model_final/0. Data/2. output/1. Train_data.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('scale_pv', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['scale_pv'], random_state=42)

# Average CV score on the training set was: -0.0010311788099689149
exported_pipeline = ExtraTreesRegressor(bootstrap=False, max_features=0.8500000000000001, min_samples_leaf=2, min_samples_split=5, n_estimators=100)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
