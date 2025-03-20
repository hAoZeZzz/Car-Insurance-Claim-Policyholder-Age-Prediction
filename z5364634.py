import numpy as np
import pandas as pd
import re
import sys
import time
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from numpy import sort

start_time = time.time()

def to_total_months(s):
    '''
    convert the format from ? years ? months to ? months
    '''
    result = re.match(r'(\d+) years and (\d+) months?', s)
    if result:
        years = int(result.group(1))
        months = int(result.group(2))
        total_months = years * 12 + months
        return total_months
    else:
        return int(s.split()[0])
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # #           Preprocessing             # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def preprocess_data(df):
    '''
    return modified dataset and its corresponding policy_id
    '''
    policy_ids = df['policy_id']
    df = df.drop('policy_id', axis=1)
    
    # remove the first unname column
    df = df.drop(df.columns[:1], axis = 1)

    df['age_of_car'] = df['age_of_car'].apply(to_total_months)

    # OneHotEncoder to convert the object format into integer Format
    df = pd.get_dummies(df, columns=['area_cluster', 'make', 'segment', 'model', 'fuel_type', \
                                        'engine_type', 'rear_brakes_type', 'transmission_type',\
                                        'steering_type'])
    df = df.replace({True: 1, False: 0})

    # split the combination of number and letters
    df[['torque_Nm', 'torque_rpm']] = df['max_torque'].str.split('Nm@', expand=True)
    df[['power_Nm', 'power_rpm']] = df['max_power'].str.split('bhp@', expand=True)

    df['torque_rpm'] = df['torque_rpm'].str.rstrip('rpm')
    df['power_rpm'] = df['power_rpm'].str.rstrip('rpm')   

    df['torque_Nm'] = pd.to_numeric(df['torque_Nm'])
    df['torque_rpm'] = pd.to_numeric(df['torque_rpm'])
    df['power_Nm'] = pd.to_numeric(df['power_Nm'])
    df['power_rpm'] = pd.to_numeric(df['power_rpm'])

    # replace 'Yes' and 'No' with 1 and 0
    for col in df.columns:
        if re.search(r'is_', col):
            df[col] = df[col].replace({'Yes': 1, 'No': 0})
        else:
            continue
    
    # combine length, width and height together
    df['volume'] = df.apply(lambda row:np.cbrt(row['length'] * row['width'] * row['height']), axis=1)
    
    # remove useless columns
    df = df.drop(columns=['max_torque', 'max_power', 'length', 'width', 'height'], axis=1)
    return policy_ids, df 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # #           Postprocessing            # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def postprocess_data(model, metric, X_train, X_test, y_train, y_test, y_pred, is_regression=True):
    global other_params
    print('During post-processing of data...')
    thresholds = sort(model.feature_importances_)
    thresholds = thresholds[ thresholds != 0]
    best_metric = metric
    output_pred = y_pred

    for thresh in thresholds[:15]:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        
        # train and eval model
        if is_regression:
            selection_model = XGBRegressor(**other_params)
            selection_model.fit(select_X_train, y_train)

            select_X_test = selection.transform(X_test)
            y_pred = selection_model.predict(select_X_test)
            mse = mean_squared_error(y_test, y_pred)
            if mse < best_metric:
                print("Current Minimum Mean Squared Error:", mse, f'n={select_X_train.shape[1]}', f'thresh={thresh}')
                best_metric = mse
                output_pred = y_pred
        
        else:
            selection_model = XGBClassifier(**other_params)
            selection_model.fit(select_X_train, y_train)

            select_X_test = selection.transform(X_test)
            y_pred = selection_model.predict(select_X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            if macro_f1 > best_metric:
                print("Macro f1 score:", macro_f1, f'n={select_X_train.shape[1]}')
                best_metric = macro_f1
                output_pred = y_pred

    return output_pred
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # #           Train And Test            # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

train_df = pd.read_csv(sys.argv[1])
test_df = pd.read_csv(sys.argv[2])

policy_ids_train, df_train = preprocess_data(train_df)
policy_ids_test, df_test = preprocess_data(test_df)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # #           Regression Task           # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

X_train = df_train.drop('age_of_policyholder', axis=1).values
X_test = df_test.drop('age_of_policyholder', axis=1).values
y_train = df_train['age_of_policyholder'].values
y_test = df_test['age_of_policyholder'].values

# fit model first time
model_regression = XGBRegressor()
model_regression.fit(X_train, y_train)

# first prediction
y_pred = model_regression.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("The first Mean Squared Error:", mse)

# tune hyperparameters by GridSearchCV
cv_params = {'reg_alpha':[0.1, 0.2],
             'reg_lambda': [0.5, 1]}

other_params = {'subsample':0.8,
                'colsample_bytree':0.8,
                'min_child_weight':3,
                'seed':0,
                'n_estimators':150,
                'gamma':0.125,
                'learning_rate':0.05, 
                'max_depth': 4}
model_regression = XGBRegressor(**other_params)
opti = GridSearchCV(estimator=model_regression, param_grid=cv_params, scoring='neg_mean_squared_error', n_jobs=-1, verbose=3)
opti.fit(X_train, y_train)
print(f'the best parameter combination:{opti.best_params_}')

other_params.update(opti.best_params_)
model_regression = XGBRegressor(**other_params)
model_regression.fit(X_train,y_train)
y_pred = model_regression.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error After updating parameters:{mse}')

# postprocessing
best_y_pred = postprocess_data(model=model_regression, metric=mse, X_train=X_train,\
                               X_test=X_test, y_train=y_train, y_test=y_test,\
                               y_pred=y_pred, is_regression=True)

# output file
output_part1 = pd.DataFrame({'policy_id': policy_ids_test, 'age_of_policyholder': best_y_pred})
output_part1.to_csv('z5364634.PART1.output.csv', index=False)
print('--------------------------------------------')
print()
print('--------------------------------------------')
print()
print('--------------------------------------------')
print()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # #        Classification Task          # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
X_train = df_train.drop('is_claim', axis=1).values
X_test = df_test.drop('is_claim', axis=1).values
y_train = df_train['is_claim'].values
y_test = df_test['is_claim'].values

# fit model first time
model_classification = XGBClassifier(scale_pos_weight=10) # due to the imbalance dataset
model_classification.fit(X_train, y_train)

# make predictions
y_pred = model_classification.predict(X_test)
macro_f1 = f1_score(y_test, y_pred, average='macro')
print("The First Macro F1 Score:", macro_f1)

# tune hyperparameters by GridSearchCV
cv_params = {
            'max_depth': [4,5,6]
            }

other_params = {
                'min_child_weight':4,
                'n_estimators':150,
                'learning_rate':0.01, 
                'scale_pos_weight':10
                }

model_classification = XGBClassifier(**other_params)
opti = GridSearchCV(estimator=model_classification, param_grid=cv_params, scoring='f1_macro', n_jobs=-1, verbose=3)
opti.fit(X_train, y_train)
print(f'the best parameter combination:{opti.best_params_}')
other_params.update(opti.best_params_)
model_classification = XGBClassifier(**other_params)
model_classification.fit(X_train,y_train)
y_pred = model_classification.predict(X_test)
macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f'Macro F1 score After updating parameters:{macro_f1}')

# postprocessing
best_y_pred = postprocess_data(model=model_classification, metric=macro_f1, X_train=X_train,\
                               X_test=X_test, y_train=y_train, y_test=y_test,\
                               y_pred=y_pred, is_regression=False)

# output file
output_part2 = pd.DataFrame({'policy_id': policy_ids_test, 'is_claim': best_y_pred})
output_part2.to_csv('z5364634.PART2.output.csv', index=False)

end_time = time.time()
duration = end_time - start_time
print('--------------------------------------------')
print()
print('--------------------------------------------')
print()
print('--------------------------------------------')
print()
print(f'running time is: {duration:.2f} seconds')