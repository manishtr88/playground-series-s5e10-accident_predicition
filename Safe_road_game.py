import random
import pandas as pd
from xgboost import XGBRegressor
import joblib
import streamlit as st


accident_model = XGBRegressor()
#Model Trained in Kaggle competition
accident_model.load_model('C:\\Pyhton\\playground-series-s5e10\\model.ubj')
scaler = joblib.load('C:\\Pyhton\\playground-series-s5e10\\scaler.pkl')

def accident_prediction(features):
    features_col = ['road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting',
       'weather', 'road_signs_present', 'public_road', 'time_of_day',
       'holiday', 'school_season', 'num_reported_accidents']

    feature_value = pd.DataFrame(features,index = features_col).T

    categorical_features = ['road_type', 'lighting', 'weather']
    uneffected_feature = ['num_lanes', 'time_of_day', 'school_season']
    final_features =['curvature', 'speed_limit', 'road_signs_present', 'public_road',
       'holiday', 'num_reported_accidents', 'road_type_highway',
       'road_type_rural', 'road_type_urban', 'lighting_daylight',
       'lighting_dim', 'lighting_night', 'weather_clear', 'weather_foggy',
       'weather_rainy']

    for variable in uneffected_feature:
        features_col.remove(variable)


    eval_value = feature_value[features_col].copy()

    if eval_value.loc[:,'road_type'].iloc[0]=='urban':
        eval_value.loc[:,'road_type_urban'] = True
        eval_value.loc[:,'road_type_highway'] = False
        eval_value.loc[:,'road_type_rural'] = False
    elif eval_value.loc[:,'road_type'].iloc[0]=='rural':
        eval_value.loc[:,'road_type_urban'] = False
        eval_value.loc[:,'road_type_highway'] = False
        eval_value.loc[:,'road_type_rural'] = True
    else:
        eval_value.loc[:,'road_type_urban'] = False
        eval_value.loc[:,'road_type_highway'] = True
        eval_value.loc[:,'road_type_rural'] = False

    if eval_value.loc[:,'lighting'].iloc[0] == 'daylight':
        eval_value.loc[:,'lighting_daylight'] = True
        eval_value.loc[:,'lighting_night'] = False
        eval_value.loc[:,'lighting_dim'] = False
    elif eval_value.loc[:,'lighting'].iloc[0] == 'night':
        eval_value.loc[:,'lighting_daylight'] = False
        eval_value.loc[:,'lighting_night'] = True
        eval_value.loc[:,'lighting_dim'] = False
    else:
        eval_value.loc[:,'lighting_daylight'] = False
        eval_value.loc[:,'lighting_night'] = False
        eval_value.loc[:,'lighting_dim'] = True

    if eval_value.loc[:,'weather'].iloc[0] == 'clear':
        eval_value.loc[:,'weather_foggy'] = False
        eval_value.loc[:,'weather_rainy'] = False
        eval_value.loc[:,'weather_clear'] = True
    elif eval_value.loc[:,'weather'].iloc[0] == 'foggy':
        eval_value.loc[:,'weather_foggy'] = True
        eval_value.loc[:,'weather_rainy'] = False
        eval_value.loc[:,'weather_clear'] = False
    else:
        eval_value['weather_foggy'] = False
        eval_value['weather_rainy'] = True
        eval_value['weather_clear'] = False
    for value in categorical_features:
        eval_value.drop(value,axis=1,inplace=True)


    eval_data = eval_value[final_features]


    numeric_cols = ['curvature', 'speed_limit', 'num_reported_accidents']
    raw_row = {}


    num_vals = [float(eval_data[c]) for c in numeric_cols]
    scaled = scaler.transform([num_vals])[0]
    for i, c in enumerate(numeric_cols):
        raw_row[c] = float(scaled[i])


    for c in final_features:
        if c in numeric_cols:
            continue
        raw_row[c] = 1 if str(eval_data[c]).lower() in ['true'] else 0

    eval_df = pd.DataFrame([raw_row], columns=final_features)
    accident_risk = accident_model.predict(eval_df)
    return accident_risk

def accident():
    if selection is not None:
        selected_option = options.loc[selection]
        chance = random.random()

        risk = accident_prediction(selected_option)
        if chance < risk:
            st.success(f'You met with Accident! ')
        else:
            st.success(f'You have travelled safely ')

test_data =pd.read_csv('C:\\Pyhton\\playground-series-s5e10\\test.csv',index_col=0)


data_col = test_data.columns
data_col=data_col.drop(['num_lanes', 'time_of_day', 'school_season'])
road_types = test_data['road_type'].unique()
unique= dict()

for data in data_col:
    unique[data] = test_data[data].unique()

choice1=dict()
choice2=dict()
for data in data_col:
    choice1[data] = random.choice(unique[data])
    choice2[data] = random.choice(unique[data])
choice1['curvature'] = random.uniform(max(unique['curvature']), min(unique['curvature']))
choice2['curvature'] = random.uniform(max(unique['curvature']), min(unique['curvature']))

options = pd.DataFrame([choice1, choice2],index=['Condition1', 'Condition2'])

with st.form('Accident_calculator'):
    st.title('Accident Risk Prediction Game')
    st.write('Chose the Safer Driving condition:')
    if "selected_index" not in st.session_state:
        st.session_state.selected_index = None
    for i, row in options.iterrows():
        with st.expander(f" {i}  "):
            st.write(f"Curvature: {row['curvature']} ")
            st.write(f"Speed Limit: {row['speed_limit']}Km/hr")
            st.write(f"Lighting: {row['lighting']}")
            st.write(f"Weather: {row['weather']}")
            st.write(f"Road Sign Present: {row['road_signs_present']}")
            st.write(f"Is it Public road: {row['public_road']}")
            st.write(f"Is it Holiday Season: {row['holiday']}")
            st.write(f"No of accidents reported in this condition: {row['num_reported_accidents']}")

    selection = st.radio('Select the road to travel',options.index,index= st.session_state.selected_index)
    st.form_submit_button('Submit',on_click=accident)




