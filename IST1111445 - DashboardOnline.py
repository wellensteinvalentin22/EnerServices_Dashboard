# # Project 2 - Dashboard
# By Valentin Wellenstein - IST1111445

# ### Importations and CSS Style definition

import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np

# #### Define CSS style

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# ### Collecting and preparing 2019 Data

# #### Load Data

df_data=pd.read_csv('testData_2019_SouthTower.csv')
df_data['Date'] = pd.to_datetime (df_data['Date'])
df_data = df_data.set_index('Date')
#df_data['Date'] = pd.to_datetime (df_data['Date'])
#df_data = df_data.set_index('Date')

# #### Renaming columns

df_data.rename(columns = {'South Tower (kWh)': 'Power (kW)'}, inplace = True)
df_data.rename(columns = {'temp_C': 'Temperature (°C)'}, inplace = True)
df_data.rename(columns = {'HR': 'HR (%)'}, inplace = True)
df_data.rename(columns = {'windSpeed_m/s': 'Wind Speed (m/s)'}, inplace = True)
df_data.rename(columns = {'windGust_m/s': 'Wind Gust (m/s)'}, inplace = True)
df_data.rename(columns = {'pres_mbar': 'Pressure (Mbar)'}, inplace = True)
df_data.rename(columns = {'solarRad_W/m2': 'Solar Radiation (W/m2)'}, inplace = True)
df_data.rename(columns = {'rain_mm/h': 'Rain (mm/h)'}, inplace = True)
df_data.rename(columns = {'rain_day': 'Rain day'}, inplace = True)

df_data.head()

# #### Features engineering

# +
df_data['Power-1']=df_data['Power (kW)'].shift(1) # Previous hour consumption
df_data['HDH']=np.maximum(0,-df_data['Temperature (°C)']+16) # Heating degree hour

df_data=df_data.drop(columns=['Wind Speed (m/s)','Wind Gust (m/s)','Rain (mm/h)','Rain day'])
df_data=df_data.dropna()

df_data=df_data.iloc[:, [0,5,1,2,3,4,6]] # Change the position of the columns so that Y=column 0 and X all the remaining columns
df_data=df_data.reset_index()

df_data.head()
# -

fig1 = px.line(df_data, x="Date", y=df_data.columns[2:8]) # Creates a figure with the raw data

# #### Loading real consumption in 2019 (basically the same 2019 test file)

# +
df_real = df_data

df_real['Date'] = pd.to_datetime (df_real['Date'])
df_real = df_real.set_index('Date')

df_real.head()
# -

# #### Preparing columns for regression

y2=df_real['Power (kW)'].values

Z=df_real.values
Y=Z[:,0]
X=Z[:,[1,2,3,4,5,6]]

# ### Loading models

# #### Load LR model

with open('LR_model.pkl','rb') as file:
    LR_model2=pickle.load(file)
y2_pred_LR = LR_model2.predict(X)

# #### Evaluate LR errors

MAE_LR=metrics.mean_absolute_error(y2,y2_pred_LR) 
MBE_LR=np.mean(y2-y2_pred_LR)
MSE_LR=metrics.mean_squared_error(y2,y2_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y2,y2_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y2)
NMBE_LR=MBE_LR/np.mean(y2)

# #### Load RF model

with open('RF_model.pkl','rb') as file:
    RF_model2=pickle.load(file)
y2_pred_RF = RF_model2.predict(X)

# #### Evaluate LR errors

MAE_RF=metrics.mean_absolute_error(y2,y2_pred_RF)
MBE_RF=np.mean(y2-y2_pred_RF) 
MSE_RF=metrics.mean_squared_error(y2,y2_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y2,y2_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y2)
NMBE_RF=MBE_RF/np.mean(y2)

# #### Load GB model

with open('GB_model.pkl','rb') as file:
    GB_model2=pickle.load(file)
y2_pred_GB = GB_model2.predict(X)

# #### Evaluate GB errors

MAE_GB=metrics.mean_absolute_error(y2,y2_pred_GB)
MBE_GB=np.mean(y2-y2_pred_GB) 
MSE_GB=metrics.mean_squared_error(y2,y2_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y2,y2_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y2)
NMBE_GB=MBE_GB/np.mean(y2)

# #### Load XGB model

with open('XGB_model.pkl','rb') as file:
    XGB_model2=pickle.load(file)
y2_pred_XGB = XGB_model2.predict(X)

# #### Evaluate XGB errors

MAE_XGB=metrics.mean_absolute_error(y2,y2_pred_XGB)
MBE_XGB=np.mean(y2-y2_pred_XGB) 
MSE_XGB=metrics.mean_squared_error(y2,y2_pred_XGB)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y2,y2_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y2)
NMBE_XGB=MBE_XGB/np.mean(y2)

# #### Load BT model

with open('BT_model.pkl','rb') as file:
    BT_model2=pickle.load(file)
y2_pred_BT = BT_model2.predict(X)

# #### Evaluate BT errors

MAE_BT=metrics.mean_absolute_error(y2,y2_pred_BT)
MBE_BT=np.mean(y2-y2_pred_BT) 
MSE_BT=metrics.mean_squared_error(y2,y2_pred_BT)  
RMSE_BT= np.sqrt(metrics.mean_squared_error(y2,y2_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y2)
NMBE_BT=MBE_BT/np.mean(y2)

# ### Preparing the plot

# #### Recleaning the real consumption file in order to merge it after

df_real = df_real.drop(columns=['Temperature (°C)','HR (%)','Pressure (Mbar)','Solar Radiation (W/m2)','Power-1','HDH'])
df_real = df_real.reset_index()
df_real.head()

# #### Create data frames with predicting results and error metrics 

d = {'Methods': ['Linear Regression','Random Forest','Gradient Boosting','Extreme Gradient Boosting','Bootstrapping'], 'MAE': [MAE_LR, MAE_RF, MAE_GB, MAE_XGB, MAE_BT],'MBE': [MBE_LR, MBE_RF, MBE_GB, MBE_XGB, MBE_BT], 'MSE': [MSE_LR, MSE_RF, MSE_GB, MSE_XGB, MSE_BT], 'RMSE': [RMSE_LR, RMSE_RF, RMSE_GB, RMSE_XGB, RMSE_BT],'cvMSE': [cvRMSE_LR, cvRMSE_RF, cvRMSE_GB, cvRMSE_XGB, cvRMSE_BT],'NMBE': [NMBE_LR, NMBE_RF, NMBE_GB, NMBE_XGB, NMBE_BT]}
df_metrics = pd.DataFrame(data=d)
d={'Date':df_real['Date'].values, 'LinearRegression': y2_pred_LR,'RandomForest': y2_pred_RF, 'Gradient Boosting': y2_pred_GB,'Extreme Gradient Boosting': y2_pred_XGB, 'Bootstrapping': y2_pred_BT}
df_forecast=pd.DataFrame(data=d)

# #### Merge real and forecast results and creates a figure with it

df_results=pd.merge(df_real,df_forecast, on='Date')

df_results.columns[1:7]

fig2 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:7])

# #### Define metrics sample in order for interactive dashboard

# +
df_metrics = pd.DataFrame({
    'Metric': ['MAE', 'MAE', 'MAE', 'MAE', 'MAE',
               'MBE', 'MBE', 'MBE', 'MBE', 'MBE',
               'MSE', 'MSE', 'MSE', 'MSE', 'MSE',
               'RMSE', 'RMSE', 'RMSE', 'RMSE', 'RMSE',
               'cvRMSE', 'cvRMSE', 'cvRMSE', 'cvRMSE', 'cvRMSE',
               'NMBE', 'NMBE', 'NMBE', 'NMBE', 'NMBE'],
    'Method': ['Linear Regression','Random Forest','Gradient Boosting','Extreme Gradient Boosting','Bootstrapping',
              'Linear Regression','Random Forest','Gradient Boosting','Extreme Gradient Boosting','Bootstrapping',
              'Linear Regression','Random Forest','Gradient Boosting','Extreme Gradient Boosting','Bootstrapping',
              'Linear Regression','Random Forest','Gradient Boosting','Extreme Gradient Boosting','Bootstrapping',
              'Linear Regression','Random Forest','Gradient Boosting','Extreme Gradient Boosting','Bootstrapping',
              'Linear Regression','Random Forest','Gradient Boosting','Extreme Gradient Boosting','Bootstrapping'],
    'Value': [MAE_LR, MAE_RF, MAE_GB, MAE_XGB, MAE_BT, 
              MBE_LR, MBE_RF, MBE_GB, MBE_XGB, MBE_BT,
              MSE_LR, MSE_RF, MSE_GB, MSE_XGB, MSE_BT,
              RMSE_LR, RMSE_RF, RMSE_GB, RMSE_XGB, RMSE_BT,
              cvRMSE_LR, cvRMSE_RF, cvRMSE_GB, cvRMSE_XGB, cvRMSE_BT,
              NMBE_LR, NMBE_RF, NMBE_GB, NMBE_XGB, NMBE_BT]
    
})
# -

# #### Define CSS style

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)


# #### Define Function to generate the table dynamically based on selected metrics

def generate_table(dataframe, selected_metrics, selected_methods, max_rows=10):
    filtered_df = dataframe[(dataframe['Metric'].isin(selected_metrics)) & (dataframe['Method'].isin(selected_methods))]
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in filtered_df.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(filtered_df.iloc[i][col]) for col in filtered_df.columns
            ]) for i in range(min(len(filtered_df), max_rows))
        ])
    ])



# #### Define Function to generate graph dynamically based on selected metric

def generate_graph(dataframe, selected_metric):
    filtered_df = dataframe[dataframe['Metric'] == selected_metric]
    return dcc.Graph(
        id='error-metrics-graph',
        figure={
            'data': [
                {'x': filtered_df['Method'], 'y': filtered_df['Value'], 'type': 'line', 'name': selected_metric},
            ],
            'layout': {
                'title': f'{selected_metric} with different methods'
            }
        }
    )


# #### Layout

app.layout = html.Div([
    html.H1('IST Energy Forecast tool (kWh)'),
    html.P('Representing Data, Forecasting and error metrics for January 2019 to March 2019 using three tabs'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
        dcc.Tab(label='Error Metrics', value='tab-3'),
    ]),
    html.Div(id='tabs-content')
])

# #### Callback to render content based on selected tab

# +
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H4('IST Raw Data'),
            dcc.Graph(
                id='yearly-data',
                figure=fig1,
            ),
            
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H4('IST Electricity Forecast (kWh)'),
            dcc.Graph(
                id='yearly-data',
                figure=fig2,
                ),
            
        ])
    elif tab == 'tab-3':
        # Dropdowns to select error metrics and methods
        metric_dropdown = dcc.Dropdown(
            id='metric-dropdown',
            options=[{'label': metric, 'value': metric} for metric in df_metrics['Metric'].unique()],
            value=['MAE'],  # Default selected metrics
            multi=True  # Allow multiple selections
        )
        method_dropdown = dcc.Dropdown(
            id='method-dropdown',
            options=[{'label': method, 'value': method} for method in df_metrics['Method'].unique()],
            value=['Linear Regression'],  # Default selected methods
            multi=True  # Allow multiple selections
        )
        
        # Render the table based on selected metrics
        return html.Div([
            html.H4('IST Electricity Forecast Error Metrics'),
            metric_dropdown,
            method_dropdown,
            html.Div(id='error-metrics-table')  # Placeholder for the table
        ])


# -

# #### Callback to update the table based on selected metrics

@app.callback(Output('error-metrics-table', 'children'),
              [Input('metric-dropdown', 'value'),
               Input('method-dropdown', 'value')])
def update_error_metrics_table(selected_metrics, selected_methods):
    table_elements = generate_table(df_metrics, selected_metrics, selected_methods)
    return html.Div(table_elements)


if __name__ == '__main__':
    app.run_server()


