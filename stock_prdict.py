import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet import Prophet
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import mean_absolute_error,mean_squared_error
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from time import time
import warnings
import datetime
from datetime import timedelta

warnings.filterwarnings("ignore")

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', "JPM")
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# n_years = st.slider('Years of forecast:', 1,2)
period = 1 * 365


@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Select forecasting model
models = ['Prophet', 'ARIMA', 'Exponential Smoothing']
selected_model = st.selectbox('Select forecasting model', models)

# Predict forecast with selected model
if selected_model == 'Prophet':
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    
   
    # Perform cross-validation
    df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days')

    
    # Calculate MAE, MSE, and RMSE
    mae = mean_absolute_error(df_cv['y'], df_cv['yhat'])
    mse = mean_squared_error(df_cv['y'], df_cv['yhat'])
    rmse = np.sqrt(mse)


    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write("Metrics:-")
    st.text(f'Mean Absolute Error: {mae:.2f}')
    st.text(f'Mean Squared Error: {mse:.2f}')
    st.text(f'Root Mean Squared Error: {rmse:.2f}')
    
        
    st.markdown(f'Forecast plot for {1} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.markdown("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)



elif selected_model == 'ARIMA':
    prices = data['Close']
    dates = data['Date']


    # Fit the ARIMA model
    model = ARIMA(prices, order=(8, 0, 5))  # Example order, adjust as needed
    fitted_model = model.fit()
    
    st.header(f'Forecast plot for {1} years')
    # Forecast the values
    forecast_values = fitted_model.forecast(steps=300, alpha=0.05)
     # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast_values.tail())

    last_date = data['Date'].iloc[-1]
    forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=period)

    # Plot actual and forecasted values
    st.subheader('ARIMA Forecast for Prices')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=prices, mode='lines', name='Actual Prices', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_values, mode='lines', name='Forecasted Prices', line=dict(color='red')))
    fig.update_layout(title='ARIMA Forecast for Prices', xaxis_title='Date', yaxis_title='Prices', showlegend=True)
    st.plotly_chart(fig)
    
    actual_values =  data['Close'].values[-len(forecast_values):]
    mae = mean_absolute_error(actual_values, forecast_values)
    mse = mean_squared_error(actual_values, forecast_values)
    rmse = np.sqrt(mse)
    
    st.write("Metrics:-")
    st.text(f'Mean Absolute Error: {mae:.2f}')
    st.text(f'Mean Squared Error: {mse:.2f}')
    st.text(f'Root Mean Squared Error: {rmse:.2f}')




elif selected_model == 'Exponential Smoothing':
    prices = data['Close']
    dates = data['Date']

    # st.sidebar.header('Exponential Smoothing Parameters')
    # alpha = st.sidebar.slider('Smoothing Level (Alpha)', min_value=0.1, max_value=1.0, value=0.3, step=0.1)
    st.header(f'Forecast plot for {1} years')
    model = ExponentialSmoothing(prices, trend='mul', seasonal='mul', seasonal_periods=12)
    fit_model = model.fit(smoothing_level=0.4)  # Use the alpha parameter for smoothing level
    forecast = fit_model.forecast(steps=365)
    last_date = dates.iloc[-1]
    forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=period)
    st.subheader('Forecast data')
    st.write(forecast.tail())

    # Plot actual and forecasted values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=forecast_index , y=forecast, mode='lines', name='Forecasted Prices'))
    fig.update_layout(title='Exponential Smoothing Forecast for Close Prices',
                    xaxis_title='Date', yaxis_title='Close Price', showlegend=True)
    st.plotly_chart(fig)

    actual_values = data['Close'].values[-len(forecast):]
    mae = mean_absolute_error(actual_values, forecast)
    mse = mean_squared_error(actual_values, forecast)
    rmse = np.sqrt(mse)
    
    st.write("Metrics:-")
    st.text(f'Mean Absolute Error: {mae:.2f}')
    st.text(f'Mean Squared Error: {mse:.2f}')
    st.text(f'Root Mean Squared Error: {rmse:.2f}')



# elif selected_model == 'XGBoost':
#     prices = data['Close']
#     dates = data['Date']

#     # Split the data into training and testing sets
#     train_size = int(len(prices) * 0.8)
#     train_data, test_data = prices[:train_size], prices[train_size:]
#     train_dates, test_dates = dates[:train_size], dates[train_size:]

#     # Convert the data into a format suitable for XGBoost
#     train_matrix = xgb.DMatrix(train_dates.values.reshape(-1, 1), label=train_data.values)
#     test_matrix = xgb.DMatrix(test_dates.values.reshape(-1, 1), label=test_data.values)
#     # Define XGBoost parameters
#     params = {
#         'objective': 'reg:squarederror',
#         'eval_metric': 'rmse',
#         'eta': 0.1,
#         'max_depth': 5,
#         'min_child_weight': 1,
#         'subsample': 0.8,
#         'colsample_bytree': 0.8,
#         'verbosity': 0
#     }
#     # Train the XGBoost model
#     num_round = 100
#     xgb_model = xgb.train(params, train_matrix, num_round)
#     # Make predictions
#     xgb_predictions = xgb_model.predict(test_matrix)

#     # Print forecasted values
#     st.write("XGBoost Forecasted Prices:")
#     for i, value in enumerate(xgb_predictions):
#         st.write(f"Day {i+1}: {value}")

    
#     # Plot actual and forecasted values for XGBoost
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=test_dates, y=test_data, mode='lines', name='Actual Prices'))
#     fig.add_trace(go.Scatter(x=test_dates, y=xgb_predictions, mode='lines', name='XGBoost Forecasted Prices'))
#     fig.update_layout(title='XGBoost Forecast for Prices', xaxis_title='Date', yaxis_title='Prices', showlegend=True)
#     st.plotly_chart(fig)


