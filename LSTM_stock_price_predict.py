
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
import json


TF_ENABLE_ONEDNN_OPTS=0

interval = 'Day'
periods_to_predict = 30
train_window = 30
#--------------------------function--------------------#
# Function for obtaining data for a specified time interval with repeated attempts
def fetch_data(stock_symbol, start_date, end_date, interval=24, retries=3, delay=5):
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{stock_symbol}/candles.json"
    params = {
        'from': start_date,
        'till': end_date,
        'interval': interval
    }
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print("Error {response.status_code}: Unable to fetch data from {start_date} to {end_date}.")
                return None
        except requests.exceptions.RequestException as e:
            attempt += 1
            print("Attempt {attempt}/{retries} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    return None

# Function to generate sequential time intervals
def generate_date_ranges(start_date, end_date, delta=timedelta(days=365)):
    current_date = start_date
    while current_date < end_date:
        next_date = min(current_date + delta, end_date)
        yield current_date, next_date - timedelta(days=1)
        current_date = next_date

# The main function for pumping out data for the entire period
def download_full_history(stock_symbol):
    all_data = []
    start_date = datetime(2007, 7, 20)  # Начальная дата
    end_date = datetime.today()
    output_file = f'{stock_symbol}_candles_history.json'
    # Generating intervals of one year
    for start, end in generate_date_ranges(start_date, end_date):
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        print(f"Fetching data from {start_str} to {end_str}...")
        data = fetch_data(stock_symbol, start_str, end_str) 
        if data and 'candles' in data:
            all_data.extend(data['candles']['data'])  # Adding data to the general list
        else:
            print(f"No data for period from {start_str} to {end_str}.")
    
    # Saving all data into one JSON file
    with open(output_file, 'w') as outfile:
        json.dump(all_data, outfile, indent=4)
    
    print(f"Data has been successfully saved to {output_file}")

# dump to DataFrame
def dump_to_df(dump_file: str): # XXXX_candles_history.json
    # read data from JSON file
    with open(dump_file, 'r') as file:
        data = json.load(file)
    # convert DataFrame
    columns = ['open', 'close', 'high', 'low', 'value', 'volume', 'begin', 'end']
    df = pd.DataFrame(data, columns=columns)
    df['Date'] = pd.to_datetime(df['end']).dt.date
    df = df[['Date', 'close']]
    df.set_index('Date', inplace=True)
    df = df.asfreq('D')
    df['Price'] = df['close'].interpolate(method = 'polynomial', order = 1)
    df = df[['Price']]
    return df

def check_security_exists(ticker):
    url = f'https://iss.moex.com/iss/securities/{ticker}.json'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if not data['description']['data']:
            st.write(f"😟No security found with ticker '{stock_symbol}'!")
            return False
        else:
            return True
    else:
        st.write(f"Error {response.status_code}: unable to retrieve data for ticker {ticker}.")
        return False

# get LATNAME of a stock
def get_latname(stock_symbol):
    response = requests.get(f'http://iss.moex.com/iss/securities/{stock_symbol}.json')
    data = response.json()
    latname = data['description']['data'][9][2]
    return latname

# plot graph
def plot_stock_df(data, window, stock_symbol):

    latname = get_latname(stock_symbol)

    # Рассчитаем скользящее среднее
    rolling_mean = data['Price'].rolling(window=window).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data['Price'], name='Price', yaxis='y', line=dict(color='#0066cc')))

    # Добавление скользящего среднего на график
    fig.add_trace(go.Scatter(x=data.index, y=rolling_mean, name=f'Moving Average ({window})', yaxis='y', line=dict(color='red', width=0.7)))

    # Настройка осей
    fig.update_layout(
         title={
            'text': f'{latname}',
            'font': dict(size=35, weight="bold")
        },
        xaxis_title='Date',
        
        # Настройка оси Y
        yaxis=dict(
            title="Price",
            anchor="x",
            side="left",
            showgrid=True,
            zeroline=True,
            showline=False
        ),

        # Настройка отображения легенды
        # legend=dict(x=0.01, y=0.9, traceorder='normal', font=dict(size=12)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1)
    )

    # Отображение графика
    st.plotly_chart(fig)

# plot predicted graph
def plot_stock_prediction(stdf, prdf, periods_to_predict, window=50, stock_symbol='Stock'):

    latname = get_latname(stock_symbol)

    # Рассчитаем скользящее среднее
    rolling_mean = stdf['Price'].rolling(window=window).mean()
    
    # Создаем график тренировки модели
    trace1 = go.Scatter(
        x=stdf['Date'],
        y=stdf['Price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue', width=3)
    )
    
    # Создаем график предсказанных данных
    trace2 = go.Scatter(
        x=prdf['index'],
        y=prdf['Price'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='red', width=3)
    )

    # График скользящего среднего (Moving Average)
    trace3 = go.Scatter(
        x=stdf['Date'],
        y=rolling_mean,
        mode='lines',
        name=f'Moving Average ({window})',
        line=dict(color='red', width=1)
    )

    # График нижней границы
    trace_lower_bound = go.Scatter(
        x=prdf['index'],
        y=prdf['lower_bound'],
        mode='lines',
        name='Lower Bound',
        line=dict(color='gray', width=0.1),
        showlegend=False
    )

    # График верхней границы с заливкой серым цветом
    trace_upper_bound = go.Scatter(
        x=prdf['index'],
        y=prdf['upper_bound'],
        mode='lines',
        name='Upper Bound',
        line=dict(color='gray', width=0.01),
        fill='tonexty',  # Заливаем пространство между верхней и нижней границами
        fillcolor='rgba(192, 192, 192, 0.5)',  # Серый цвет с прозрачностью
        showlegend=False
    )

    # Оси и заголовок
    layout = go.Layout(
        title={
            'text': f'{latname} Next {periods_to_predict}-Days Price Prediction',
            'font': dict(size=20, weight="bold")
        },
        xaxis=dict(title='Date'),
        yaxis=dict(
            title='Price',
            side='left',
            overlaying='y',
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1)
    )
    
    # Собираем фигуру
    fig = go.Figure(data=[trace2, trace1, trace_lower_bound, trace_upper_bound], layout=layout)

    # Отображаем график
    #pio.show(fig)
    st.plotly_chart(fig)

#download model
with open('LSTM_stock_model.pkl', 'rb') as f:
    load_data = pickle.load(f)
model = load_data['model']

# Generate predictions and update input data
def make_predictions(periods, inputs_test):
    pred_data = []
    for _ in range(periods):
        next_pred = model.predict(inputs_test)
        pred_data.append(next_pred[0, 0])  # Store prediction
        next_pred_scaled = np.reshape(next_pred, (1, 1, 1))  # Reshape prediction
        inputs_test = np.append(inputs_test[:, 1:, :], next_pred_scaled, axis=1)
    return pred_data

# Function to set a date index
def set_date_index(df, start_date):
    date_range = pd.date_range(start=start_date, periods=len(df), freq='D')
    df.index = date_range
    return df

# Function to find deviation bounds on a chart
def add_deviation_bounds(df, initial_spread=0.1, spread_growth_rate=0.08):
    rolling_mean = df['Price'].rolling(window=1, min_periods=1).mean()  # Calculate rolling mean
    std_dev = df['Price'].std()  # Calculate standard deviation for the entire dataset
    
    # Add columns for lower and upper bounds of the spread
    spread = initial_spread + spread_growth_rate * np.arange(len(df))
    df['lower_bound'] = rolling_mean - spread * std_dev
    df['upper_bound'] = rolling_mean + spread * std_dev
    return df

#-------------------------end-of-functions--------------------------------#

with st.sidebar:
    st.title("Info")

st.title("MOEX Stock Price Prediction")
st.divider()
st.markdown("### Step 1. Plotting A History Graph")
col1, col2 = st.columns(2)
stock_symbol = col1.text_input('Input Stock Symbol', placeholder='SBER')

if not stock_symbol or not check_security_exists(stock_symbol):
    st.stop()

download_full_history(stock_symbol)

dump_file_name = f'{stock_symbol}_candles_history.json'
stock_df = dump_to_df(dump_file_name)

plot_stock_df(stock_df, 200, stock_symbol)

st.markdown("### Step 2. Plotting A Prediction Graph")
col1, col2, col3 = st.columns(3)
ok = col1.button('Make Prediction', type="primary", use_container_width = True)

if not ok:
    st.stop()

# Initialize progress bar
progress_text = "Operation in progress. Please wait."
my_bar = st.progress(10, text=progress_text)

# Step 1: Data preparation (10% progress)
my_bar.progress(20, text=progress_text)

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(stock_df)
inputs_data = stock_df[-train_window:].values.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)
X_inputs_test = np.array([inputs_data])
X_inputs_test = np.reshape(X_inputs_test, (X_inputs_test.shape[0], X_inputs_test.shape[1], 1))

# Step 2: Making predictions (50% progress)
my_bar.progress(30, text=progress_text)

# Convert predicted data and inverse scale
predicted_data = make_predictions(periods_to_predict, X_inputs_test)
predicted_data = scaler.inverse_transform(np.array(predicted_data).reshape(-1, 1))

# Step 3: Create DataFrame and add deviation bounds (80% progress)
my_bar.progress(80, text=progress_text)

# Create DataFrame with predicted data
predicted_df = pd.DataFrame(predicted_data, columns=['Price'])

# Set date index and add deviation bounds
predicted_df = set_date_index(predicted_df, stock_df.index[-1] + timedelta(1))
predicted_df = add_deviation_bounds(predicted_df, initial_spread=0.1, spread_growth_rate=0.1)

# Step 4: Plot the data (100% progress)
my_bar.progress(100, text="Plotting the graph...")

# Display the plot
num_last_periods = 300
stdf = stock_df.reset_index()
prdf = predicted_df.reset_index()
plot_stock_prediction(stdf[-num_last_periods:], prdf, periods_to_predict, window=50, stock_symbol=stock_symbol)

# Clear progress bar
time.sleep(1)
my_bar.empty()

st.write('#### Quotes')
df_table = prdf[['index','Price']]
df_table = df_table.rename(columns={'index':'Date'})
df_table['Date'] = pd.to_datetime(df_table['Date']).dt.date
st.dataframe(df_table, width=2000)