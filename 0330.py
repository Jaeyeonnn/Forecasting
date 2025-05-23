import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from sklearn.metrics import r2_score
from PIL import Image
import plotly.graph_objects as go
import requests
from io import BytesIO

# Calculate AIC (Akaike Information Criterion)
def calculate_aic(model_fit):
    return model_fit.aic

data = {
    'SKU_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Best Lag (Months)': [6, 11, 3, 3, 5, 2, 7, 11, 6, 12],
    'MAE': [73.3122, 24.1384, 260.0804, 128.426, 67.1348, 10.617, 19.3923, 36.5853, 213.986, 25.9784],
    '2023 Monthly Average Sales': [796.13, 797.70, 807.71, 823.60, 801.82, 122.71, 88.40, 802.50, 809.29, 87.75],
    '2023 Monthly Sales Std': [100.24, 131.25, 204.15, 144.39, 92.92, 114.37, 61.42, 97.49, 185.12, 52.54],
    'Sales Pattern': ['Stable', 'Stable', 'Unstable', 'Stable', 'Stable', 'Unstable', 'Unstable', 'Stable', 'Unstable', 'Unstable']
}

# Convert the data into a DataFrame
df1 = pd.DataFrame(data)


# Data loading and preprocessing (using st.cache_data)
@st.cache_data
def load_data():
    file_url = "https://github.com/jaeyeonnn/Forecasting/raw/main/Sample_Data.csv"
    df = pd.read_csv(file_url)
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

# Forecast sales with ARIMA model
def forecast_sales_arima(df, sku_id, forecast_horizon=6, lag_months=3):
    # Filter data for the selected SKU
    sku_data = df[df['SKU_ID'] == sku_id].copy()

    # Resample weekly data to monthly data by summing sales
    sku_data = sku_data.set_index('Date').resample('M').sum().reset_index()

    # Latest data for forecasting (using Dec 2023 as the cut-off)
    end_date = pd.to_datetime("2023-12-31")
    sku_data_train = sku_data[sku_data['Date'] <= end_date]  # Training data (until Dec 2023)
    sku_data_test = sku_data[sku_data['Date'] > end_date]  # Test data (from Jan 2024 to Jun 2024)

    # Ensure there are enough test data points (at least `forecast_horizon` periods)
    if len(sku_data_test) < forecast_horizon:
        raise ValueError(f"Not enough test data points to make {forecast_horizon}-month forecast")

    # Filter training data based on the lag months for recency
    start_date = end_date - pd.DateOffset(months=lag_months)
    sku_data_train_lag = sku_data_train[sku_data_train['Date'] >= start_date]

    # Fit ARIMA model
    model = ARIMA(sku_data_train_lag['Weekly_Sales'], order=(3, 1, 5))  # (p,d,q) order
    model_fit = model.fit()

    # Forecast future sales (for the specified forecast horizon)
    forecast = model_fit.forecast(steps=forecast_horizon)
    forecast = np.maximum(forecast, 0)

    # Ensure forecast length matches test data length
    if len(forecast) != len(sku_data_test):
        raise ValueError("Forecast length does not match the number of test data points")

    # Align forecast with the correct dates (ensure it's mapped to the correct months)
    forecast_dates = pd.date_range(start=sku_data_test['Date'].iloc[0], periods=forecast_horizon, freq='M')

    # Create a DataFrame for forecast
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Model Forecast': forecast
    })

    # Merge forecast with the test data
    sku_data_test = sku_data_test.merge(forecast_df, on='Date', how='left')

    # Calculate forecast accuracy (MSE, MAE)
    mse = mean_squared_error(sku_data_test['Weekly_Sales'], sku_data_test['Model Forecast'])
    mae = mean_absolute_error(sku_data_test['Weekly_Sales'], sku_data_test['Model Forecast'])
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    bias = (forecast - sku_data_test['Weekly_Sales']).mean()
    aic = calculate_aic(model_fit) 
    mape = np.mean(np.abs((sku_data_test['Weekly_Sales'] - sku_data_test['Model Forecast']) / sku_data_test['Weekly_Sales'])) * 100
    forecast_error = sku_data_test['Model Forecast'] - sku_data_test['Weekly_Sales']

    return forecast, mse, mae, rmse, bias, aic, mape, forecast_error, sku_data_test, sku_data_train_lag

# KPI calculation function (using ARIMA)
def calculate_kpis(df, sku_id, lag_months):
    forecast, mse, mae, rmse, bias, r_squared, mape, forecast_error, sku_data_test, sku_data_train_lag = forecast_sales_arima(df, sku_id, forecast_horizon=6, lag_months=lag_months)
    return mse, mae, rmse, bias, r_squared, mape, forecast_error, forecast, sku_data_test, sku_data_train_lag

# Plot monthly sales and compare year-to-year sales
def plot_yearly_sales_comparison(df, sku_id, year):
    # Filter data for the selected SKU and year
    sku_data = df[df['SKU_ID'] == sku_id].copy()
    sku_data['Year'] = sku_data['Date'].dt.year
    sku_data['Month'] = sku_data['Date'].dt.month

    # Filter data for the selected year and previous year
    current_year_data = sku_data[sku_data['Year'] == year]
    previous_year_data = sku_data[sku_data['Year'] == (year - 1)]

    # Resample monthly sales
    current_year_monthly = current_year_data.groupby('Month').agg({'Weekly_Sales': 'sum'}).reset_index()
    previous_year_monthly = previous_year_data.groupby('Month').agg({'Weekly_Sales': 'sum'}).reset_index()

    # Plot the sales comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(current_year_monthly['Month'], current_year_monthly['Weekly_Sales'], label=f'{year} Sales', color='blue', marker='o', linestyle='-', markersize=6)
    ax.plot(previous_year_monthly['Month'], previous_year_monthly['Weekly_Sales'], label=f'{year-1} Sales', color='grey', marker='o', linestyle='-', markersize=6)

    ax.set_xlabel('Month', fontsize=13)
    ax.set_xticks(range(1, 13)) 
    ax.set_xticklabels(range(1, 13))
    ax.set_ylabel('Sales', fontsize=13)
    ax.set_title(f'Monthly Sales Comparison: This Year vs Last Year {sku_id}: {year-1} vs {year}', fontsize=16, fontweight='bold')
    ax.legend()

    st.pyplot(fig)

# Plot interactive sales forecast
def plot_sales_interactive(sku_data_monthly_filtered, sku_data_test_filtered, forecast, lag_months):
    fig = go.Figure()

    # Plot the actual sales data (monthly)
    fig.add_trace(go.Scatter(x=sku_data_monthly_filtered['Date'], 
                             y=sku_data_monthly_filtered['Weekly_Sales'], 
                             mode='lines+markers', 
                             name='Actual Sales', 
                             line=dict(dash='dash', color='blue', width=2),
                             marker=dict(color='blue', size=6)))

    # Plot the forecast for the selected lag period
    fig.add_trace(go.Scatter(x=sku_data_test_filtered['Date'], 
                             y=forecast[:len(sku_data_test_filtered)], 
                             mode='lines+markers', 
                             name=f'Forecast', 
                             line=dict(color='red', width=2),
                             marker=dict(color='red', size=6)))

    # Customize the layout for better appearance
    fig.update_layout(
        title=f'Forecast vs Actual Sales with {lag_months}-month Lag',
        title_font=dict(size=20),  
        xaxis_title='Date',
        yaxis_title='Sales',
        showlegend=True,
        hovermode='closest',  
        xaxis=dict(tickvals=sku_data_test_filtered['Date'], tickformat='%b %Y'), 
        height=600,
        width=1000,
        font=dict(size=16)  
    )

    return fig

# Display Monthly Comparison Table (January to June 2024) with additional Forecast Error explanation
def display_monthly_comparison_table(sku_data_test, forecast, bias, mape, forecast_error, r_squared):
    def calculate_accuracy(actual, predicted):
        actual = np.where(actual == 0, np.nan, actual)
        accuracy = (1 - np.abs(predicted - actual) / actual) * 100
        accuracy = np.clip(accuracy, 0, 100)
        return accuracy

    sku_data_test['Accuracy'] = calculate_accuracy(sku_data_test['Weekly_Sales'], sku_data_test['Model Forecast'])
    sku_data_test['Accuracy'] = sku_data_test['Accuracy'].apply(lambda x: round(x, 2) if not np.isinf(x) and not np.isnan(x) else "NA")

    comparison_df = pd.DataFrame({
        'Month': sku_data_test['Date'].dt.strftime('%b %Y'),
        'Model Forecast': sku_data_test['Model Forecast'],
        'Actual Sales': sku_data_test['Weekly_Sales'],
        'Forecast Error': (sku_data_test['Model Forecast'] - sku_data_test['Weekly_Sales']),
        'Accuracy(%)': sku_data_test['Accuracy'].astype(str) + '%'
    })

    st.markdown("### **Monthly Comparison of Forecast and Actual Sales (Jan - Jun 2024):**")
    st.table(comparison_df)

    # Calculate the total forecast error
    total_forecast_error = forecast_error.sum()

    # Add narrative explanation based on the forecast error
    if total_forecast_error < 0:
        explanation = f"Total Forecast Error: {total_forecast_error:.2f} (Underestimated). This results in missed sales opportunities, which could lead to lost revenue due to stockouts."
    elif total_forecast_error > 0:
        explanation = f"Total Forecast Error: {total_forecast_error:.2f} (Overestimated). This results in higher logistics storage costs, increased risk of slow-moving inventory, and potential waste for perishable products."
    else:
        explanation = "Total Forecast Error: 0. The forecasted and actual sales match perfectly."

    # Display the explanation in a text box
    st.markdown(f"""
    <div style="border: 2px solid #f44336; padding: 10px; background-color: #fce4e4; font-size: 16px; margin-top: 20px;">
        <strong>🔍 Forecast Error Summary:</strong>
        <p>{explanation}</p>
    </div>
    """, unsafe_allow_html=True)

# SMAPE Calculation
def calculate_smape(actual, predicted):
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    diff = np.abs(actual - predicted) / denominator
    diff[denominator == 0] = 0  # Handle case where both actual and predicted are zero
    return np.mean(diff) * 100 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Assuming calculate_kpis function is already defined

def display_lag_performance_table(df, sku_id):
    lag_results = []

    # Iterate over lag_months from 2 to 12
    for lag_months in range(2, 13):
        # Get KPIs for each lag_months
        mse, mae, rmse, bias, aic, mape, forecast_error, forecast, sku_data_test, _ = calculate_kpis(df, sku_id, lag_months)

        # Accuracy calculation
        actual = sku_data_test['Weekly_Sales']
        predicted = sku_data_test['Model Forecast']
        actual = np.where(actual == 0, np.nan, actual)
        accuracy = (1 - np.abs(predicted - actual) / actual) * 100
        accuracy = np.clip(accuracy, 0, 100)
        accuracy = accuracy.mean()  

        # Format accuracy and MAPE as percentage
        accuracy_percentage = f"{accuracy:.2f}%" if not np.isnan(accuracy) else "NA"
        mape_percentage = f"{mape:.2f}%" if not np.isnan(mape) and not np.isinf(mape) else "NA"
        
        # Calculate weighted score (70% MAE, 30% AIC)
        weighted_score = 0.7 * mae + 0.3 * aic  # 가중치 적용

        # Append results including weighted score
        lag_results.append([lag_months, mae, mape_percentage, accuracy_percentage, aic, weighted_score])

    # Convert lag_results to DataFrame
    lag_performance_df = pd.DataFrame(lag_results, columns=['Lag (Months)', 'MAE', 'MAPE(%)', 'Accuracy(%)', 'AIC', 'Weighted Score (MAE & AIC)'])

    # Rank based on weighted score
    lag_performance_df['Rank'] = lag_performance_df['Weighted Score (MAE & AIC)'].rank(ascending=True)
    lag_performance_df['Rank'] = lag_performance_df['Rank'].astype(int)

    # Display lag performance table
    st.markdown("### **Lag-Based Model Performance (Lag 2 to Lag 12):**")
    
    # Styling for the table
    st.markdown("""
        <style>
            .st-table th, .st-table td {
                text-align: center;
                vertical-align: middle;
            }
        </style>
    """, unsafe_allow_html=True)

    # Display the updated table
    st.table(lag_performance_df)

    # Find the lag with the lowest MAE
    best_lag = lag_performance_df.loc[lag_performance_df['MAE'].idxmin()]
    best_lag_value = best_lag['Lag (Months)']
    best_mae_value = best_lag['MAE']

    # Plot MAE for each lag as a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(lag_performance_df['Lag (Months)'], lag_performance_df['MAE'], color='skyblue')

    ax.set_xlabel('Lag (Months)', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('MAE for Each Lag (Months)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(2, 13))  # Set x-ticks to 2, 3, 4, ..., 12
    ax.set_xticklabels(range(2, 13))

    # Display the bar chart
    st.pyplot(fig)
    
    # Add recommendation message based on best MAE
    st.markdown(f"""
    <div style="border: 2px solid #4CAF50; padding: 10px; background-color: #e8f5e9; font-size: 18px;">
        <strong>📊 Recommended Lag (MAE):</strong>
        <br><br>
        Based on the analysis, the model with a <strong>{best_lag_value}-month lag</strong> produces the lowest MAE of <strong>{round(best_mae_value, 2)}</strong>.
        Therefore, we can use the forecasting model with a <strong>{best_lag_value}-month lag</strong> for better accuracy in predicting future sales.
    </div>
""", unsafe_allow_html=True)
    
    # Plot the Lag (Months) vs Rank as a vertical bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by Rank and plot vertically
    lag_performance_df_sorted = lag_performance_df.sort_values(by='Rank')  # Sort by Rank
    ax.bar(lag_performance_df_sorted['Lag (Months)'], lag_performance_df_sorted['Rank'], color='skyblue')

    # Labels and title
    ax.set_ylabel('Rank', fontsize=12)
    ax.set_xlabel('Lag (Months)', fontsize=12)
    ax.set_title('Rank by Lag (using weighted score)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(2, 13))

    # Add ranking as text labels on the bars
    for i, rank in enumerate(lag_performance_df_sorted['Rank']):
        ax.text(lag_performance_df_sorted['Lag (Months)'].iloc[i], lag_performance_df_sorted['Rank'].iloc[i] + 0.1,
                f"Rank {int(rank)}", va='center', fontsize=12, color='black')

    # Display the vertical bar chart
    st.pyplot(fig)


    # Find the lag with the lowest weighted score (Rank 1)
    best_lag_rank = lag_performance_df.loc[lag_performance_df['Rank'] == 1].iloc[0]
    best_lag_value_rank = best_lag_rank['Lag (Months)']
    best_mae_value_rank = best_lag_rank['MAE']
    best_aic_value_rank = best_lag_rank['AIC']
    best_weighted_score_rank = best_lag_rank['Weighted Score (MAE & AIC)']
    
    # Add recommendation message based on best MAE and AIC weighted score (Rank 1)
    st.markdown(f"""
    <div style="border: 2px solid #4CAF50; padding: 10px; background-color: #e8f5e9; font-size: 18px;">
        <strong>📊 Recommended Lag (MAE & AIC weighted):</strong>
        <br><br>
        Based on the analysis, the model with a <strong>{best_lag_value_rank}-month lag</strong> achieves the lowest weighted score (Rank 1), combining MAE (70%) and AIC (30%). This combination is ideal when we want to balance prediction accuracy (low MAE) with model simplicity (low AIC). 
        Using both MAE and AIC ensures that we avoid overfitting while maintaining accurate forecasts. Therefore, we can use the <strong>{best_lag_value_rank}-month lag</strong> when seeking the best balance between accuracy and model complexity for future sales forecasting.
    </div>
        """, unsafe_allow_html=True)


# Explanation for each metric
def display_metric_explanations():
    st.sidebar.write("### KPI Metric Explanations")
    st.sidebar.write("""
        **MAE (Mean Absolute Error):** 
        - The average of the absolute differences between predicted and actual sales. A lower MAE means better prediction accuracy.
                                         
        **MAPE (Mean Absolute Percentage Error):** 
        - The average of the absolute percentage differences between predicted and actual sales. A lower MAPE indicates better prediction accuracy. It is more useful for comparing the accuracy of models across different scales. MAPE is computed by taking the absolute difference between actual and predicted values, dividing it by the actual value, and converting it into a percentage. 

        **Forecast Error:** 
        - The gap between predicted and actual sales (Model Forecast - Actual Sales). A forecast error closer to 0 is ideal, as it means the model is neither overestimating nor underestimating the values. Positive error indicates overestimation, while negative error suggests underestimation. 

        **Accuracy(%):**             
        - Accuracy measures how much the predicted values deviate from the actual values in percentage terms. 

        **AIC (Akaike Information Criterion):** 
        - AIC is a measure used to evaluate and compare statistical models. It helps to determine which model best balances fit and complexity. Lower AIC is better: When comparing multiple models, the lowest AIC indicates the best trade-off between model fit and complexity.
    """)
import plotly.graph_objects as go

def plot_yearly_sales_comparison(df, sku_id, year):
    # Filter the data for the selected SKU and year
    sku_data = df[df['SKU_ID'] == sku_id].copy()
    sku_data['Year'] = sku_data['Date'].dt.year
    sku_data['Month'] = sku_data['Date'].dt.month

    # Filter data for the current and previous year
    current_year_data = sku_data[sku_data['Year'] == year]
    previous_year_data = sku_data[sku_data['Year'] == (year - 1)]

    # Aggregate the sales by month for both years
    current_year_monthly = current_year_data.groupby('Month').agg({'Weekly_Sales': 'sum'}).reset_index()
    previous_year_monthly = previous_year_data.groupby('Month').agg({'Weekly_Sales': 'sum'}).reset_index()

    # Add Year column to both DataFrames before combining it with Month for proper labeling
    current_year_monthly['Year'] = year
    previous_year_monthly['Year'] = year - 1

    # Combine the Month and Year columns to create a single x-axis label
    current_year_monthly['Month_Year'] = current_year_monthly['Month'].astype(str) + ' ' + current_year_monthly['Year'].astype(str)
    previous_year_monthly['Month_Year'] = previous_year_monthly['Month'].astype(str) + ' ' + previous_year_monthly['Year'].astype(str)

    fig = go.Figure()

    # Add trace for current year sales
    fig.add_trace(go.Scatter(
        x=current_year_monthly['Month'], 
        y=current_year_monthly['Weekly_Sales'],
        mode='lines+markers',
        name=f'{year} Sales',
        line=dict(color='red', width=2),
        marker=dict(size=8),
        hovertemplate=(
            'Sales: %{y}<br>'  
            'Month: %{x}<br>'
            'Year: ' + str(year) + '<br>'  
            '<extra></extra>'  
        )
    ))

    # Add trace for previous year sales
    fig.add_trace(go.Scatter(
        x=previous_year_monthly['Month'],  
        y=previous_year_monthly['Weekly_Sales'],
        mode='lines+markers',
        name=f'{year-1} Sales',
        line=dict(color='grey', width=2),
        marker=dict(size=8),
        hovertemplate=(
            'Sales: %{y}<br>'  
            'Month: %{x}<br>'
            'Year: ' + str(year-1) + '<br>'  
            '<extra></extra>' 
        )
    ))

    # Customize layout
    fig.update_layout(
        title=f'Monthly Sales Comparison: {year-1} vs {year}',
        xaxis_title='Month',
        yaxis_title='Sales',
        xaxis=dict(
            tickvals=list(range(1, 13)),
            ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ),
        showlegend=True,
        hovermode='closest',  
        height=600, 
        width=1000, 
        font=dict(size=16),
        title_font=dict(size=20),
    )

    return fig

# Dashboard UI setup
def create_dashboard():
    image_url = "https://github.com/jaeyeonnn/Forecasting/raw/main/05_0x0-Tesla_Wordmark_20_Black.png"
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    st.image(img, use_column_width=True)  
    st.title('Demand Planning Dashboard (ARIMA Model)')

    st.subheader("Lag Recommendation and Sales Pattern Analysis by SKU")
    st.dataframe(df1)

    df = load_data()

    sku_id = st.selectbox('Select SKU', df['SKU_ID'].unique())
    lag_months = st.slider('Recency Period (Lag 2 to 12 months)', 2, 12, 3)

    st.subheader("Forecast for Selected Lag Period:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sku_data = df[df['SKU_ID'] == sku_id]
    sku_data_monthly = sku_data.set_index('Date').resample('M').sum().reset_index()

    # Calculate forecast for the selected lag period and update graph
    mse, mae, rmse, bias, r_squared, mape, forecast_error, forecast, sku_data_test, sku_data_train_lag = calculate_kpis(df, sku_id, lag_months)

    # Plot the actual sales data (monthly)
    # Filter sku_data_monthly to show data from January 2024 onwards
    sku_data_monthly_filtered = sku_data_monthly[sku_data_monthly['Date'] >= pd.to_datetime("2024-01-01")]

    # Plot the actual sales data (monthly) starting from January 2024
    ax.plot(sku_data_monthly_filtered['Date'], sku_data_monthly_filtered['Weekly_Sales'], label='Actual Sales', marker='o', color='blue', linestyle='--')
    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('Sales', fontsize=13)
    ax.set_title(f'Forecast vs Actual Sales for SKU {sku_id} with {lag_months}-month Lag', fontsize=14, fontweight= 'bold')
    ax.legend()

    start_date = pd.to_datetime("2024-01-01")
    end_date = pd.to_datetime("2024-06-30")

    # Filter out data outside the desired range
    sku_data_test_filtered = sku_data_test[sku_data_test['Date'] >= start_date]
    forecast_dates_filtered = pd.date_range(start=start_date, end=end_date, freq='M')

    # Plot with filtered data
    ax.plot(sku_data_test_filtered['Date'], forecast[:len(sku_data_test_filtered)], label=f'Forecast with {lag_months}-month Lag', marker='o', color='red')
    ax.legend()
    ax.set_xticklabels(['Jan 2024', 'Feb 2024', 'Mar 2024', 'Apr 2024', 'May 2024', 'Jun 2024'])  # Set the labels

    interactive_fig = plot_sales_interactive(sku_data_monthly_filtered, sku_data_test_filtered, forecast, lag_months)
    st.plotly_chart(interactive_fig)

    # Display monthly comparison table (Jan - Jun 2024)
    display_monthly_comparison_table(sku_data_test, forecast, bias, mape, forecast_error, r_squared)

    # Display lag-based performance comparison table
    display_lag_performance_table(df, sku_id)

    # Explain the metrics in the sidebar
    display_metric_explanations()

    # Yearly sales comparison
    st.markdown("### Year-over-Year Monthly Sales Comparison:")
    year = st.slider('Select Year for Sales Comparison', 2019, 2024, 2024)
    interactive_yearly_sales_fig = plot_yearly_sales_comparison(df, sku_id, year)
    st.plotly_chart(interactive_yearly_sales_fig)

if __name__ == "__main__":
    create_dashboard()
