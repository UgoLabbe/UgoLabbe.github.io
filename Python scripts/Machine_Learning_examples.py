import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Generate sample data
np.random.seed(42)
n_samples = 300

annual_income = np.random.normal(loc=60000, scale=15000, size=n_samples)
spending_score = np.random.normal(loc=50, scale=25, size=n_samples)

# Function to segment customers
def segment_customer(income, spending):
    if income > 75000 and spending > 60:
        return "Premium", "red"
    elif income < 45000 and spending < 40:
        return "Budget", "blue"
    elif income > 60000 and spending < 30:
        return "Potential", "green"
    else:
        return "Standard", "gray"

# Segment customers
segments = [segment_customer(inc, spend) for inc, spend in zip(annual_income, spending_score)]
segment_names, colors = zip(*segments)

# Create a scatter plot
plt.figure(figsize=(12, 8))
for segment in set(segment_names):
    mask = np.array(segment_names) == segment
    plt.scatter(annual_income[mask], spending_score[mask], 
                c=np.array(colors)[mask], label=segment, alpha=0.6)

plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation')
plt.legend()

plt.tight_layout()
plt.show()

# Print segment information
for segment in set(segment_names):
    segment_data = [(inc, spend) for inc, spend, seg in zip(annual_income, spending_score, segment_names) if seg == segment]
    if segment_data:
        avg_income = np.mean([inc for inc, _ in segment_data])
        avg_spending = np.mean([spend for _, spend in segment_data])
        print(f"\n{segment} Customers:")
        print(f"  Number of customers: {len(segment_data)}")
        print(f"  Average Annual Income: ${avg_income:.2f}")
        print(f"  Average Spending Score: {avg_spending:.2f}")


from prophet import Prophet

# Generate sample daily sales data for 2 years
np.random.seed(42)
date_rng = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
sales = 1000 + np.random.normal(loc=0, scale=100, size=len(date_rng))  # Base sales with some noise
sales += np.sin(np.arange(len(date_rng)) * 2 * np.pi / 365) * 200  # Yearly seasonality
sales += np.sin(np.arange(len(date_rng)) * 2 * np.pi / 7) * 50  # Weekly seasonality
sales = np.maximum(sales, 0)  # Ensure no negative sales

df = pd.DataFrame({'ds': date_rng, 'y': sales})

# Create and fit the Prophet model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model.fit(df)

# Generate future dates for forecasting (next 6 months)
future_dates = model.make_future_dataframe(periods=180)

# Make predictions
forecast = model.predict(future_dates)

# Plot the forecast
fig1 = model.plot(forecast)
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend(['Actual Sales', 'Forecast', 'Uncertainty Interval'])
plt.show()

# Plot the components of the forecast
fig2 = model.plot_components(forecast)
plt.show()

# Print summary of the forecast for the next 30 days
print("\nForecast Summary for the Next 30 Days:")
summary = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
summary['ds'] = pd.to_datetime(summary['ds']).dt.date
summary.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
print(summary.to_string(index=False))

# Calculate and print some metrics
actual_sales = df['y'].values
predicted_sales = forecast['yhat'][:len(actual_sales)].values

mape = np.mean(np.abs((actual_sales - predicted_sales) / actual_sales)) * 100
rmse = np.sqrt(np.mean((actual_sales - predicted_sales)**2))

print(f"\nModel Performance Metrics:")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")

