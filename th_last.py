import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf

# 1. Đọc dữ liệu
file_path = r"D:\Du_doan_bao\currency_exchange_rates_02-01-1995_-_02-05-2018.csv"
df = pd.read_csv(file_path)

# 2. Chuyển cột Date sang datetime và set làm index
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.set_index('Date').sort_index()

# 3. Lấy chuỗi thời gian từ cột 'Indian Rupee' với index là Date
series = df['Indian Rupee']

# 4. Nội suy tuyến tính để xử lý giá trị thiếu (nếu có)
series = series.interpolate(method='linear')
series = series.fillna(method='ffill').fillna(method='bfill')

# 5. Vẽ chuỗi gốc với index là Date
plt.figure(figsize=(12,6))
plt.plot(series)
plt.title('Indian Rupee Exchange Rate Over Time (After Interpolation)')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.show(block = True)

# 6. Kiểm tra tính dừng với ADF test
result = adfuller(series)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# 7. Sai phân bậc 1 nếu chuỗi không dừng
if result[1] > 0.05:
    series_diff = series.diff().dropna()
    result_diff = adfuller(series_diff)
    print('After differencing:')
    print('ADF Statistic:', result_diff[0])
    print('p-value:', result_diff[1])
else:
    series_diff = series

# 8. Vẽ chuỗi sai phân
plt.figure(figsize=(12,6))
plt.plot(series_diff)
plt.title('Differenced Indian Rupee Exchange Rate')
plt.xlabel('Date')
plt.ylabel('Differenced Exchange Rate')
plt.show(block =  True)

# 9. Xây dựng mô hình AR(1)
model = AutoReg(series_diff, lags=1).fit()
print(model.summary())

# 10. Dự báo 10 bước tiếp theo
forecast = model.predict(start=len(series_diff), end=len(series_diff)+9)

# 11. Tạo index cho dự báo tiếp theo dựa trên index chuỗi gốc
last_date = series_diff.index[-1]
forecast_dates = pd.date_range(start=last_date, periods=11, freq='D')[1:]  # giả sử dữ liệu hàng ngày

# 12. Tạo DataFrame dự báo với index thời gian
df_forecast = pd.DataFrame({'Forecasted Value': forecast.values}, index=forecast_dates)
print(df_forecast)

phi = model.params[1]  # Hệ số phi là tham số của lag 1
print(f'Hệ số phi (phi) của mô hình AR(1): {phi}')

# Vẽ ACF chuỗi gốc
plt.figure(figsize=(12,4))
plot_acf(series.dropna(), lags=40, alpha=0.05)
plt.title('ACF of Original Series (Indian Rupee)')
plt.show()

# Nếu sai phân thì vẽ ACF chuỗi sai phân
if result[1] > 0.05:
    plt.figure(figsize=(12,4))
    plot_acf(series_diff, lags=40, alpha=0.05)
    plt.title('ACF of Differenced Series (Indian Rupee)')
    plt.show(block = True)