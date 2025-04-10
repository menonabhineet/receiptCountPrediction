from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load saved model coefficients
with open("model_theta.pkl", "rb") as f:
    theta = pickle.load(f)

degree = len(theta) - 1

def predict(months):
    X_poly = np.hstack([months ** i for i in range(degree + 1)])
    y_pred_raw = X_poly @ theta

    # Rescaling logic
    start_2022 = 310_000_000  # Last month of 2021 was 309948684, so I started the prediction from around there
    range_2021 = 89_915_224  # Dec - Jan 2021 

    full_months = np.arange(1, 13).reshape(-1, 1)
    full_poly = np.hstack([full_months ** i for i in range(degree + 1)])
    full_pred_raw = full_poly @ theta
    range_pred = full_pred_raw.max() - full_pred_raw.min()

    if range_pred == 0:
        return y_pred_raw  # avoid division by zero

    y_pred_scaled = ((y_pred_raw - full_pred_raw.min()) / range_pred) * range_2021 + start_2022
    return y_pred_scaled

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    chart_paths = []

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Load actual 2021 data
    df = pd.read_csv("data_daily.csv")
    df['# Date'] = pd.to_datetime(df['# Date'])
    df['Month'] = df['# Date'].dt.month
    monthly_data = df.groupby('Month')['Receipt_Count'].sum().reset_index()
    actual_2021 = monthly_data['Receipt_Count'].values

    if request.method == 'POST':
        month_input = request.form.get("month")
        if month_input.lower() == "all":
            months = np.arange(1, 13).reshape(-1, 1)
            preds = predict(months)

            # Chart 1 – 2021 Actual
            plt.figure(figsize=(6, 4))
            plt.plot(month_names, actual_2021, marker='o', color='skyblue')
            plt.title("2021 Actual Receipt Counts", color='white')
            plt.xlabel("Month", color='white')
            plt.ylabel("Receipt Count", color='white')
            plt.grid(True)
            plt.xticks(color='white')
            plt.yticks(color='white')
            plt.tight_layout()
            chart_2021 = "static/chart_2021.png"
            plt.savefig(chart_2021, facecolor="#121212")
            plt.close()

            # Chart 2 – 2022 Prediction
            plt.figure(figsize=(6, 4))
            plt.plot(month_names, preds.flatten(), marker='o', linestyle='--', color='salmon')
            plt.title("2022 Predicted Receipt Counts", color='white')
            plt.xlabel("Month", color='white')
            plt.ylabel("Receipt Count", color='white')
            plt.grid(True)
            plt.xticks(color='white')
            plt.yticks(color='white')
            plt.tight_layout()
            chart_2022 = "static/chart_2022.png"
            plt.savefig(chart_2022, facecolor="#121212")
            plt.close()

            chart_paths = [chart_2021, chart_2022]

            prediction = {month_names[i]: {
                '2021': int(actual_2021[i]),
                '2022': int(preds[i])
            } for i in range(12)}

        else:
            month = int(month_input)
            months = np.array([[month]])
            pred = predict(months)[0]
            actual = actual_2021[month - 1]
            prediction = {month_names[month - 1]: {
                '2021': int(actual),
                '2022': int(pred)
            }}

    return render_template("index.html", prediction=prediction, chart_paths=chart_paths)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')