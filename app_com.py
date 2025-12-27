import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # IMPORTANT
import matplotlib.pyplot as plt
import yfinance as yf
from flask import Flask, render_template, request
from datetime import date
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
from tensorflow.keras.models import load_model
import os
import time

app = Flask(__name__)

close_model = load_model("keras_model.keras", compile=False)
high_model = load_model("keras_model1.keras", compile=False)

@app.route("/", methods=["GET", "POST"])
def index():
    plot_url = plot_ma_url = plot_ma2_url = stats = price_label = None

    if request.method == "POST":
        stock = request.form["stock"]
        price_type = request.form["price_type"]

        df = yf.download(stock, "2015-01-01", date.today().strftime("%Y-%m-%d"))
        if df.empty:
            return render_template("index.html", error="Invalid Stock Symbol")

        column = "Close" if price_type == "close" else "High"
        price_label = "Closing Price Prediction" if price_type == "close" else "High Price Prediction"

        model = close_model if price_type == "close" else high_model
        stats = df.describe().to_html()

        os.makedirs("static", exist_ok=True)
        uid = str(int(time.time()))

        ma100 = df[column].rolling(100).mean()
        ma200 = df[column].rolling(200).mean()

        plt.figure(figsize=(12,6))
        plt.plot(ma100, label="MA100")
        plt.plot(df[column], label=column)
        plt.legend()
        plot_ma = f"static/ma100_{column}_{uid}.png"
        plt.savefig(plot_ma)
        plt.close("all")

        plt.figure(figsize=(12,6))
        plt.plot(ma100, label="MA100")
        plt.plot(ma200, label="MA200")
        plt.plot(df[column], label=column)
        plt.legend()
        plot_ma2 = f"static/ma200_{column}_{uid}.png"
        plt.savefig(plot_ma2)
        plt.close("all")

        data_training = pd.DataFrame(df[column][:int(len(df)*0.8)])
        data_testing = pd.DataFrame(df[column][int(len(df)*0.8):])

        # scaler = MaxAbsScaler()
        scaler = MinMaxScaler()
        scaler.fit(data_training)

        past_100 = data_training.tail(100)
        final_df = pd.concat([past_100, data_testing])
        input_data = scaler.transform(final_df)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        y_pred = model.predict(x_test, verbose=0)

        scale_factor = 1 / scaler.scale_[0]
        y_pred *= scale_factor
        y_test *= scale_factor

        plt.figure(figsize=(10,5))
        plt.plot(y_test, label="Original")
        plt.plot(y_pred, label="Predicted")
        plt.legend()
        plot_path = f"static/prediction_{column}_{uid}.png"
        plt.savefig(plot_path)
        plt.close("all")

        plot_url = plot_path
        plot_ma_url = plot_ma
        plot_ma2_url = plot_ma2

    return render_template(
        "index.html",
        plot_url=plot_url,
        plot_ma_url=plot_ma_url,
        plot_ma2_url=plot_ma2_url,
        stats=stats,
        price_label=price_label
    )
if __name__ == "__main__": 
    app.run(debug=True, threaded=False)
