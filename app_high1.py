import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from flask import Flask, render_template, request
from datetime import date
from sklearn.preprocessing import MaxAbsScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

model = load_model("keras_model1.keras")

@app.route("/", methods=["GET", "POST"])
def index():
    plot_url_high = None
    plot_ma_url_high = None
    plot_ma2_url_high = None
    stats = None

    if request.method == "POST":
        stock = request.form["stock"]
        start = "2015-01-01"
        end = date.today().strftime("%Y-%m-%d")

        df = yf.download(stock, start, end)

        if df.empty:
            return render_template("index_high.html", error="Invalid Stock Symbol")

        stats = df.describe().to_html()

        ma100 = df['High'].rolling(100).mean()

        plt.figure(figsize=(12,6))
        plt.plot(ma100, label="MA100")
        plt.plot(df['High'], label="Close")
        plt.legend()

        plot_ma_high = "static/ma100_high.png"
        plt.savefig(plot_ma_high)
        plt.close()
        plot_ma_url_high =  plot_ma_high

        ma200 = df['High'].rolling(200).mean()
        plt.figure(figsize=(12,6))
        plt.plot(ma100,color='r', label="MA100")
        plt.plot(ma200,label="MA200")
        plt.plot(df['High'],color='g',label="Close")
        plt.legend()

        plot_ma2_high = "static/ma200_high.png"
        plt.savefig(plot_ma2_high)
        plt.close()

        plot_ma2_url_high = plot_ma2_high
        # Scaling
        data_training = pd.DataFrame(df["High"][:int(len(df)*0.8)])
        data_testing = pd.DataFrame(df["High"][int(len(df)*0.8):])

        scaler = MaxAbsScaler()
        scaler.fit(data_training)

        past_100 = data_training.tail(100)
        final_df = pd.concat([past_100, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)

        x_test, y_test = [], []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i,0])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        y_pred = model.predict(x_test)

        scale_factor = 1 / scaler.scale_[0]
        y_pred *= scale_factor
        y_test *= scale_factor

        # Plot
        plt.figure(figsize=(10,5))
        plt.plot(y_test, label="Original")
        plt.plot(y_pred, label="Predicted")
        plt.legend()

        if not os.path.exists("static"):
            os.makedirs("static")

        plot_path_high = "static/plot_high.png"
        plt.savefig(plot_path_high)
        plt.close()

        plot_url_high = plot_path_high

    return render_template(
        "index_high.html",
        plot_ma_url_high=plot_ma_url_high,
        plot_ma2_url_high=plot_ma2_url_high,
        plot_url_high=plot_url_high,
        stats=stats
    )

if __name__ == "__main__":
    app.run(debug=True)
