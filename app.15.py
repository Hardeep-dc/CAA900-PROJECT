from flask import Flask, request, jsonify, render_template
import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings
import os
import boto3
import csv
import matplotlib.pyplot as plt
from io import BytesIO



app = Flask(__name__)

# AWS configuration
s3_client = boto3.client('s3')

# Replace with your actual Keepa API key and AWS resources names
api_key = 'apikey'
raw_s3_bucket_name = 'product-raw-data-keepa'
processed_s3_bucket_name = 'product-processed-data-keepa'

def fetch_product_data(asin):
    cleaned_asin = re.sub(r'[^\w]', '', asin)  # Remove any non-word characters
    url = f'https://api.keepa.com/product?key={api_key}&domain=6&asin={cleaned_asin}&history=1&rating=1'
    response = requests.get(url)
    data = response.json()
    if 'products' in data and data['products']:
        product = data['products'][0]
        time.sleep(3)  # Wait for the page to load
        return product, data
    else:
        return None, None

def extract_product_info(product):
    csv_data = product.get('csv', [])
    rating_count_history = csv_data[16] if len(csv_data) > 16 and csv_data[16] is not None else []
    review_count_history = csv_data[17] if len(csv_data) > 17 and csv_data[17] is not None else []
    
    latest_rating = round(rating_count_history[-1] / 10.0, 2) if rating_count_history else 'N/A'
    latest_review_count = review_count_history[-1] if review_count_history else 'N/A'

    product_info = {
        'title': product.get('title', 'N/A'),
        'brand': product.get('brand', 'N/A'),
        'category': ', '.join([cat['name'] for cat in product.get('categoryTree', [])]) if product.get('categoryTree') else 'N/A',
        'asin': product.get('asin', 'N/A'),
        'last_update': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(((product.get('lastUpdate') + 21564000) * 60))),
        'rating': latest_rating,
        'total_reviews': latest_review_count,
    }

    historical_prices = {}
    latest_price = None
    if len(csv_data) > 1 and csv_data[1] is not None:
        price_history = csv_data[1]
        for i in range(0, len(price_history), 2):
            timestamp = price_history[i]
            price = price_history[i + 1]
            date = time.strftime('%Y-%m-%d', time.gmtime((timestamp + 21564000) * 60))
            if price > 0:
                historical_prices[date] = round(price / 100.0, 2)
        latest_price = round(price_history[-1] / 100.0, 2) if len(price_history) > 1 and price_history[-1] > 0 else None

    last_update_date = product_info['last_update'][:10]
    if latest_price:
        historical_prices[last_update_date] = latest_price
        product_info['latest_price'] = latest_price

    return product_info, historical_prices

def save_data_to_s3(data, filename, bucket_name):
    s3_client.put_object(Bucket=bucket_name, Key=filename, Body=json.dumps(data))

@app.route('/')
def home():
    return render_template('index.html')

import logging
logging.basicConfig(level=logging.DEBUG)
import re

@app.route('/api/search', methods=['GET'])
def search():
    query = request.args.get('query')
    cleaned_query = re.sub(r'[^\w]', '', query)  # Remove any non-word characters
    product, raw_data = fetch_product_data(cleaned_query)
    if product:
        # Save raw JSON data to S3
        save_data_to_s3(raw_data, f'raw_data/{cleaned_query}.json', raw_s3_bucket_name)
        
        product_info, historical_prices = extract_product_info(product)
        
        # Save processed data to S3
        processed_data = {
            'product_info': product_info,
            'historical_prices': historical_prices
        }
        save_data_to_s3(processed_data, f'processed_data/{cleaned_query}.json', processed_s3_bucket_name)
        
        return jsonify(product_info)
    else:
        print(f"No product information found for ASIN: {cleaned_query}")  # Log the issue
        return jsonify({"error": "No product information found"}), 404

@app.route('/api/predict', methods=['GET'])
def predict():
    asin = request.args.get('asin')
    try:
        obj = s3_client.get_object(Bucket=processed_s3_bucket_name, Key=f'processed_data/{asin}.json')
        processed_data = json.loads(obj['Body'].read())
    except s3_client.exceptions.NoSuchKey:
        return jsonify({"error": "No historical price data found for the given ASIN"}), 404

    prediction_results = run_prophet_prediction(processed_data)
    return jsonify(prediction_results)

def run_prophet_prediction(data):
    historical_prices = data['historical_prices']
    product_info = data['product_info']

    amazon_prices = pd.DataFrame(list(historical_prices.items()), columns=['date', 'price'])
    amazon_prices['date'] = pd.to_datetime(amazon_prices['date'])

    if len(amazon_prices) < 14:
        return {"error": "Not enough data points for prediction"}

    amazon_prices.set_index('date', inplace=True)
    amazon_prices = amazon_prices.asfreq('D').ffill()

    df = amazon_prices.reset_index().rename(columns={'date': 'ds', 'price': 'y'})
    train = df[df['ds'] < df['ds'].max() - pd.Timedelta(days=30)]
    test = df[df['ds'] >= df['ds'].max() - pd.Timedelta(days=30)]

    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    y_true = test.set_index('ds')['y']
    forecast_indexed = forecast.set_index('ds')
    y_true = y_true[y_true.index.isin(forecast_indexed.index)]
    y_pred = forecast_indexed.loc[y_true.index, 'yhat']
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    current_price = df['y'].values[-1]
    future_price_30_days = round(forecast['yhat'].values[-1], 2)  # Format to 2 decimal places

    recommendation = 'wait' if future_price_30_days < current_price else 'buy'
    
    return {
        "message": "Prediction completed",
        "product_id": product_info['asin'],
        "current_price": current_price,
        "future_price_30_days": future_price_30_days,
        "recommendation": recommendation
    }

if __name__ == '__main__':
     app.run(debug=True, host='0.0.0.0', port=5000)
