HerdingGNN for Stock Price Forecasting using Event Sentiment and Behavioral Graphs

This repository contains an implementation of our HerdingGNN for time-series forecasting of S&P 500 stock prices.

The model leverages a CSAD-based herding adjacency matrix to capture behavioral co-movement between stocks, while preserving temporal dynamics through LSTM and attention layers.

Overview

The proposed architecture combines:

Graph Convolutional Networks (GCN): to model structural relationships between stocks based on herding similarity.

LSTM (Long Short-Term Memory): to capture temporal dependencies in financial time series or optional BiLSTM

Self-Attention Mechanism: to focus on important time steps.

Percentile-Based Thresholding: adjacency matrix constructed using 25%, 50%, and 75% similarity thresholds for robustness analysis.

Unlike traditional correlation-based graphs, the proposed graph is constructed using behavioral dispersion similarity (CSAD), enabling the model to capture investor herding effects.

Data
Market Data Sources

Historical stock price data
Fetched from Yahoo Finance
https://finance.yahoo.com/

List of S&P 500 tickers
Fetched from Wikipedia
https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

Event-Based Sentiment Data

Twitter Event Dataset (2012–2016)
Used for extracting event-driven sentiment features.
https://figshare.com/articles/dataset/Twitter_event_datasets_2012-2016_/5100460
