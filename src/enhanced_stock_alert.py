#!/usr/bin/env python3
"""
Enhanced Real-Time Stock Price Prediction and Trading Alert System
Provides buy/sell recommendations at 12pm based on 4-hour price predictions
Includes sentiment analysis from Reddit and news sources
"""

import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import requests
from textblob import TextBlob
from datetime import datetime, timedelta, time
import time as time_module
import schedule
import logging
import os
import json
from typing import Dict, List, Tuple, Optional
import praw
import re
from dotenv import load_dotenv

class EnhancedStockPredictor:
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize the enhanced stock predictor with configuration
        
        Args:
            config_file: Path to configuration JSON file
        """
        self.config = self.load_config(config_file)
        self.setup_logging()
        
        # Initialize components
        self.ticker = self.config['ticker'].upper()
        self.lookback_days = self.config['lookback_days']
        self.prediction_hours = self.config['prediction_hours']
        
        # Scalers
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Model
        self.model = None
        self.last_training_date = None
        
        # Initialize Reddit client if credentials are available
        self.reddit_client = self.init_reddit_client()
        
        # Load environment variables for Slack
        load_dotenv(os.path.expanduser("~/.env"))
        self.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        self.slack_bot_token = os.getenv('SLACK_BOT_TOKEN')
        
        logger.info(f"Enhanced Stock Predictor initialized for {self.ticker}")
    
    def load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found, using defaults")
            return {
                "ticker": "NVDA",
                "lookback_days": 60,
                "prediction_hours": 4,
                "alert_time": "12:00",
                "buy_threshold": 1.0,
                "sell_threshold": -1.0,
                "confidence_threshold": 0.5,
                "training_epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "enable_sentiment_analysis": True,
                "enable_technical_indicators": True,
                "log_level": "INFO",
                "save_alerts_to_file": True,
                "alert_file": "trading_alerts.txt",
                "log_file": "stock_alerts.log"
            }
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.get('log_file', 'stock_alerts.log')),
                logging.StreamHandler()
            ]
        )
        global logger
        logger = logging.getLogger(__name__)
    
    def init_reddit_client(self) -> Optional[dict]:
        """Initialize Reddit client using the logic from the notebook"""
        try:
            # Check for Reddit credentials in separate files (like in the notebook)
            if (os.path.exists('pw.txt') and 
                os.path.exists('client_id.txt') and 
                os.path.exists('client_secret.txt')):
                
                with open('pw.txt', 'r') as f:
                    pw = f.read().strip()
                
                with open('client_id.txt', 'r') as f:
                    client_id = f.read().strip()
                
                with open('client_secret.txt', 'r') as f:
                    client_secret = f.read().strip()
                
                # Use the same authentication logic as the notebook
                auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
                data = {
                    "grant_type": "password",
                    "username": "InterestingRun2732",
                    "password": pw
                }
                headers = {"User-Agent": "MyAPI/0.0.1"}
                
                res = requests.post("https://www.reddit.com/api/v1/access_token", 
                                  auth=auth, data=data, headers=headers)
                
                if res.status_code == 200:
                    token = res.json()['access_token']
                    headers = {**headers, **{'Authorization': f"bearer {token}"}}
                    
                    logger.info("Reddit API client initialized successfully using notebook logic")
                    return {
                        'headers': headers,
                        'token': token,
                        'auth_method': 'oauth'
                    }
                else:
                    logger.warning(f"Failed to get Reddit token: {res.status_code}")
                    return None
            else:
                logger.info("Reddit credential files not found, sentiment analysis will use dummy data")
                return None
        except Exception as e:
            logger.warning(f"Failed to initialize Reddit client: {str(e)}")
            return None
    
    def get_reddit_sentiment(self, subreddits: List[str] = None, limit: int = 100) -> pd.DataFrame:
        """Get sentiment data from Reddit using the notebook logic"""
        if not self.reddit_client:
            return self.get_dummy_sentiment_data()
        
        if subreddits is None:
            subreddits = ['stocks']  # Focus on stocks subreddit as requested
        
        try:
            all_posts = []
            headers = self.reddit_client['headers']
            
            for subreddit_name in subreddits:
                try:
                    # Get posts from the subreddit using the notebook logic
                    res = requests.get(f"https://oauth.reddit.com/r/{subreddit_name}/new", 
                                     headers=headers, params={"limit": limit})
                    
                    if res.status_code == 200:
                        posts = res.json()['data']['children']
                        
                        for post in posts:
                            post_data = post['data']
                            
                            # Check if post mentions the ticker
                            title = post_data.get('title', '').lower()
                            selftext = post_data.get('selftext', '').lower()
                            ticker_lower = self.ticker.lower()
                            
                            # Only include posts that mention the ticker
                            if (ticker_lower in title or 
                                ticker_lower in selftext or 
                                f"${ticker_lower}" in title or 
                                f"${ticker_lower}" in selftext):
                                
                                # Analyze sentiment
                                title_sentiment = TextBlob(post_data['title']).sentiment.polarity
                                body_sentiment = TextBlob(post_data['selftext']).sentiment.polarity if post_data['selftext'] else 0
                                
                                # Weighted sentiment (title more important)
                                weighted_sentiment = (title_sentiment * 0.7) + (body_sentiment * 0.3)
                                
                                all_posts.append({
                                    'date': datetime.fromtimestamp(post_data['created_utc']),
                                    'subreddit': subreddit_name,
                                    'title': post_data['title'],
                                    'score': post_data['score'],
                                    'upvote_ratio': post_data['upvote_ratio'],
                                    'polarity': weighted_sentiment,
                                    'volume': post_data['score'] + post_data.get('num_comments', 0)
                                })
                    else:
                        logger.warning(f"Failed to fetch from r/{subreddit_name}: {res.status_code}")
                        
                except Exception as e:
                    logger.warning(f"Error fetching from r/{subreddit_name}: {str(e)}")
                    continue
            
            if not all_posts:
                logger.info(f"No posts found mentioning {self.ticker} in r/stocks")
                return self.get_dummy_sentiment_data()
            
            df = pd.DataFrame(all_posts)
            
            # Aggregate by date
            df['date'] = pd.to_datetime(df['date']).dt.date
            daily_sentiment = df.groupby('date').agg({
                'polarity': ['mean', 'std'],
                'score': 'mean',
                'upvote_ratio': 'mean',
                'volume': 'sum'
            }).reset_index()
            
            # Flatten column names
            daily_sentiment.columns = ['date', 'avg_polarity', 'std_polarity', 'avg_score', 'avg_upvote_ratio', 'total_volume']
            
            logger.info(f"Found {len(all_posts)} posts mentioning {self.ticker} in r/stocks")
            return daily_sentiment
            
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment: {str(e)}")
            return self.get_dummy_sentiment_data()
    
    def get_ticker_specific_sentiment(self, limit: int = 100) -> pd.DataFrame:
        """Get sentiment data specifically for the ticker from multiple subreddits"""
        if not self.reddit_client:
            return self.get_dummy_sentiment_data()
        
        try:
            all_posts = []
            headers = self.reddit_client['headers']
            
            # Search for posts mentioning the ticker across multiple subreddits
            subreddits = ['wallstreetbets', 'stocks', 'investing']
            
            for subreddit_name in subreddits:
                try:
                    # Get posts from the subreddit
                    res = requests.get(f"https://oauth.reddit.com/r/{subreddit_name}/new", 
                                     headers=headers, params={"limit": limit})
                    
                    if res.status_code == 200:
                        posts = res.json()['data']['children']
                        
                        for post in posts:
                            post_data = post['data']
                            
                            # Check if post mentions the ticker
                            title = post_data.get('title', '').lower()
                            selftext = post_data.get('selftext', '').lower()
                            ticker_lower = self.ticker.lower()
                            
                            # Only include posts that mention the ticker
                            if (ticker_lower in title or 
                                ticker_lower in selftext or 
                                f"${ticker_lower}" in title or 
                                f"${ticker_lower}" in selftext):
                                
                                # Analyze sentiment
                                title_sentiment = TextBlob(post_data['title']).sentiment.polarity
                                body_sentiment = TextBlob(post_data['selftext']).sentiment.polarity if post_data['selftext'] else 0
                                
                                # Weighted sentiment (title more important)
                                weighted_sentiment = (title_sentiment * 0.7) + (body_sentiment * 0.3)
                                
                                all_posts.append({
                                    'date': datetime.fromtimestamp(post_data['created_utc']),
                                    'subreddit': subreddit_name,
                                    'title': post_data['title'],
                                    'score': post_data['score'],
                                    'upvote_ratio': post_data['upvote_ratio'],
                                    'polarity': weighted_sentiment,
                                    'volume': post_data['score'] + post_data.get('num_comments', 0)
                                })
                    else:
                        logger.warning(f"Failed to fetch from r/{subreddit_name}: {res.status_code}")
                        
                except Exception as e:
                    logger.warning(f"Error fetching from r/{subreddit_name}: {str(e)}")
                    continue
            
            if not all_posts:
                logger.info(f"No posts found mentioning {self.ticker} across subreddits")
                return self.get_dummy_sentiment_data()
            
            df = pd.DataFrame(all_posts)
            
            # Aggregate by date
            df['date'] = pd.to_datetime(df['date']).dt.date
            daily_sentiment = df.groupby('date').agg({
                'polarity': ['mean', 'std'],
                'score': 'mean',
                'upvote_ratio': 'mean',
                'volume': 'sum'
            }).reset_index()
            
            # Flatten column names
            daily_sentiment.columns = ['date', 'avg_polarity', 'std_polarity', 'avg_score', 'avg_upvote_ratio', 'total_volume']
            
            logger.info(f"Found {len(all_posts)} posts mentioning {self.ticker} across all subreddits")
            return daily_sentiment
            
        except Exception as e:
            logger.error(f"Error getting ticker-specific sentiment: {str(e)}")
            return self.get_dummy_sentiment_data()
    
    def get_news_sentiment(self) -> pd.DataFrame:
        """Get sentiment data from news sources"""
        try:
            # Try to get news from Yahoo Finance
            stock = yf.Ticker(self.ticker)
            news = stock.news
            
            if not news:
                return self.get_dummy_sentiment_data()
            
            news_data = []
            for article in news[:50]:  # Limit to 50 articles
                try:
                    title = article.get('title', '')
                    if not title:
                        continue
                    
                    # Analyze sentiment
                    sentiment = TextBlob(title).sentiment.polarity
                    
                    news_data.append({
                        'date': datetime.fromtimestamp(article.get('providerPublishTime', datetime.now().timestamp())),
                        'title': title,
                        'source': article.get('publisher', 'Unknown'),
                        'polarity': sentiment,
                        'volume': 1
                    })
                    
                except Exception as e:
                    continue
            
            if not news_data:
                return self.get_dummy_sentiment_data()
            
            df = pd.DataFrame(news_data)
            
            # Aggregate by date
            df['date'] = pd.to_datetime(df['date']).dt.date
            daily_news = df.groupby('date').agg({
                'polarity': ['mean', 'std'],
                'volume': 'count'
            }).reset_index()
            
            # Flatten column names
            daily_news.columns = ['date', 'avg_polarity', 'std_polarity', 'article_count']
            
            return daily_news
            
        except Exception as e:
            logger.error(f"Error getting news sentiment: {str(e)}")
            return self.get_dummy_sentiment_data()
    
    def get_dummy_sentiment_data(self) -> pd.DataFrame:
        """Generate dummy sentiment data for testing"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                            end=datetime.now(), freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'avg_polarity': np.random.normal(0, 0.3, len(dates)),
            'std_polarity': np.random.uniform(0.1, 0.5, len(dates)),
            'avg_score': np.random.randint(10, 100, len(dates)),
            'avg_upvote_ratio': np.random.uniform(0.7, 0.95, len(dates)),
            'total_volume': np.random.randint(50, 200, len(dates)),
            'article_count': np.random.randint(5, 20, len(dates))
        })
    
    def get_historical_data(self, days: int = None) -> pd.DataFrame:
        """Fetch historical stock data with technical indicators"""
        if days is None:
            days = self.config['lookback_days']
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = yf.download(
                self.ticker, 
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False
            )
            
            if data.empty:
                raise ValueError(f"No data found for {self.ticker}")
            
            # Reset index and rename columns
            data.reset_index(inplace=True)
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # Add technical indicators if enabled
            if self.config.get('enable_technical_indicators', True):
                data = self.add_technical_indicators(data)
            
            logger.info(f"Fetched {len(data)} days of historical data for {self.ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        # Moving averages
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_10'] = data['Close'].rolling(window=10).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['MA_200'] = data['Close'].rolling(window=200).mean()
        
        # Price relative to moving averages
        data['Price_vs_MA20'] = (data['Close'] / data['MA_20'] - 1) * 100
        data['Price_vs_MA50'] = (data['Close'] / data['MA_50'] - 1) * 100
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        # Volume indicators
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        data['Volume_Price_Trend'] = (data['Volume'] * data['Close']).rolling(window=20).mean()
        
        # Price changes and volatility
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Change_5'] = data['Close'].pct_change(periods=5)
        data['Volatility'] = data['Price_Change'].rolling(window=20).std()
        
        # Support and resistance levels
        data['High_20'] = data['High'].rolling(window=20).max()
        data['Low_20'] = data['Low'].rolling(window=20).min()
        data['Support_Resistance_Ratio'] = (data['Close'] - data['Low_20']) / (data['High_20'] - data['Low_20'])
        
        # Fill NaN values
        data = data.fillna(method='bfill').fillna(0)
        
        return data
    
    def prepare_data(self, stock_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training with enhanced features"""
        try:
            # Select features for training
            feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'MA_5', 'MA_10', 'MA_20', 'MA_50', 'MA_200',
                'Price_vs_MA20', 'Price_vs_MA50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'BB_Position', 'Volume_Ratio', 'Price_Change', 'Price_Change_5', 'Volatility',
                'Support_Resistance_Ratio'
            ]
            
            # Filter columns that exist in the data
            available_columns = [col for col in feature_columns if col in stock_data.columns]
            
            # Normalize price features
            price_features = stock_data[available_columns].values
            normalized_price = self.price_scaler.fit_transform(price_features)
            
            # Prepare sentiment features
            if not sentiment_data.empty and self.config.get('enable_sentiment_analysis', True):
                sentiment_columns = ['avg_polarity', 'std_polarity', 'avg_score', 'avg_upvote_ratio', 'total_volume']
                available_sentiment = [col for col in sentiment_columns if col in sentiment_data.columns]
                
                if available_sentiment:
                    sentiment_features = sentiment_data[available_sentiment].values
                    normalized_sentiment = self.sentiment_scaler.fit_transform(sentiment_features)
                    
                    # Align sentiment data with stock data
                    sentiment_aligned = np.zeros((len(normalized_price), normalized_sentiment.shape[1]))
                    for i, date in enumerate(stock_data['Date']):
                        sentiment_idx = sentiment_data['date'].dt.date == date.date()
                        if sentiment_idx.any():
                            sentiment_aligned[i] = normalized_sentiment[sentiment_idx.idxmax()]
                    
                    # Combine features
                    combined_features = np.hstack((normalized_price, sentiment_aligned))
                else:
                    combined_features = normalized_price
            else:
                combined_features = normalized_price
            
            # Create sequences for LSTM
            X, y = [], []
            for i in range(len(combined_features) - self.lookback_days):
                X.append(combined_features[i:(i + self.lookback_days)])
                # Predict the close price after prediction_hours
                if i + self.lookback_days + self.prediction_hours < len(combined_features):
                    y.append(combined_features[i + self.lookback_days + self.prediction_hours, 3])  # Close price
                else:
                    y.append(combined_features[i + self.lookback_days, 3])  # Current close price
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build an enhanced LSTM model with attention mechanism"""
        try:
            # Create a more sophisticated model
            inputs = tf.keras.Input(shape=input_shape)
            
            # LSTM layers with residual connections
            x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
            x = tf.keras.layers.Dropout(0.2)(x)
            
            # Add attention mechanism
            attention = tf.keras.layers.Dense(1)(x)
            attention = tf.keras.layers.Softmax(axis=1)(attention)
            x = tf.keras.layers.Multiply()([x, attention])
            
            # Additional LSTM layers
            x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.LSTM(32)(x)
            
            # Dense layers with batch normalization
            x = tf.keras.layers.Dense(32, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.1)(x)
            x = tf.keras.layers.Dense(16, activation='relu')(x)
            outputs = tf.keras.layers.Dense(1)(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Compile with configuration
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.get('learning_rate', 0.001)),
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def train_model(self, force_retrain: bool = False) -> None:
        """Train the prediction model"""
        try:
            # Check if we need to retrain
            today = datetime.now().date()
            if (not force_retrain and 
                self.last_training_date == today and 
                self.model is not None):
                logger.info("Model already trained today, skipping training")
                return
            
            logger.info("Starting model training...")
            
            # Get data
            stock_data = self.get_historical_data()
            
            # Get sentiment data
            if self.config.get('enable_sentiment_analysis', True):
                # Get ticker-specific sentiment from Reddit (focusing on stocks subreddit and ticker mentions)
                reddit_sentiment = self.get_ticker_specific_sentiment()
                news_sentiment = self.get_news_sentiment()
                
                # Combine sentiment data
                sentiment_data = pd.concat([reddit_sentiment, news_sentiment], axis=1).fillna(0)
            else:
                sentiment_data = pd.DataFrame()
            
            # Prepare data
            X, y = self.prepare_data(stock_data, sentiment_data)
            
            if len(X) < 100:
                raise ValueError("Insufficient data for training")
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build and train model
            self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=0.00001
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=self.config.get('training_epochs', 100),
                batch_size=self.config.get('batch_size', 32),
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate model
            test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"Model training completed. Test MAE: {test_mae:.4f}")
            
            self.last_training_date = today
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def get_current_price(self) -> float:
        """Get current stock price"""
        try:
            stock = yf.Ticker(self.ticker)
            current_price = stock.info.get('regularMarketPrice', 0)
            
            if current_price == 0:
                # Fallback to real-time data
                data = yf.download(self.ticker, period="1d", progress=False)
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
            
            return current_price
            
        except Exception as e:
            logger.error(f"Error getting current price: {str(e)}")
            return 0
    
    def predict_future_price(self) -> float:
        """Predict the stock price in prediction_hours"""
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")
            
            # Get recent data
            stock_data = self.get_historical_data(days=self.lookback_days + 10)
            
            # Get sentiment data
            if self.config.get('enable_sentiment_analysis', True):
                # Get ticker-specific sentiment from Reddit (focusing on stocks subreddit and ticker mentions)
                reddit_sentiment = self.get_ticker_specific_sentiment()
                news_sentiment = self.get_news_sentiment()
                sentiment_data = pd.concat([reddit_sentiment, news_sentiment], axis=1).fillna(0)
            else:
                sentiment_data = pd.DataFrame()
            
            # Prepare the most recent sequence
            X, _ = self.prepare_data(stock_data, sentiment_data)
            
            if len(X) == 0:
                raise ValueError("No data available for prediction")
            
            # Get the most recent sequence
            last_sequence = X[-1:]
            
            # Make prediction
            predicted_scaled = self.model.predict(last_sequence, verbose=0)
            
            # Inverse transform to get actual price
            dummy = np.zeros((1, self.price_scaler.scale_.shape[0]))
            dummy[0, 3] = predicted_scaled[0, 0]  # Close price position
            
            predicted_price = self.price_scaler.inverse_transform(dummy)[0, 3]
            
            return predicted_price
            
        except Exception as e:
            logger.error(f"Error predicting future price: {str(e)}")
            return 0
    
    def generate_trading_signal(self) -> Dict[str, any]:
        """Generate buy/sell signal based on prediction"""
        try:
            current_price = self.get_current_price()
            predicted_price = self.predict_future_price()
            
            if current_price == 0 or predicted_price == 0:
                return {
                    'signal': 'HOLD',
                    'reason': 'Unable to get current or predicted price',
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'confidence': 0.0
                }
            
            # Calculate price change percentage
            price_change_pct = ((predicted_price - current_price) / current_price) * 100
            
            # Get thresholds from config
            buy_threshold = self.config.get('buy_threshold', 1.0)
            sell_threshold = self.config.get('sell_threshold', -1.0)
            
            # Determine signal
            if price_change_pct > buy_threshold:
                signal = 'BUY'
                confidence = min(abs(price_change_pct) / 5.0, 1.0)
            elif price_change_pct < sell_threshold:
                signal = 'SELL'
                confidence = min(abs(price_change_pct) / 5.0, 1.0)
            else:
                signal = 'HOLD'
                confidence = 0.5
            
            return {
                'signal': signal,
                'reason': f'Based on current market data and sentiment analysis, the stock is predicted to {signal.lower()} by 4pm EST today',
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_pct': price_change_pct,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {str(e)}")
            return {
                'signal': 'ERROR',
                'reason': str(e),
                'current_price': 0,
                'predicted_price': 0,
                'confidence': 0.0
            }
    
    def send_alert(self, signal_data: Dict[str, any]) -> None:
        """Send trading alert with enhanced formatting"""
        try:
            # Calculate 4pm EST time
            current_time = datetime.now()
            four_pm_est = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # Create enhanced alert message
            signal_emoji = {
                'BUY': '🟢',
                'SELL': '🔴',
                'HOLD': '🟡',
                'ERROR': '⚠️'
            }
            
            emoji = signal_emoji.get(signal_data['signal'], '❓')
            
            alert_msg = f"""
{emoji} STOCK ALERT: {self.ticker} {emoji}

⏰ ALERT TIME: {current_time.strftime('%Y-%m-%d %H:%M:%S')} EST

💰 CURRENT STOCK VALUE: ${signal_data['current_price']:.2f}

🎯 PREDICTED VALUE AT 4PM EST: ${signal_data['predicted_price']:.2f}

📊 PREDICTION: {signal_data['signal']}
📈 EXPECTED CHANGE: {signal_data.get('price_change_pct', 0):.2f}%
🎯 CONFIDENCE LEVEL: {signal_data['confidence']:.1%}

📝 ANALYSIS: {signal_data['reason']}

🔧 MODEL: Enhanced LSTM with Sentiment Analysis
            """.strip()
            
            # Log the alert
            logger.info(f"Trading Alert:\n{alert_msg}")
            
            # Save to file if enabled
            if self.config.get('save_alerts_to_file', True):
                alert_file = self.config.get('alert_file', 'trading_alerts.txt')
                with open(alert_file, 'a') as f:
                    f.write(f"\n{alert_msg}\n{'='*60}\n")
            
            # Print to console
            print(alert_msg)
            
            # Send to Slack
            self.send_slack_notification(alert_msg, "#stock-price-alerts")
            
            # Here you could add integrations for:
            # - Email notifications
            # - SMS alerts
            # - Trading platform APIs
            
        except Exception as e:
            logger.error(f"Error sending alert: {str(e)}")
    
    def send_slack_notification(self, message: str, channel: str = "#stock-price-alerts") -> None:
        """Send notification to Slack using the provided logic"""
        try:
            if self.slack_bot_token:
                # Use Slack API with bot token to specify channel
                response = requests.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={"Authorization": f"Bearer {self.slack_bot_token}"},
                    json={
                        "channel": channel,
                        "text": message
                    }
                )
                result = response.json()
                if not result["ok"]:
                    logger.error(f"Failed to send Slack message: {result.get('error', 'Unknown error')}")
                else:
                    logger.info(f"Successfully sent Slack notification to {channel}")
            elif self.slack_webhook_url:
                # Fallback to webhook if bot token not available
                response = requests.post(self.slack_webhook_url, json={"text": message})
                if response.status_code == 200:
                    logger.info(f"Successfully sent Slack webhook notification")
                else:
                    logger.error(f"Failed to send Slack webhook: {response.status_code}")
            else:
                logger.warning("No Slack credentials configured")
                
        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
    
    def run_daily_analysis(self) -> None:
        """Run the daily analysis and generate alerts"""
        try:
            logger.info(f"Starting daily analysis for {self.ticker}")
            
            # Train model if needed
            self.train_model()
            
            # Generate trading signal
            signal_data = self.generate_trading_signal()
            
            # Send alert
            self.send_alert(signal_data)
            
            logger.info(f"Daily analysis completed for {self.ticker}")
            
        except Exception as e:
            logger.error(f"Error in daily analysis: {str(e)}")
            self.send_alert({
                'signal': 'ERROR',
                'reason': f'Analysis failed: {str(e)}',
                'current_price': 0,
                'predicted_price': 0,
                'confidence': 0.0
            })

def main():
    """Main function to run the enhanced stock predictor"""
    try:
        # Initialize predictor
        predictor = EnhancedStockPredictor()
        
        # Schedule daily analysis
        alert_time = predictor.config.get('alert_time', '12:00')
        schedule.every().day.at(alert_time).do(predictor.run_daily_analysis)
        
        # Also run immediately for testing
        logger.info("Running initial analysis...")
        predictor.run_daily_analysis()
        
        logger.info(f"Enhanced Stock Predictor started for {predictor.ticker}")
        logger.info(f"Scheduled to run daily at {alert_time}")
        logger.info("Press Ctrl+C to stop")
        
        # Keep the script running
        while True:
            schedule.run_pending()
            time_module.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("Stock predictor stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main() 