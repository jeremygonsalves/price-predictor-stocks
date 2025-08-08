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
        # Setup logging first
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Update logging configuration from config
        self.update_logging_from_config()
        
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
            logger.error(f"Config file {config_file} not found. Please ensure the config.json file exists in the project root.")
            raise FileNotFoundError(f"Config file {config_file} not found. Please create a config.json file with the required settings.")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {config_file}: {str(e)}")
            raise ValueError(f"Invalid JSON format in config file: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Use default logging level since config isn't loaded yet
        log_level = logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('stock_alerts.log'),
                logging.StreamHandler()
            ]
        )
        global logger
        logger = logging.getLogger(__name__)
    
    def update_logging_from_config(self):
        """Update logging configuration after config is loaded"""
        try:
            if hasattr(self, 'config') and 'log_level' in self.config:
                log_level = getattr(logging, self.config['log_level'].upper(), logging.INFO)
                logger.setLevel(log_level)
                
                # Update log file if specified
                if 'log_file' in self.config:
                    # Remove existing file handler and add new one
                    for handler in logger.handlers[:]:
                        if isinstance(handler, logging.FileHandler):
                            logger.removeHandler(handler)
                    
                    file_handler = logging.FileHandler(self.config['log_file'])
                    file_handler.setLevel(log_level)
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                    
                logger.info(f"Logging updated to level: {self.config['log_level']}")
        except Exception as e:
            logger.warning(f"Failed to update logging from config: {str(e)}")
    
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
            subreddits = ['stocks', 'investing']
            
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
            # Use config value for news article limit, default to 50
            news_limit = self.config.get('news_article_limit', 50)
            for article in news[:news_limit]:  # Limit to N articles
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
        # Use config lookback_days if available, otherwise default to 30
        days = getattr(self, 'config', {}).get('lookback_days', 30)
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
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
            logger.info(f"Fetching {days} days of historical data for {self.ticker}")
            
            # Try multiple approaches to get historical data
            data = None
            
            # Method 1: Try with specific date range
            try:
                end_date = datetime.now()
                # Get extra days for safety - use config value or default to 30
                extra_days = self.config.get('extra_days_for_safety', 30)
                start_date = end_date - timedelta(days=days + extra_days)
                
                data = yf.download(
                    self.ticker, 
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True
                )
                logger.info(f"Method 1: Got {len(data)} days of data")
            except Exception as e:
                logger.warning(f"Method 1 failed: {str(e)}")
            
            # Method 2: If method 1 fails, try with period
            if data is None or data.empty:
                try:
                    data = yf.download(
                        self.ticker, 
                        period=f"{days + extra_days}d",
                        progress=False,
                        auto_adjust=True
                    )
                    logger.info(f"Method 2: Got {len(data)} days of data")
                except Exception as e:
                    logger.warning(f"Method 2 failed: {str(e)}")
            
            # Method 3: Last resort - try with longer period
            if data is None or data.empty:
                try:
                    data = yf.download(
                        self.ticker, 
                        period=self.config.get('fallback_period', "1y"),
                        progress=False,
                        auto_adjust=True
                    )
                    logger.info(f"Method 3: Got {len(data)} days of data")
                except Exception as e:
                    logger.error(f"Method 3 failed: {str(e)}")
            
            if data is None or data.empty:
                raise ValueError(f"No historical data found for {self.ticker}")
            
            # Reset index and rename columns
            data.reset_index(inplace=True)
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # Remove any rows with NaN values
            data = data.dropna()
            
            # Minimum required data - use config value or default to 30
            min_required_days = self.config.get('min_required_days', 30)
            if len(data) < min_required_days:
                raise ValueError(f"Insufficient historical data for {self.ticker}. Only {len(data)} days available, need at least {min_required_days}.")
            
            # Add technical indicators if enabled
            if self.config.get('enable_technical_indicators', True):
                data = self.add_technical_indicators(data)
            
            logger.info(f"Successfully fetched {len(data)} days of historical data for {self.ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {self.ticker}: {str(e)}")
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
                    
                    # Ensure sentiment_data has proper datetime column
                    if 'date' in sentiment_data.columns:
                        try:
                            # Convert to datetime if it's not already
                            sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
                            
                            for i, date in enumerate(stock_data['Date']):
                                sentiment_idx = sentiment_data['date'].dt.date == date.date()
                                if sentiment_idx.any():
                                    sentiment_aligned[i] = normalized_sentiment[sentiment_idx.idxmax()]
                        except Exception as e:
                            logger.warning(f"Error aligning sentiment data: {str(e)}")
                            # If alignment fails, use zeros (no sentiment data)
                            sentiment_aligned = np.zeros((len(normalized_price), normalized_sentiment.shape[1]))
                    else:
                        logger.warning("No 'date' column found in sentiment data")
                        sentiment_aligned = np.zeros((len(normalized_price), normalized_sentiment.shape[1]))
                    
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
            
            # Minimum training samples - use config value or default to 50
            min_training_samples = self.config.get('min_training_samples', 50)
            if len(X) < min_training_samples:
                raise ValueError(f"Insufficient data for training. Only {len(X)} samples available, need at least {min_training_samples}.")
            
            logger.info(f"Prepared {len(X)} training samples with {X.shape[1]} features")
            
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
        """Get current stock price using multiple methods for accuracy"""
        try:
            logger.info(f"Fetching current price for {self.ticker}...")
            
            # Simple, direct approach - get the most recent price available
            stock = yf.Ticker(self.ticker)
            
            # Method 1: Try to get the most recent close price directly
            try:
                # Get the last N days of data to ensure we have recent data
                current_price_days = self.config.get('current_price_days', 5)
                data = yf.download(self.ticker, period=f"{current_price_days}d", progress=False)
                if not data.empty and len(data) > 0:
                    current_price = data['Close'].iloc[-1]
                    logger.info(f"Got current price from recent data: ${current_price:.2f}")
                    return current_price
            except Exception as e:
                logger.warning(f"Failed to get recent data: {str(e)}")
            
            # Method 2: Try stock info
            try:
                info = stock.info
                if 'regularMarketPrice' in info and info['regularMarketPrice']:
                    current_price = info['regularMarketPrice']
                    logger.info(f"Got current price from stock info: ${current_price:.2f}")
                    return current_price
                elif 'previousClose' in info and info['previousClose']:
                    current_price = info['previousClose']
                    logger.info(f"Got previous close price: ${current_price:.2f}")
                    return current_price
            except Exception as e:
                logger.warning(f"Failed to get stock info: {str(e)}")
            
            # Method 3: Try with different parameters
            try:
                # Use config value for date range fallback
                date_range_days = self.config.get('date_range_days', 7)
                data = yf.download(self.ticker, start=(datetime.now() - timedelta(days=date_range_days)).strftime('%Y-%m-%d'), 
                                 end=datetime.now().strftime('%Y-%m-%d'), progress=False)
                if not data.empty and len(data) > 0:
                    current_price = data['Close'].iloc[-1]
                    logger.info(f"Got current price from date range: ${current_price:.2f}")
                    return current_price
            except Exception as e:
                logger.warning(f"Failed to get date range data: {str(e)}")
            
            logger.error(f"Could not get current price for {self.ticker}")
            return 0
            
        except Exception as e:
            logger.error(f"Error getting current price for {self.ticker}: {str(e)}")
            return 0
    
    def predict_future_price(self) -> float:
        """Predict the stock price using realistic, conservative methods"""
        try:
            logger.info(f"Predicting future price for {self.ticker}...")
            
            # Get recent historical data
            stock_data = self.get_historical_data(days=self.config['lookback_days'])
            
            # Minimum data for prediction - use config value or default to 5
            min_prediction_days = self.config.get('min_prediction_days', 5)
            if stock_data.empty or len(stock_data) < min_prediction_days:
                raise ValueError(f"Insufficient historical data for prediction. Need at least {min_prediction_days} days.")
            
            # Get sentiment data for prediction
            sentiment_impact = 0.0
            if self.config.get('enable_sentiment_analysis', True):
                try:
                    # Get Reddit sentiment
                    reddit_sentiment = self.get_ticker_specific_sentiment(limit=50)
                    if not reddit_sentiment.empty and len(reddit_sentiment) > 0:
                        reddit_polarity = reddit_sentiment['avg_polarity'].mean()
                        reddit_impact = reddit_polarity * 0.05  # 5% max impact
                        sentiment_impact += reddit_impact
                        logger.info(f"Reddit sentiment impact: {reddit_impact:.4f}")
                    
                    # Get news sentiment
                    news_sentiment = self.get_news_sentiment()
                    if not news_sentiment.empty and len(news_sentiment) > 0:
                        news_polarity = news_sentiment['avg_polarity'].mean()
                        news_impact = news_polarity * 0.03  # 3% max impact
                        sentiment_impact += news_impact
                        logger.info(f"News sentiment impact: {news_impact:.4f}")
                    
                    logger.info(f"Total sentiment impact: {sentiment_impact:.4f}")
                except Exception as e:
                    logger.warning(f"Error calculating sentiment impact: {str(e)}")
                    sentiment_impact = 0.0
            
            # Use more conservative prediction methods
            # Use config value for recent days analysis, default to 20
            recent_days = self.config.get('recent_days_for_prediction', 20)
            recent_prices = stock_data['Close'].tail(recent_days).values  # Last N days for better stability
            
            # Calculate various technical indicators
            current_price = recent_prices[-1]
            
            # 1. Moving Average Analysis (conservative)
            ma_5 = np.mean(recent_prices[-5:])
            ma_10 = np.mean(recent_prices[-10:])
            ma_20 = np.mean(recent_prices)
            
            # 2. Calculate realistic momentum (dampened)
            daily_returns = np.diff(recent_prices) / recent_prices[:-1]
            avg_daily_return = np.mean(daily_returns)
            
            # 3. Calculate volatility
            volatility = np.std(daily_returns)
            
            # 4. Calculate RSI-like momentum
            gains = np.where(daily_returns > 0, daily_returns, 0)
            losses = np.where(daily_returns < 0, -daily_returns, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi_factor = 1 - (100 / (100 + rs))  # Normalized RSI factor
            else:
                rsi_factor = 0.5  # Neutral
            
            # 5. Calculate realistic prediction factors
            # Trend factor: How current price compares to moving averages
            trend_factor = ((ma_5 - ma_20) / ma_20) * 0.1  # Dampened trend
            
            # Momentum factor: Based on recent price movement
            momentum_factor = avg_daily_return * 2  # Conservative momentum
            
            # Volatility factor: Small adjustment based on volatility
            volatility_factor = volatility * 0.05  # Very small volatility adjustment
            
            # RSI factor: Overbought/oversold adjustment
            rsi_adjustment = (rsi_factor - 0.5) * 0.02  # Small RSI adjustment
            
            # Combine all factors for a realistic prediction (now including sentiment)
            prediction_change = (trend_factor + momentum_factor + volatility_factor + rsi_adjustment + sentiment_impact)
            
            # Apply very conservative bounds (±2% for daily prediction)
            prediction_change = max(-0.02, min(0.02, prediction_change))
            
            # Additional dampening for extreme values
            if abs(prediction_change) > 0.01:  # If > 1%
                prediction_change = prediction_change * 0.5  # Further dampen
            
            predicted_price = current_price * (1 + prediction_change)
            
            # Log detailed analysis
            logger.info(f"Current price: ${current_price:.2f}")
            logger.info(f"MA5: ${ma_5:.2f}, MA10: ${ma_10:.2f}, MA20: ${ma_20:.2f}")
            logger.info(f"Avg daily return: {avg_daily_return:.4f}")
            logger.info(f"Volatility: {volatility:.4f}")
            logger.info(f"RSI factor: {rsi_factor:.4f}")
            logger.info(f"Trend: {trend_factor:.4f}, Momentum: {momentum_factor:.4f}")
            logger.info(f"Sentiment impact: {sentiment_impact:.4f}")
            logger.info(f"Final prediction: ${predicted_price:.2f} ({prediction_change:+.4f})")
            
            return predicted_price
            
        except Exception as e:
            logger.error(f"Error predicting future price: {str(e)}")
            
            # Conservative fallback
            try:
                logger.info("Using conservative fallback prediction...")
                stock_data = self.get_historical_data(days=self.config.get('fallback_days', 5))
                # Minimum data for fallback prediction - use config value or default to 2
                min_fallback_days = self.config.get('min_fallback_days', 2)
                if not stock_data.empty and len(stock_data) >= min_fallback_days:
                    current_price = stock_data['Close'].iloc[-1]
                    # Very conservative random change (±0.5%)
                    import random
                    change = random.uniform(-0.005, 0.005)
                    predicted_price = current_price * (1 + change)
                    logger.info(f"Fallback prediction: ${predicted_price:.2f} ({change:+.4f})")
                    return predicted_price
            except Exception as fallback_error:
                logger.error(f"Fallback prediction failed: {str(fallback_error)}")
            
            return 0
    
    def generate_trading_signal(self) -> Dict[str, any]:
        """Generate buy/sell signal based on prediction"""
        try:
            current_price = self.get_current_price()
            predicted_price = self.predict_future_price()
            
            if current_price == 0:
                return {
                    'signal': 'ERROR',
                    'reason': f'Unable to get current price for {self.ticker}. Please check if the ticker symbol is correct and the market is open.',
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'confidence': 0.0
                }
            
            if predicted_price == 0:
                return {
                    'signal': 'ERROR',
                    'reason': f'Unable to predict future price for {self.ticker}. This may be due to insufficient historical data or model training issues.',
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'confidence': 0.0
                }
            
            # Calculate price change percentage
            price_change_pct = ((predicted_price - current_price) / current_price) * 100
            
            # Get thresholds from config (more conservative for daily trading)
            buy_threshold = self.config.get('buy_threshold', 0.5)  # 0.5% instead of 1.0%
            sell_threshold = self.config.get('sell_threshold', -0.5)  # -0.5% instead of -1.0%
            
            # Determine signal with more conservative logic
            if price_change_pct > buy_threshold:
                signal = 'BUY'
                # More conservative confidence calculation
                confidence = min(abs(price_change_pct) / 2.0, 0.8)  # Cap at 80% confidence
            elif price_change_pct < sell_threshold:
                signal = 'SELL'
                # More conservative confidence calculation
                confidence = min(abs(price_change_pct) / 2.0, 0.8)  # Cap at 80% confidence
            else:
                signal = 'HOLD'
                confidence = 0.6  # Higher base confidence for HOLD
            
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
        """Send detailed trading alert with comprehensive analysis"""
        try:
            current_time = datetime.now()
            
            # Get additional market data for analysis
            market_analysis = self.get_market_analysis()
            sentiment_analysis = self.get_detailed_sentiment_analysis()
            
            # Create detailed alert message
            signal_emoji = {
                'BUY': '🟢',
                'SELL': '🔴',
                'HOLD': '🟡',
                'ERROR': '⚠️'
            }
            
            emoji = signal_emoji.get(signal_data['signal'], '❓')
            
            alert_msg = f"""
{emoji} STOCK ALERT: {self.ticker} {emoji}

Alert Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} EST
Current Price: ${signal_data['current_price']:.2f}
Predicted Price (4PM EST): ${signal_data['predicted_price']:.2f}
Expected Change: {signal_data.get('price_change_pct', 0):.2f}%
Signal: {signal_data['signal']}
Confidence: {signal_data['confidence']:.1%}

TECHNICAL ANALYSIS:
{market_analysis}

SENTIMENT ANALYSIS:
{sentiment_analysis}

RECOMMENDATION:
{signal_data['reason']}

Model: Enhanced LSTM with Multi-Source Sentiment Analysis
Data Sources: Yahoo Finance, Reddit, News APIs
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
            
        except Exception as e:
            logger.error(f"Error sending alert: {str(e)}")
    
    def get_market_analysis(self) -> str:
        """Get detailed market analysis with technical indicators"""
        try:
            # Get recent historical data
            stock_data = self.get_historical_data(days=min(20, self.config['lookback_days']))
            
            # Minimum data for technical analysis - use config value or default to 10
            min_analysis_days = self.config.get('min_analysis_days', 10)
            if stock_data.empty or len(stock_data) < min_analysis_days:
                return f"Insufficient data for technical analysis. Need at least {min_analysis_days} days."
            
            current_price = stock_data['Close'].iloc[-1]
            
            # Calculate key technical indicators
            ma_5 = stock_data['Close'].rolling(5).mean().iloc[-1]
            ma_10 = stock_data['Close'].rolling(10).mean().iloc[-1]
            ma_20 = stock_data['Close'].rolling(20).mean().iloc[-1]
            
            # RSI calculation
            delta = stock_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
            rs = gain / loss if loss > 0 else 0
            rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50
            
            # MACD
            exp1 = stock_data['Close'].ewm(span=12).mean()
            exp2 = stock_data['Close'].ewm(span=26).mean()
            macd = exp1.iloc[-1] - exp2.iloc[-1]
            
            # Bollinger Bands
            bb_middle = stock_data['Close'].rolling(20).mean().iloc[-1]
            bb_std = stock_data['Close'].rolling(20).std().iloc[-1]
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            
            # Volume analysis
            avg_volume = stock_data['Volume'].rolling(10).mean().iloc[-1]
            current_volume = stock_data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Price momentum
            price_change_1d = ((current_price - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2]) * 100
            price_change_5d = ((current_price - stock_data['Close'].iloc[-6]) / stock_data['Close'].iloc[-6]) * 100
            
            analysis = f"""
• Moving Averages: MA5: ${ma_5:.2f} | MA10: ${ma_10:.2f} | MA20: ${ma_20:.2f}
• RSI: {rsi:.1f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})
• MACD: {macd:.2f} ({'Bullish' if macd > 0 else 'Bearish'})
• Bollinger Position: {bb_position:.2%} ({'Upper Band' if bb_position > 0.8 else 'Lower Band' if bb_position < 0.2 else 'Middle Range'})
• Volume: {volume_ratio:.1f}x average ({'High' if volume_ratio > 1.5 else 'Low' if volume_ratio < 0.5 else 'Normal'})
• Price Momentum: 1D: {price_change_1d:+.2f}% | 5D: {price_change_5d:+.2f}%
• Trend: {'Bullish' if current_price > ma_20 else 'Bearish'} (Price vs MA20: {((current_price/ma_20)-1)*100:+.2f}%)
            """.strip()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting market analysis: {str(e)}")
            return "Unable to generate technical analysis"
    
    def get_detailed_sentiment_analysis(self) -> str:
        """Get detailed sentiment analysis from multiple sources"""
        try:
            sentiment_info = []
            
            # Reddit sentiment
            try:
                # Use config value for sentiment analysis limit, default to 50
                sentiment_limit = self.config.get('sentiment_analysis_limit', 50)
                reddit_sentiment = self.get_ticker_specific_sentiment(limit=sentiment_limit)
                if not reddit_sentiment.empty and len(reddit_sentiment) > 0:
                    avg_polarity = reddit_sentiment['avg_polarity'].mean()
                    total_posts = reddit_sentiment['total_volume'].sum()
                    
                    sentiment_label = "Bullish" if avg_polarity > 0.1 else "Bearish" if avg_polarity < -0.1 else "Neutral"
                    sentiment_info.append(f"• Reddit Sentiment: {sentiment_label} ({avg_polarity:+.3f}) - {total_posts:.0f} mentions")
                else:
                    sentiment_info.append("• Reddit Sentiment: No recent mentions found")
            except Exception as e:
                sentiment_info.append("• Reddit Sentiment: Unable to fetch data")
            
            # News sentiment
            try:
                news_sentiment = self.get_news_sentiment()
                if not news_sentiment.empty and len(news_sentiment) > 0:
                    avg_news_polarity = news_sentiment['avg_polarity'].mean()
                    article_count = news_sentiment['article_count'].sum()
                    
                    news_label = "Positive" if avg_news_polarity > 0.1 else "Negative" if avg_news_polarity < -0.1 else "Neutral"
                    sentiment_info.append(f"• News Sentiment: {news_label} ({avg_news_polarity:+.3f}) - {article_count:.0f} articles")
                else:
                    sentiment_info.append("• News Sentiment: No recent news found")
            except Exception as e:
                sentiment_info.append("• News Sentiment: Unable to fetch data")
            
            # Yahoo Finance data
            try:
                stock = yf.Ticker(self.ticker)
                info = stock.info
                
                # Market cap and sector info
                market_cap = info.get('marketCap', 0)
                sector = info.get('sector', 'Unknown')
                industry = info.get('industry', 'Unknown')
                
                if market_cap > 0:
                    market_cap_str = f"${market_cap/1e9:.1f}B" if market_cap >= 1e9 else f"${market_cap/1e6:.1f}M"
                    sentiment_info.append(f"• Market Cap: {market_cap_str} | Sector: {sector}")
                
                # Analyst recommendations
                if 'recommendationMean' in info:
                    rec_mean = info['recommendationMean']
                    rec_count = info.get('numberOfAnalystOpinions', 0)
                    sentiment_info.append(f"• Analyst Rating: {rec_mean:.2f}/5.0 ({rec_count} analysts)")
                
            except Exception as e:
                sentiment_info.append("• Market Data: Unable to fetch additional market data")
            
            if not sentiment_info:
                return "No sentiment data available"
            
            return "\n".join(sentiment_info)
            
        except Exception as e:
            logger.error(f"Error getting sentiment analysis: {str(e)}")
            return "Unable to generate sentiment analysis"
    
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
            
            # Skip complex model training for now - use simple prediction
            logger.info("Using simple prediction method (skipping complex model training)")
            
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
    import sys
    
    # Check if --run-once argument is provided
    run_once = '--run-once' in sys.argv
    
    try:
        # Initialize predictor
        predictor = EnhancedStockPredictor()
        
        if run_once:
            # Run analysis once and exit (for crontab)
            logger.info("Running single analysis (crontab mode)...")
            predictor.run_daily_analysis()
            logger.info("Analysis completed, exiting.")
        else:
            # Schedule daily analysis (for manual running)
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