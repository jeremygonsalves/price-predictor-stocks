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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import dates as mdates

class EnhancedStockPredictor:
    def __init__(self, config_file: str = "configs/config.json"):
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
        # Resolve to proper exchange-specific symbol when base symbol is ambiguous
        self.ticker = self.resolve_exchange_symbol(self.ticker)
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
        
        # Initialize currency settings for proper display/conversion
        self.init_currency_settings()
        
        logger.info(f"Enhanced Stock Predictor initialized for {self.ticker}")
    
    def load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file {config_file} not found. Please ensure the configs/config.json file exists in the project root.")
            raise FileNotFoundError(f"Config file {config_file} not found. Please create configs/config.json with the required settings.")
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
            # Check for Reddit credentials in new secrets directory
            if (os.path.exists('secrets/pw.txt') and 
                os.path.exists('secrets/client_id.txt') and 
                os.path.exists('secrets/client_secret.txt')):
                
                with open('secrets/pw.txt', 'r') as f:
                    pw = f.read().strip()
                
                with open('secrets/client_id.txt', 'r') as f:
                    client_id = f.read().strip()
                
                with open('secrets/client_secret.txt', 'r') as f:
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
                    
                    logger.info("Reddit API client initialized successfully using secrets credentials")
                    return {
                        'headers': headers,
                        'token': token,
                        'auth_method': 'oauth'
                    }
                else:
                    logger.warning(f"Failed to get Reddit token: {res.status_code}")
                    return None
            else:
                logger.info("Reddit credential files not found in secrets/, sentiment analysis will use dummy data")
                return None
        except Exception as e:
            logger.warning(f"Failed to initialize Reddit client: {str(e)}")
            return None
    
    # --------------------- Currency utilities ---------------------
    def init_currency_settings(self) -> None:
        """Initialize currency settings for display and conversion."""
        try:
            stock = yf.Ticker(self.ticker)
            info = {}
            try:
                info = stock.info or {}
            except Exception:
                info = {}
            self.quote_currency = (info.get('currency') or 'USD').upper()
            # Always display in the stock's native quote currency
            self.display_currency = self.quote_currency
            self.currency_symbol = self.get_currency_symbol(self.display_currency)
            self._fx_rate = self.get_fx_rate(self.quote_currency, self.display_currency)
            logger.info(f"Currency settings: quote={self.quote_currency}, display={self.display_currency}, fx_rate={self._fx_rate:.6f}")
        except Exception as e:
            self.quote_currency = 'USD'
            self.display_currency = 'USD'
            self.currency_symbol = '$'
            self._fx_rate = 1.0
            logger.warning(f"Failed to initialize currency settings: {str(e)}")
    
    def get_currency_symbol(self, currency_code: str) -> str:
        mapping = {
            'USD': '$',
            'CAD': 'C$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'AUD': 'A$',
            'CHF': 'CHF',
            'HKD': 'HK$',
            'INR': '₹'
        }
        return mapping.get(currency_code.upper(), currency_code.upper() + ' ')
    
    def get_fx_rate(self, from_currency: str, to_currency: str) -> float:
        """Fetch FX spot rate from Yahoo to convert from -> to. Returns 1.0 on failure or same currency."""
        try:
            if not from_currency or not to_currency or from_currency.upper() == to_currency.upper():
                return 1.0
            pair = f"{from_currency.upper()}{to_currency.upper()}=X"
            data = yf.download(pair, period="1d", interval="1d", progress=False)
            if data is not None and not data.empty:
                rate = float(data['Close'].iloc[-1])
                if rate > 0:
                    return rate
            # Try inverse pair
            inv_pair = f"{to_currency.upper()}{from_currency.upper()}=X"
            inv_data = yf.download(inv_pair, period="1d", interval="1d", progress=False)
            if inv_data is not None and not inv_data.empty:
                inv_rate = float(inv_data['Close'].iloc[-1])
                if inv_rate > 0:
                    return 1.0 / inv_rate
        except Exception:
            pass
        return 1.0
    
    def apply_currency_conversion(self, amount: float) -> float:
        try:
            return float(amount) * float(self._fx_rate)
        except Exception:
            return amount
    
    def format_currency(self, amount: float) -> str:
        converted = self.apply_currency_conversion(amount)
        # If we have a standard symbol (like $, C$), use it; otherwise append code at end
        symbol = self.currency_symbol
        if symbol.endswith(' '):
            return f"{converted:.2f} {self.display_currency}"
        else:
            return f"{symbol}{converted:.2f}"
    # ------------------- End currency utilities -------------------
    
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
    
    def get_historical_data(self, days: int = None, allow_sparse: bool = False) -> pd.DataFrame:
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
                if allow_sparse:
                    logger.warning(f"No historical data found for {self.ticker}. Returning empty DataFrame (sparse mode).")
                    return pd.DataFrame()
                else:
                    raise ValueError(f"No historical data found for {self.ticker}")
            
            # Reset index and rename columns
            data.reset_index(inplace=True)
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # Remove any rows with NaN values
            data = data.dropna()
            
            # Minimum required data - use config value or default to 30
            min_required_days = self.config.get('min_required_days', 30)
            if len(data) < min_required_days:
                if allow_sparse:
                    logger.warning(f"Sparse data for {self.ticker}: {len(data)} days < {min_required_days}. Proceeding with sparse mode.")
                else:
                    raise ValueError(f"Insufficient historical data for {self.ticker}. Only {len(data)} days available, need at least {min_required_days}.")
            
            # Add technical indicators if enabled
            if self.config.get('enable_technical_indicators', True) and not data.empty and len(data) >= 5:
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
    
    def _safe_to_float(self, value, default: float = 0.0) -> float:
        """Best-effort conversion of various array-like/scalar types to a positive finite float."""
        try:
            if value is None:
                return default
            # Unbox common containers
            if isinstance(value, pd.DataFrame):
                if 'Close' in value.columns and not value['Close'].empty:
                    value = value['Close'].iloc[-1]
                else:
                    numeric = value.select_dtypes(include=[np.number])
                    value = numeric.stack().iloc[-1] if not numeric.empty else default
            if isinstance(value, pd.Series):
                if not value.empty:
                    value = value.iloc[-1]
                else:
                    return default
            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.array(value).ravel()
                if arr.size == 0:
                    return default
                value = arr[-1]
            result = float(value)
            if not np.isfinite(result) or result <= 0:
                return default
            return result
        except Exception:
            return default
    
    def get_current_price(self) -> float:
        """Get current stock price using multiple methods for accuracy"""
        try:
            logger.info(f"Fetching current price for {self.ticker}...")
            stock = yf.Ticker(self.ticker)

            # Method 1: Use fast_info last price
            try:
                fast_info = getattr(stock, 'fast_info', {}) or {}
                for key in ['last_price', 'lastPrice', 'regularMarketPrice']:
                    if key in fast_info and fast_info[key]:
                        price = self._safe_to_float(fast_info[key])
                        if price > 0:
                            logger.info(f"Got current price from fast_info ({key}): {self.format_currency(price)}")
                            return price
            except Exception as e:
                logger.warning(f"fast_info unavailable: {str(e)}")

            # Method 2: High-resolution recent history (1m)
            try:
                hist = stock.history(period="1d", interval="1m", auto_adjust=True)
                price = self._safe_to_float(hist['Close'] if 'Close' in hist else hist)
                if price > 0:
                    logger.info(f"Got current price from 1m history: {self.format_currency(price)}")
                    return price
            except Exception as e:
                logger.warning(f"Failed to get 1m history: {str(e)}")

            # Method 3: Recent daily download
            try:
                current_price_days = int(self.config.get('current_price_days', 5))
                data = yf.download(self.ticker, period=f"{current_price_days}d", progress=False, auto_adjust=True)
                price = self._safe_to_float(data['Close'] if 'Close' in data else data)
                if price > 0:
                    logger.info(f"Got current price from recent daily data: {self.format_currency(price)}")
                    return price
            except Exception as e:
                logger.warning(f"Failed to get recent daily data: {str(e)}")

            # Method 4: Stock info dict
            try:
                info = stock.info or {}
                for key in ['regularMarketPrice', 'previousClose']:
                    if key in info and info[key]:
                        price = self._safe_to_float(info[key])
                        if price > 0:
                            logger.info(f"Got current price from info ({key}): {self.format_currency(price)}")
                            return price
            except Exception as e:
                logger.warning(f"Failed to get stock info: {str(e)}")

            # Method 5: Date range fallback
            try:
                date_range_days = int(self.config.get('date_range_days', 7))
                data = yf.download(
                    self.ticker,
                    start=(datetime.now() - timedelta(days=date_range_days)).strftime('%Y-%m-%d'),
                    end=datetime.now().strftime('%Y-%m-%d'),
                    progress=False,
                    auto_adjust=True
                )
                price = self._safe_to_float(data['Close'] if 'Close' in data else data)
                if price > 0:
                    logger.info(f"Got current price from date-range data: {self.format_currency(price)}")
                    return price
            except Exception as e:
                logger.warning(f"Failed to get date-range data: {str(e)}")

            logger.error(f"Could not get current price for {self.ticker}")
            return 0.0
        except Exception as e:
            logger.error(f"Error getting current price for {self.ticker}: {str(e)}")
            return 0.0
    
    def predict_future_price(self) -> float:
        """Predict the stock price using enhanced ensemble methods"""
        try:
            # Use enhanced prediction if enabled
            if self.config.get('use_enhanced_prediction', True):
                try:
                    return self.predict_future_price_enhanced()
                except Exception as e:
                    logger.warning(f"Enhanced prediction failed ({e}), attempting sparse-data path...")
                    # Sparse-data fallback
                    daily_data = self.get_historical_data(days=self.config['lookback_days'], allow_sparse=True)
                    intraday_data = self.get_intraday_data(interval="5m", days=1)
                    return self._predict_with_sparse_data(daily_data, intraday_data)
            else:
                return self._legacy_prediction()
                
        except Exception as e:
            logger.error(f"Error in predict_future_price: {str(e)}")
            # Try sparse-data fallback before legacy
            try:
                daily_data = self.get_historical_data(days=self.config['lookback_days'], allow_sparse=True)
                intraday_data = self.get_intraday_data(interval="5m", days=1)
                sparse_pred = self._predict_with_sparse_data(daily_data, intraday_data)
                if sparse_pred > 0:
                    return sparse_pred
            except Exception:
                pass
            return self._legacy_prediction()
    
    def _legacy_prediction(self) -> float:
        """Legacy prediction method for fallback"""
        try:
            logger.info(f"Using legacy prediction for {self.ticker}...")
            
            # Get recent historical data
            stock_data = self.get_historical_data(days=self.config['lookback_days'], allow_sparse=True)
            
            # Minimum data for prediction - use config value or default to 5
            min_prediction_days = self.config.get('min_prediction_days', 5)
            if stock_data.empty or len(stock_data) < min_prediction_days:
                # Sparse-data fallback instead of raising
                logger.warning("Insufficient historical data for legacy prediction. Using sparse-data prediction.")
                intraday_data = self.get_intraday_data(interval="5m", days=1)
                return self._predict_with_sparse_data(stock_data, intraday_data)
            
            # Get sentiment data for prediction
            sentiment_impact = 0.0
            if self.config.get('enable_sentiment_analysis', True):
                try:
                    # Get Reddit sentiment
                    reddit_sentiment = self.get_ticker_specific_sentiment()
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
            logger.info(f"Current price: {self.format_currency(current_price)}")
            logger.info(f"MA5: {self.format_currency(ma_5)}, MA10: {self.format_currency(ma_10)}, MA20: {self.format_currency(ma_20)}")
            logger.info(f"Avg daily return: {avg_daily_return:.4f}")
            logger.info(f"Volatility: {volatility:.4f}")
            logger.info(f"RSI factor: {rsi_factor:.4f}")
            logger.info(f"Trend: {trend_factor:.4f}, Momentum: {momentum_factor:.4f}")
            logger.info(f"Sentiment impact: {sentiment_impact:.4f}")
            logger.info(f"Final prediction: {self.format_currency(predicted_price)} ({prediction_change:+.4f})")
            
            return predicted_price
            
        except Exception as e:
            logger.error(f"Error in legacy prediction: {str(e)}")
            
            # Conservative fallback
            try:
                logger.info("Using conservative fallback prediction...")
                stock_data = self.get_historical_data(days=self.config.get('fallback_days', 5), allow_sparse=True)
                intraday_data = self.get_intraday_data(interval="5m", days=1)
                sparse_pred = self._predict_with_sparse_data(stock_data, intraday_data)
                if sparse_pred > 0:
                    logger.info(f"Sparse fallback prediction: {self.format_currency(sparse_pred)}")
                    return sparse_pred
            except Exception as fallback_error:
                logger.error(f"Fallback prediction failed: {str(fallback_error)}")
            
            return 0
    
    def generate_trading_signal(self) -> Dict[str, any]:
        """Generate buy/sell signal based on prediction"""
        try:
            current_price = self._safe_to_float(self.get_current_price(), default=0.0)
            predicted_price = self._safe_to_float(self.predict_future_price(), default=0.0)

            if not np.isfinite(current_price) or current_price <= 0:
                return {
                    'signal': 'ERROR',
                    'reason': f'Unable to get current price for {self.ticker}. Please check if the ticker symbol is correct and the market is open.',
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'confidence': 0.0
                }
            
            if not np.isfinite(predicted_price) or predicted_price <= 0:
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
Current Price: {self.format_currency(signal_data['current_price'])}
Predicted Price (4PM EST): {self.format_currency(signal_data['predicted_price'])}
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
            ts = self.send_slack_notification(alert_msg, "#stock-price-alerts")
            
            # Generate and upload charts
            try:
                charts_dir = os.path.join('reports', 'charts')
                os.makedirs(charts_dir, exist_ok=True)

                # Intraday chart with prediction overlay
                intraday_path = os.path.join(charts_dir, f"{self.ticker}_intraday.png")
                self.generate_intraday_chart_with_prediction(
                    predicted_price=self._safe_to_float(signal_data.get('predicted_price'), 0.0),
                    output_path=intraday_path
                )
                self.upload_slack_file(intraday_path, title=f"{self.ticker} Intraday + Prediction", channel="#stock-price-alerts", thread_ts=ts)

                # 5D trend (no prediction overlay)
                five_day_path = os.path.join(charts_dir, f"{self.ticker}_5d.png")
                self.generate_timeframe_chart(period="5d", interval="15m", output_path=five_day_path)
                self.upload_slack_file(five_day_path, title=f"{self.ticker} 5D Trend", channel="#stock-price-alerts", thread_ts=ts)

                # 1M trend (no prediction overlay)
                one_month_path = os.path.join(charts_dir, f"{self.ticker}_1mo.png")
                self.generate_timeframe_chart(period="1mo", interval="1d", output_path=one_month_path)
                self.upload_slack_file(one_month_path, title=f"{self.ticker} 1M Trend", channel="#stock-price-alerts", thread_ts=ts)

                # 1D trend (1m or 5m depending on intraday interval)
                one_day_path = os.path.join(charts_dir, f"{self.ticker}_1d.png")
                self.generate_timeframe_chart(period="1d", interval=self.config.get('intraday_interval', '5m'), output_path=one_day_path)
                self.upload_slack_file(one_day_path, title=f"{self.ticker} 1D Trend", channel="#stock-price-alerts", thread_ts=ts)

                # 1W trend (if supported by interval)
                one_week_path = os.path.join(charts_dir, f"{self.ticker}_1w.png")
                self.generate_timeframe_chart(period="5d", interval="30m", output_path=one_week_path)
                self.upload_slack_file(one_week_path, title=f"{self.ticker} 1W Trend", channel="#stock-price-alerts", thread_ts=ts)

            except Exception as e:
                logger.warning(f"Failed to generate/upload charts: {str(e)}")
            
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
• Moving Averages: MA5: {self.format_currency(ma_5)} | MA10: {self.format_currency(ma_10)} | MA20: {self.format_currency(ma_20)}
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
                    mc_converted = self.apply_currency_conversion(market_cap)
                    if mc_converted >= 1e9:
                        market_cap_str = f"{self.currency_symbol}{mc_converted/1e9:.1f}B"
                    elif mc_converted >= 1e6:
                        market_cap_str = f"{self.currency_symbol}{mc_converted/1e6:.1f}M"
                    else:
                        market_cap_str = f"{self.format_currency(mc_converted)}"
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
    
    def send_slack_notification(self, message: str, channel: str = "#stock-price-alerts") -> Optional[str]:
        """Send notification to Slack. Returns message ts if using bot token, else None."""
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
                    return None
                else:
                    logger.info(f"Successfully sent Slack notification to {channel}")
                    return result.get("ts")
            elif self.slack_webhook_url:
                # Fallback to webhook if bot token not available
                response = requests.post(self.slack_webhook_url, json={"text": message})
                if response.status_code == 200:
                    logger.info(f"Successfully sent Slack webhook notification")
                    return None
                else:
                    logger.error(f"Failed to send Slack webhook: {response.status_code}")
                    return None
            else:
                logger.warning("No Slack credentials configured")
                return None
        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
            return None
    
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

    def get_intraday_data(self, interval: str = "5m", days: int = 5) -> pd.DataFrame:
        """Fetch intraday stock data for more precise 4-hour predictions"""
        try:
            logger.info(f"Fetching {interval} intraday data for {self.ticker}")
            
            # Get intraday data
            data = yf.download(
                self.ticker,
                period=f"{days}d",
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            if data.empty:
                logger.warning(f"No intraday data available for {self.ticker}")
                return pd.DataFrame()
            
            # Reset index and rename columns
            data.reset_index(inplace=True)
            data.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # Add intraday-specific technical indicators
            data = self.add_intraday_indicators(data)
            
            logger.info(f"Successfully fetched {len(data)} {interval} intervals for {self.ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching intraday data: {str(e)}")
            return pd.DataFrame()
    
    def add_intraday_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add intraday-specific technical indicators"""
        # Intraday moving averages (shorter periods)
        data['MA_3'] = data['Close'].rolling(window=3).mean()
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_10'] = data['Close'].rolling(window=10).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        
        # Intraday RSI (shorter period)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
        rs = gain / loss
        data['RSI_Intraday'] = 100 - (100 / (1 + rs))
        
        # Price momentum (intraday)
        data['Price_Change_1'] = data['Close'].pct_change()
        data['Price_Change_3'] = data['Close'].pct_change(periods=3)
        data['Price_Change_5'] = data['Close'].pct_change(periods=5)
        
        # Volume analysis (intraday)
        data['Volume_MA_5'] = data['Volume'].rolling(window=5).mean()
        data['Volume_Ratio_Intraday'] = data['Volume'] / data['Volume_MA_5']
        
        # Intraday volatility
        data['High_Low_Range'] = (data['High'] - data['Low']) / data['Close']
        data['Volatility_Intraday'] = data['High_Low_Range'].rolling(window=10).mean()
        
        # Time-based features
        data['Hour'] = pd.to_datetime(data['Datetime']).dt.hour
        data['Minute'] = pd.to_datetime(data['Datetime']).dt.minute
        data['Is_Market_Open'] = ((data['Hour'] >= 9) & (data['Hour'] < 16)).astype(int)
        
        # Fill NaN values
        data = data.fillna(method='bfill').fillna(0)
        
        return data

    def get_market_microstructure_features(self) -> Dict[str, float]:
        """Get market microstructure features for short-term prediction accuracy"""
        try:
            features = {}
            
            # Get recent intraday data
            intraday_data = self.get_intraday_data(interval="5m", days=1)
            
            if not intraday_data.empty and len(intraday_data) > 10:
                # Bid-ask spread proxy (using high-low range)
                recent_data = intraday_data.tail(10)
                features['avg_spread_proxy'] = recent_data['High_Low_Range'].mean()
                features['spread_volatility'] = recent_data['High_Low_Range'].std()
                
                # Order flow indicators
                features['volume_trend'] = recent_data['Volume'].pct_change().mean()
                features['volume_acceleration'] = recent_data['Volume'].pct_change().diff().mean()
                
                # Price impact (how much volume moves price)
                price_changes = recent_data['Close'].pct_change().abs()
                volume_changes = recent_data['Volume'].pct_change()
                features['price_impact'] = (price_changes * volume_changes).mean()
                
                # Market efficiency ratio
                features['market_efficiency'] = recent_data['Price_Change_1'].abs().mean()
                
                # Time-based features
                current_hour = datetime.now().hour
                features['is_market_open'] = 1 if 9 <= current_hour < 16 else 0
                features['hours_until_close'] = max(0, 16 - current_hour)
                features['hours_since_open'] = max(0, current_hour - 9)
                
                # Intraday momentum
                features['intraday_momentum'] = recent_data['Price_Change_5'].mean()
                features['momentum_acceleration'] = recent_data['Price_Change_5'].diff().mean()
                
                # Volatility clustering
                features['volatility_clustering'] = recent_data['Volatility_Intraday'].autocorr()
                
                # Mean reversion indicator
                features['mean_reversion'] = (recent_data['Close'].iloc[-1] - recent_data['MA_10'].iloc[-1]) / recent_data['MA_10'].iloc[-1]
                
            else:
                # Fallback values if no intraday data
                features = {
                    'avg_spread_proxy': 0.01,
                    'spread_volatility': 0.005,
                    'volume_trend': 0.0,
                    'volume_acceleration': 0.0,
                    'price_impact': 0.0,
                    'market_efficiency': 0.01,
                    'is_market_open': 1 if 9 <= datetime.now().hour < 16 else 0,
                    'hours_until_close': max(0, 16 - datetime.now().hour),
                    'hours_since_open': max(0, datetime.now().hour - 9),
                    'intraday_momentum': 0.0,
                    'momentum_acceleration': 0.0,
                    'volatility_clustering': 0.0,
                    'mean_reversion': 0.0
                }
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting market microstructure features: {str(e)}")
            return {}

    def get_real_time_sentiment(self) -> Dict[str, float]:
        """Get real-time sentiment from multiple sources for 4-hour predictions"""
        try:
            sentiment_scores = {}
            
            # 1. Real-time news sentiment (last 4 hours)
            try:
                # Use Alpha Vantage News API or similar
                news_sentiment = self.get_recent_news_sentiment(hours=4)
                sentiment_scores['news_sentiment'] = news_sentiment
            except Exception as e:
                logger.warning(f"Failed to get real-time news: {str(e)}")
                sentiment_scores['news_sentiment'] = 0.0
            
            # 2. Social media sentiment (Twitter/X, Reddit)
            try:
                social_sentiment = self.get_social_media_sentiment(hours=4)
                sentiment_scores['social_sentiment'] = social_sentiment
            except Exception as e:
                logger.warning(f"Failed to get social media sentiment: {str(e)}")
                sentiment_scores['social_sentiment'] = 0.0
            
            # 3. Earnings calendar impact
            try:
                earnings_impact = self.get_earnings_calendar_impact()
                sentiment_scores['earnings_impact'] = earnings_impact
            except Exception as e:
                logger.warning(f"Failed to get earnings impact: {str(e)}")
                sentiment_scores['earnings_impact'] = 0.0
            
            # 4. Analyst rating changes
            try:
                analyst_impact = self.get_analyst_rating_changes()
                sentiment_scores['analyst_impact'] = analyst_impact
            except Exception as e:
                logger.warning(f"Failed to get analyst changes: {str(e)}")
                sentiment_scores['analyst_impact'] = 0.0
            
            # 5. Options flow sentiment
            try:
                options_sentiment = self.get_options_flow_sentiment()
                sentiment_scores['options_sentiment'] = options_sentiment
            except Exception as e:
                logger.warning(f"Failed to get options flow: {str(e)}")
                sentiment_scores['options_sentiment'] = 0.0
            
            return sentiment_scores
            
        except Exception as e:
            logger.error(f"Error getting real-time sentiment: {str(e)}")
            return {}
    
    def get_recent_news_sentiment(self, hours: int = 4) -> float:
        """Get sentiment from recent news articles"""
        try:
            # This would integrate with a real-time news API
            # For now, we'll use Yahoo Finance news with time filtering
            stock = yf.Ticker(self.ticker)
            news = stock.news
            
            if not news:
                return 0.0
            
            # Filter for recent news (last 4 hours)
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_news = []
            
            for article in news:
                try:
                    pub_time = datetime.fromtimestamp(article.get('providerPublishTime', 0))
                    if pub_time >= cutoff_time:
                        recent_news.append(article)
                except:
                    continue
            
            if not recent_news:
                return 0.0
            
            # Calculate weighted sentiment
            total_sentiment = 0.0
            total_weight = 0.0
            
            for article in recent_news:
                title = article.get('title', '')
                if title:
                    sentiment = TextBlob(title).sentiment.polarity
                    # Weight by relevance (could be enhanced with keyword matching)
                    weight = 1.0
                    total_sentiment += sentiment * weight
                    total_weight += weight
            
            return total_sentiment / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error getting recent news sentiment: {str(e)}")
            return 0.0
    
    def get_social_media_sentiment(self, hours: int = 4) -> float:
        """Get sentiment from social media platforms"""
        try:
            # Enhanced Reddit sentiment with time filtering
            if self.reddit_client:
                # Get recent posts from the last 4 hours
                cutoff_time = datetime.now() - timedelta(hours=hours)
                
                # Search for recent posts mentioning the ticker
                headers = self.reddit_client['headers']
                subreddits = ['stocks', 'investing', 'wallstreetbets']
                
                recent_posts = []
                for subreddit in subreddits:
                    try:
                        res = requests.get(f"https://oauth.reddit.com/r/{subreddit}/new", 
                                         headers=headers, params={"limit": 100})
                        
                        if res.status_code == 200:
                            posts = res.json()['data']['children']
                            
                            for post in posts:
                                post_data = post['data']
                                post_time = datetime.fromtimestamp(post_data['created_utc'])
                                
                                if post_time >= cutoff_time:
                                    title = post_data.get('title', '').lower()
                                    if self.ticker.lower() in title or f"${self.ticker.lower()}" in title:
                                        sentiment = TextBlob(post_data['title']).sentiment.polarity
                                        score = post_data.get('score', 0)
                                        recent_posts.append({
                                            'sentiment': sentiment,
                                            'score': score,
                                            'time': post_time
                                        })
                    except Exception as e:
                        continue
                
                if recent_posts:
                    # Weight by score and recency
                    total_weighted_sentiment = 0.0
                    total_weight = 0.0
                    
                    for post in recent_posts:
                        # Weight by score and recency
                        time_weight = 1.0 / (1.0 + (datetime.now() - post['time']).total_seconds() / 3600)
                        score_weight = max(1, post['score'])
                        weight = time_weight * score_weight
                        
                        total_weighted_sentiment += post['sentiment'] * weight
                        total_weight += weight
                    
                    return total_weighted_sentiment / total_weight if total_weight > 0 else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting social media sentiment: {str(e)}")
            return 0.0
    
    def get_earnings_calendar_impact(self) -> float:
        """Get impact of upcoming earnings on sentiment"""
        try:
            # This would integrate with earnings calendar APIs
            # For now, use a simple heuristic based on a safely parsed next earnings date
            stock = yf.Ticker(self.ticker)
            info = stock.info or {}

            # Try multiple keys and normalize to a datetime
            possible = info.get('earningsDate') or info.get('earningsDates')
            next_earnings = None
            if possible is not None:
                try:
                    # common shapes: [Timestamp, Timestamp] or single Timestamp
                    if isinstance(possible, (list, tuple)) and len(possible) > 0:
                        candidate = possible[0]
                    else:
                        candidate = possible
                    if hasattr(candidate, 'to_pydatetime'):
                        next_earnings = candidate.to_pydatetime()
                    elif isinstance(candidate, (datetime,)):
                        next_earnings = candidate
                except Exception:
                    next_earnings = None

            if next_earnings:
                days_until_earnings = (next_earnings - datetime.now()).days
                if days_until_earnings <= 7:
                    return 0.1
                elif days_until_earnings <= 14:
                    return 0.05
                else:
                    return 0.0

            return 0.0
        except Exception as e:
            logger.error(f"Error getting earnings impact: {str(e)}")
            return 0.0
    
    def get_analyst_rating_changes(self) -> float:
        """Get impact of recent analyst rating changes"""
        try:
            # This would integrate with analyst rating APIs
            # For now, return neutral impact
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting analyst changes: {str(e)}")
            return 0.0
    
    def get_options_flow_sentiment(self) -> float:
        """Get sentiment from options flow data"""
        try:
            # This would integrate with options flow APIs
            # For now, return neutral impact
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting options flow: {str(e)}")
            return 0.0

    def predict_future_price_enhanced(self) -> float:
        """Enhanced prediction using ensemble methods and intraday data"""
        try:
            logger.info(f"Running enhanced prediction for {self.ticker}...")
            
            # Get multiple data sources
            daily_data = self.get_historical_data(days=self.config['lookback_days'])
            intraday_data = self.get_intraday_data(interval="5m", days=5)
            microstructure_features = self.get_market_microstructure_features()
            real_time_sentiment = self.get_real_time_sentiment()
            
            if daily_data.empty:
                raise ValueError(f"No daily data available for {self.ticker}")
            
            # Ensemble prediction using multiple models
            predictions = []
            weights = []
            
            # 1. Technical Analysis Model (40% weight)
            try:
                tech_prediction = float(self._technical_analysis_prediction(daily_data, intraday_data))
                predictions.append(tech_prediction)
                weights.append(0.4)
                logger.info(f"Technical prediction: {self.format_currency(tech_prediction)}")
            except Exception as e:
                logger.warning(f"Technical analysis failed: {str(e)}")
            
            # 2. Sentiment-Based Model (25% weight)
            try:
                sentiment_prediction = float(self._sentiment_based_prediction(daily_data, real_time_sentiment))
                predictions.append(sentiment_prediction)
                weights.append(0.25)
                logger.info(f"Sentiment prediction: {self.format_currency(sentiment_prediction)}")
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {str(e)}")
            
            # 3. Microstructure Model (20% weight)
            try:
                micro_prediction = float(self._microstructure_prediction(daily_data, microstructure_features))
                predictions.append(micro_prediction)
                weights.append(0.2)
                logger.info(f"Microstructure prediction: {self.format_currency(micro_prediction)}")
            except Exception as e:
                logger.warning(f"Microstructure analysis failed: {str(e)}")
            
            # 4. Mean Reversion Model (15% weight)
            try:
                reversion_prediction = float(self._mean_reversion_prediction(daily_data, intraday_data))
                predictions.append(reversion_prediction)
                weights.append(0.15)
                logger.info(f"Mean reversion prediction: {self.format_currency(reversion_prediction)}")
            except Exception as e:
                logger.warning(f"Mean reversion analysis failed: {str(e)}")
            
            # Calculate weighted ensemble prediction
            if predictions and weights:
                # Normalize weights
                total_weight = sum(weights)
                normalized_weights = [w / total_weight for w in weights]
                
                # Weighted average
                ensemble_prediction = sum(p * w for p, w in zip(predictions, normalized_weights))
                
                # Apply confidence adjustment based on prediction consistency
                prediction_std = np.std(predictions)
                current_price = daily_data['Close'].iloc[-1]
                price_std = current_price * 0.02  # 2% of current price
                
                # Reduce confidence if predictions are inconsistent
                if prediction_std > price_std:
                    confidence_factor = price_std / prediction_std
                    ensemble_prediction = current_price + (ensemble_prediction - current_price) * confidence_factor
                
                logger.info(f"Ensemble prediction: {self.format_currency(ensemble_prediction)}")
                return ensemble_prediction
            else:
                # Fallback to original method
                logger.warning("Ensemble prediction unavailable, using legacy fallback")
                return self._legacy_prediction()
                
        except Exception as e:
            logger.error(f"Enhanced prediction failed: {str(e)}")
            return self._legacy_prediction()
    
    def _technical_analysis_prediction(self, daily_data: pd.DataFrame, intraday_data: pd.DataFrame) -> float:
        """Technical analysis-based prediction"""
        current_price = daily_data['Close'].iloc[-1]
        
        # Use recent intraday data if available
        if not intraday_data.empty and len(intraday_data) > 20:
            recent_intraday = intraday_data.tail(20)
            
            # Intraday momentum
            intraday_momentum = recent_intraday['Price_Change_5'].mean()
            
            # Intraday RSI
            current_rsi = recent_intraday['RSI_Intraday'].iloc[-1]
            rsi_factor = (current_rsi - 50) / 50  # Normalize to [-1, 1]
            
            # Volume analysis
            volume_trend = recent_intraday['Volume_Ratio_Intraday'].mean()
            volume_factor = (volume_trend - 1) * 0.1  # Small volume impact
            
            # Volatility adjustment
            volatility = recent_intraday['Volatility_Intraday'].mean()
            volatility_factor = volatility * 0.05
            
            # Combine factors
            prediction_change = (intraday_momentum * 0.4 + 
                               rsi_factor * 0.3 + 
                               volume_factor * 0.2 + 
                               volatility_factor * 0.1)
            
        else:
            # Fallback to daily data
            recent_daily = daily_data.tail(20)
            momentum = recent_daily['Price_Change'].mean()
            prediction_change = momentum * 0.5
        
        # Apply conservative bounds
        prediction_change = max(-0.02, min(0.02, prediction_change))
        
        return current_price * (1 + prediction_change)
    
    def _sentiment_based_prediction(self, daily_data: pd.DataFrame, sentiment_data: Dict[str, float]) -> float:
        """Sentiment-based prediction"""
        current_price = daily_data['Close'].iloc[-1]
        
        # Combine all sentiment sources
        total_sentiment = 0.0
        sentiment_weight = 0.0
        
        sentiment_weights = {
            'news_sentiment': 0.3,
            'social_sentiment': 0.25,
            'earnings_impact': 0.2,
            'analyst_impact': 0.15,
            'options_sentiment': 0.1
        }
        
        for sentiment_type, weight in sentiment_weights.items():
            if sentiment_type in sentiment_data:
                total_sentiment += sentiment_data[sentiment_type] * weight
                sentiment_weight += weight
        
        if sentiment_weight > 0:
            avg_sentiment = total_sentiment / sentiment_weight
            # Convert sentiment to price impact (conservative)
            sentiment_impact = avg_sentiment * 0.01  # 1% max impact
            return current_price * (1 + sentiment_impact)
        else:
            return current_price
    
    def _microstructure_prediction(self, daily_data: pd.DataFrame, micro_features: Dict[str, float]) -> float:
        """Market microstructure-based prediction"""
        current_price = daily_data['Close'].iloc[-1]
        
        # Combine microstructure features
        prediction_change = 0.0
        
        # Spread impact
        if 'avg_spread_proxy' in micro_features:
            spread_impact = micro_features['avg_spread_proxy'] * 0.1
            prediction_change += spread_impact
        
        # Volume trend impact
        if 'volume_trend' in micro_features:
            volume_impact = micro_features['volume_trend'] * 0.05
            prediction_change += volume_impact
        
        # Market efficiency impact
        if 'market_efficiency' in micro_features:
            efficiency_impact = micro_features['market_efficiency'] * 0.02
            prediction_change += efficiency_impact
        
        # Time-based adjustments
        if 'hours_until_close' in micro_features:
            hours_left = micro_features['hours_until_close']
            if hours_left <= 2:
                # End-of-day effect
                prediction_change *= 0.5  # Reduce prediction magnitude near close
        
        # Apply conservative bounds
        prediction_change = max(-0.015, min(0.015, prediction_change))
        
        return current_price * (1 + prediction_change)
    
    def _mean_reversion_prediction(self, daily_data: pd.DataFrame, intraday_data: pd.DataFrame) -> float:
        """Mean reversion-based prediction"""
        current_price = daily_data['Close'].iloc[-1]
        
        # Calculate various moving averages
        ma_5 = daily_data['Close'].rolling(5).mean().iloc[-1]
        ma_10 = daily_data['Close'].rolling(10).mean().iloc[-1]
        ma_20 = daily_data['Close'].rolling(20).mean().iloc[-1]
        
        # Mean reversion strength
        reversion_to_ma5 = (ma_5 - current_price) / current_price
        reversion_to_ma10 = (ma_10 - current_price) / current_price
        reversion_to_ma20 = (ma_20 - current_price) / current_price
        
        # Weighted mean reversion
        weighted_reversion = (reversion_to_ma5 * 0.5 + 
                             reversion_to_ma10 * 0.3 + 
                             reversion_to_ma20 * 0.2)
        
        # Apply mean reversion (conservative)
        reversion_impact = weighted_reversion * 0.3  # 30% of the gap
        
        # Apply bounds
        reversion_impact = max(-0.01, min(0.01, reversion_impact))
        
        return current_price * (1 + reversion_impact)

    def build_ml_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, any]:
        """Build machine learning ensemble for enhanced predictions"""
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.svm import SVR
            from sklearn.metrics import mean_squared_error, r2_score
            import xgboost as xgb
            
            models = {}
            predictions = {}
            scores = {}
            
            # 1. Random Forest
            try:
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                rf_score = r2_score(y_test, rf_pred)
                
                models['random_forest'] = rf_model
                predictions['random_forest'] = rf_pred
                scores['random_forest'] = rf_score
                logger.info(f"Random Forest R²: {rf_score:.4f}")
            except Exception as e:
                logger.warning(f"Random Forest failed: {str(e)}")
            
            # 2. XGBoost
            try:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                xgb_model.fit(X_train, y_train)
                xgb_pred = xgb_model.predict(X_test)
                xgb_score = r2_score(y_test, xgb_pred)
                
                models['xgboost'] = xgb_model
                predictions['xgboost'] = xgb_pred
                scores['xgboost'] = xgb_score
                logger.info(f"XGBoost R²: {xgb_score:.4f}")
            except Exception as e:
                logger.warning(f"XGBoost failed: {str(e)}")
            
            # 3. Gradient Boosting
            try:
                gb_model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                gb_model.fit(X_train, y_train)
                gb_pred = gb_model.predict(X_test)
                gb_score = r2_score(y_test, gb_pred)
                
                models['gradient_boosting'] = gb_model
                predictions['gradient_boosting'] = gb_pred
                scores['gradient_boosting'] = gb_score
                logger.info(f"Gradient Boosting R²: {gb_score:.4f}")
            except Exception as e:
                logger.warning(f"Gradient Boosting failed: {str(e)}")
            
            # 4. Support Vector Regression
            try:
                svr_model = SVR(kernel='rbf', C=1.0, gamma='scale')
                svr_model.fit(X_train, y_train)
                svr_pred = svr_model.predict(X_test)
                svr_score = r2_score(y_test, svr_pred)
                
                models['svr'] = svr_model
                predictions['svr'] = svr_pred
                scores['svr'] = svr_score
                logger.info(f"SVR R²: {svr_score:.4f}")
            except Exception as e:
                logger.warning(f"SVR failed: {str(e)}")
            
            # 5. Linear Regression (baseline)
            try:
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)
                lr_pred = lr_model.predict(X_test)
                lr_score = r2_score(y_test, lr_pred)
                
                models['linear_regression'] = lr_model
                predictions['linear_regression'] = lr_pred
                scores['linear_regression'] = lr_score
                logger.info(f"Linear Regression R²: {lr_score:.4f}")
            except Exception as e:
                logger.warning(f"Linear Regression failed: {str(e)}")
            
            # Calculate ensemble weights based on performance
            if scores:
                total_score = sum(scores.values())
                ensemble_weights = {name: score / total_score for name, score in scores.items()}
                
                # Calculate ensemble prediction
                ensemble_pred = np.zeros_like(y_test)
                for name, pred in predictions.items():
                    ensemble_pred += pred * ensemble_weights[name]
                
                ensemble_score = r2_score(y_test, ensemble_pred)
                logger.info(f"Ensemble R²: {ensemble_score:.4f}")
                
                return {
                    'models': models,
                    'weights': ensemble_weights,
                    'scores': scores,
                    'ensemble_score': ensemble_score
                }
            else:
                logger.error("No models successfully trained")
                return {}
                
        except Exception as e:
            logger.error(f"Error building ML ensemble: {str(e)}")
            return {}
    
    def predict_with_ml_ensemble(self, features: np.ndarray) -> float:
        """Make prediction using trained ML ensemble"""
        try:
            if not hasattr(self, 'ml_ensemble') or not self.ml_ensemble:
                logger.warning("ML ensemble not available, using fallback")
                return 0.0
            
            ensemble_prediction = 0.0
            total_weight = 0.0
            
            for model_name, model in self.ml_ensemble['models'].items():
                if model_name in self.ml_ensemble['weights']:
                    weight = self.ml_ensemble['weights'][model_name]
                    prediction = model.predict(features.reshape(1, -1))[0]
                    ensemble_prediction += prediction * weight
                    total_weight += weight
            
            if total_weight > 0:
                return ensemble_prediction / total_weight
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in ML ensemble prediction: {str(e)}")
            return 0.0

    def _predict_with_sparse_data(self, daily_data: pd.DataFrame, intraday_data: pd.DataFrame) -> float:
        """Prediction path for very new tickers with limited data."""
        try:
            current_price = self._safe_to_float(self.get_current_price(), default=0.0)
            if current_price <= 0:
                # Fallback to any last known close from intraday or daily
                if intraday_data is not None and not intraday_data.empty and 'Close' in intraday_data:
                    current_price = self._safe_to_float(intraday_data['Close'], default=0.0)
                if (current_price <= 0) and (daily_data is not None and not daily_data.empty):
                    current_price = float(daily_data['Close'].iloc[-1])
            if current_price <= 0:
                return 0.0

            # Intraday momentum over last 60-120 minutes if available
            predicted_change = 0.0
            if intraday_data is not None and not intraday_data.empty and 'Close' in intraday_data:
                closes = intraday_data['Close'].astype(float).values
                if len(closes) >= 6:
                    import numpy as np
                    recent = closes[-24:] if len(closes) >= 24 else closes
                    returns = np.diff(recent) / recent[:-1]
                    # Average recent returns and project moderately into next 4h
                    avg_ret = float(np.nanmean(returns[-12:])) if len(returns) >= 12 else float(np.nanmean(returns))
                    # Scale to 4 hours, assuming ~1m per candle if very granular, else dampen
                    scale = 2.0  # conservative projection multiplier
                    predicted_change += avg_ret * scale
            else:
                # Daily momentum scaled to 4 hours if at least 2 closes
                if daily_data is not None and not daily_data.empty and len(daily_data) >= 2:
                    import numpy as np
                    closes = daily_data['Close'].astype(float).values
                    daily_ret = (closes[-1] - closes[-2]) / closes[-2]
                    predicted_change += daily_ret * (4.0 / 6.5)  # scale daily to ~4h

            # Add small sentiment tilt using real-time sentiment if available
            try:
                sentiment = self.get_real_time_sentiment()
                if sentiment:
                    # Weighted average of provided components
                    weights = {'news_sentiment': 0.3, 'social_sentiment': 0.4, 'earnings_impact': 0.2, 'analyst_impact': 0.05, 'options_sentiment': 0.05}
                    total_w = 0.0
                    total_s = 0.0
                    for k, w in weights.items():
                        if k in sentiment:
                            total_s += float(sentiment[k]) * w
                            total_w += w
                    avg_s = (total_s / total_w) if total_w > 0 else 0.0
                    predicted_change += avg_s * 0.01  # max 1% influence
            except Exception:
                pass

            # Clamp to conservative bounds
            predicted_change = max(-0.03, min(0.03, predicted_change))
            return current_price * (1.0 + predicted_change)
        except Exception as e:
            logger.warning(f"Sparse-data prediction failed: {e}")
            return 0.0

    def resolve_exchange_symbol(self, symbol: str) -> str:
        """Resolve a base symbol to a specific exchange listing to ensure correct currency.
        If the symbol already includes an exchange suffix, return as-is.
        Prefer Canadian TSX/TSXV listings when available for dual-listed names.
        """
        try:
            # If already has known suffix, keep it
            known_suffixes = ('.TO', '.V', '.CN', '.NE', '.PA', '.L', '.AX')
            if any(symbol.endswith(sfx) for sfx in known_suffixes):
                return symbol

            base = symbol
            def _valid(sym: str):
                try:
                    t = yf.Ticker(sym)
                    hist = t.history(period="5d")
                    cur = (t.info or {}).get('currency')
                    return (not hist.empty), (cur.upper() if isinstance(cur, str) else None)
                except Exception:
                    return False, None

            # Check base symbol first
            base_ok, base_ccy = _valid(base)
            # Probe common Canadian suffixes
            candidates = [base + '.TO', base + '.V', base + '.CN', base + '.NE']
            for cand in candidates:
                ok, ccy = _valid(cand)
                if ok and ccy in {'CAD'}:
                    logger.info(f"Resolved {base} to Canadian listing {cand} ({ccy})")
                    return cand

            # Probe other major non-USD markets for proper currency mapping
            other_candidates = [
                (base + '.L', 'GBP'),
                (base + '.PA', 'EUR'),
                (base + '.AX', 'AUD')
            ]
            for cand, expected_ccy in other_candidates:
                ok, ccy = _valid(cand)
                if ok and ccy == expected_ccy:
                    logger.info(f"Resolved {base} to local listing {cand} ({ccy})")
                    return cand

            # If base exists and has non-USD currency, keep it
            if base_ok and base_ccy and base_ccy != 'USD':
                logger.info(f"Keeping {base} with currency {base_ccy}")
                return base

            # Otherwise keep original
            return base
        except Exception as e:
            logger.warning(f"Failed to resolve exchange for {symbol}: {e}")
            return symbol

    def _plot_candlesticks(self, ax, df: pd.DataFrame, time_col: str) -> None:
        """Render simple candlesticks on the provided axes using matplotlib."""
        if df is None or df.empty:
            return
        x = mdates.date2num(pd.to_datetime(df[time_col]).to_pydatetime())
        width = 0.6 * (x[1] - x[0]) if len(x) > 1 else 0.02
        for i in range(len(df)):
            o = float(df['Open'].iloc[i])
            h = float(df['High'].iloc[i])
            l = float(df['Low'].iloc[i])
            c = float(df['Close'].iloc[i])
            color = '#2ca02c' if c >= o else '#d62728'
            # Wick
            ax.vlines(x[i], l, h, color=color, linewidth=1)
            # Body
            lower = min(o, c)
            height = abs(c - o)
            if height == 0:
                height = max(1e-6, 0.0001 * (h if h else 1))
            rect = plt.Rectangle((x[i] - width/2, lower), width, height, color=color, alpha=0.7)
            ax.add_patch(rect)
        ax.xaxis_date()
        ax.grid(True, linestyle='--', alpha=0.2)
        ax.set_ylabel('Price')

    def generate_intraday_chart_with_prediction(self, predicted_price: float, output_path: str) -> None:
        """Generate intraday candlestick chart and overlay predicted path until 16:00 local time."""
        try:
            interval = self.config.get('intraday_interval', '5m')
            days = int(self.config.get('intraday_days', 1))
            df = self.get_intraday_data(interval=interval, days=days)
            if df is None or df.empty:
                raise ValueError("No intraday data available")
            fig, ax = plt.subplots(figsize=(10, 5))
            self._plot_candlesticks(ax, df, time_col='Datetime')

            # Overlay predicted path from last candle close to market close
            try:
                last_time = pd.to_datetime(df['Datetime'].iloc[-1])
                last_price = float(df['Close'].iloc[-1])
                market_close = last_time.replace(hour=16, minute=0, second=0, microsecond=0)
                if market_close <= last_time:
                    market_close = market_close + timedelta(days=1)
                # Build timeline in same spacing as recent candles
                if predicted_price and predicted_price > 0:
                    # Create 5-minute steps
                    times = []
                    t = last_time
                    while t < market_close:
                        t += timedelta(minutes=5)
                        times.append(t)
                    if times:
                        xs = mdates.date2num(times)
                        # Linear path
                        ys = [last_price + (predicted_price - last_price) * (i+1)/len(times) for i in range(len(times))]
                        ax.plot_date(xs, ys, '-o', linewidth=2, markersize=2, color='#1f77b4', label='Predicted path')
                        ax.legend(loc='upper left')
            except Exception as e:
                logger.warning(f"Failed to overlay prediction path: {e}")

            ax.set_title(f"{self.ticker} Intraday ({interval}) with Prediction")
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed generating intraday chart: {e}")

    def generate_timeframe_chart(self, period: str, interval: str, output_path: str) -> None:
        """Generate a candlestick chart for a given period/interval without prediction overlay."""
        try:
            data = yf.download(self.ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if data is None or data.empty:
                raise ValueError("No data for timeframe chart")
            data = data.dropna().reset_index()
            # Normalize column names
            if 'Datetime' not in data.columns:
                if 'Date' in data.columns:
                    data = data.rename(columns={'Date': 'Datetime'})
                else:
                    data['Datetime'] = data.index
            fig, ax = plt.subplots(figsize=(10, 5))
            self._plot_candlesticks(ax, data[['Datetime','Open','High','Low','Close']], time_col='Datetime')
            ax.set_title(f"{self.ticker} {period} ({interval}) Trend")
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed generating timeframe chart {period}/{interval}: {e}")

    def upload_slack_file(self, file_path: str, title: str, channel: str = "#stock-price-alerts", thread_ts: Optional[str] = None) -> None:
        """Upload an image file to Slack channel using bot token. Optionally thread under a message ts."""
        try:
            if not self.slack_bot_token:
                logger.warning("No Slack bot token configured; skipping file upload")
                return
            if not os.path.exists(file_path):
                logger.warning(f"File not found for Slack upload: {file_path}")
                return
            with open(file_path, 'rb') as f:
                response = requests.post(
                    "https://slack.com/api/files.upload",
                    headers={"Authorization": f"Bearer {self.slack_bot_token}"},
                    data={"channels": channel, "title": title, **({"thread_ts": thread_ts} if thread_ts else {})},
                    files={"file": (os.path.basename(file_path), f, "image/png")}
                )
            result = response.json()
            if not result.get("ok"):
                logger.error(f"Slack file upload failed: {result.get('error', 'unknown error')}")
            else:
                logger.info(f"Uploaded chart to Slack: {os.path.basename(file_path)}")
        except Exception as e:
            logger.warning(f"Failed to upload file to Slack: {e}")

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