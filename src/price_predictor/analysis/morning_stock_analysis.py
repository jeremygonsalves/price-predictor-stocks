#!/usr/bin/env python3
"""
Morning Stock Analysis Script
Analyzes news feeds from Reddit, Alpha Vantage, and Yahoo Finance to identify top stocks to watch
Runs at 8:30 AM every weekday, analyzing news from 4 PM previous day to 8 AM current day
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from textblob import TextBlob
from datetime import datetime, timedelta, time
import schedule
import logging
import os
import json
from typing import Dict, List, Tuple, Optional
import praw
import re
from dotenv import load_dotenv
import time as time_module
from collections import defaultdict, Counter
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class MorningStockAnalyzer:
    def __init__(self, config_file: str = "configs/config.json"):
        """
        Initialize the morning stock analyzer
        
        Args:
            config_file: Path to configuration JSON file
        """
        # Setup logging first
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Load environment variables
        load_dotenv(os.path.expanduser("~/.env"))
        
        # Initialize API keys and credentials using the same logic as enhanced_stock_alert.py
        self.alpha_vantage_api_key = self.load_alpha_vantage_key()
        self.reddit_client = self.init_reddit_client()
        self.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        self.slack_bot_token = os.getenv('SLACK_BOT_TOKEN')
        
        # Analysis parameters
        self.analysis_start_time = time(16, 0)  # 4 PM previous day
        self.analysis_end_time = time(8, 0)     # 8 AM current day
        self.max_stocks_to_return = 10
        self.min_news_count = 3
        self.min_sentiment_score = 0.1
        
        # Stock categories for analysis
        self.stock_categories = {
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC'],
            'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF'],
            'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN'],
            'energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL'],
            'consumer': ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'DIS', 'NKE', 'SBUX', 'TGT']
        }
        
        # Popular stock subreddits
        self.stock_subreddits = [
            'wallstreetbets', 'stocks', 'investing', 'StockMarket', 
            'RobinHoodPennyStocks', 'pennystocks', 'options', 'Daytrading'
        ]
        
        logger.info("Morning Stock Analyzer initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('morning_stock_analysis.log'),
                logging.StreamHandler()
            ]
        )
        global logger
        logger = logging.getLogger(__name__)
        # Quiet third-party noisy loggers
        logging.getLogger("yfinance").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        logging.getLogger("requests").setLevel(logging.ERROR)
    
    def load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file {config_file} not found")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {str(e)}")
            raise
    
    def load_alpha_vantage_key(self) -> Optional[str]:
        """Load Alpha Vantage API key"""
        try:
            # Check for Alpha Vantage key in secrets directory
            if os.path.exists('secrets/alphavantage.txt'):
                with open('secrets/alphavantage.txt', 'r') as f:
                    api_key = f.read().strip()
                logger.info("Alpha Vantage API key loaded from secrets/alphavantage.txt")
                return api_key
            
            # Fallback to environment variable
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if api_key:
                logger.info("Alpha Vantage API key loaded from environment variable")
                return api_key
            else:
                logger.warning("Alpha Vantage API key not found")
                return None
        except Exception as e:
            logger.warning(f"Failed to load Alpha Vantage API key: {str(e)}")
            return None
    
    def init_reddit_client(self) -> Optional[dict]:
        """Initialize Reddit client - requires user's own credentials"""
        try:
            # Check for Reddit credentials in secrets directory
            required_files = ['secrets/pw.txt', 'secrets/client_id.txt', 'secrets/client_secret.txt', 'secrets/username.txt']
            missing_files = []
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
            
            if missing_files:
                logger.warning(f"Reddit credentials not found. Missing files: {', '.join(missing_files)}")
                logger.info("To enable Reddit sentiment analysis, please create the following files in the 'secrets/' directory:")
                logger.info("  - secrets/username.txt (your Reddit username)")
                logger.info("  - secrets/pw.txt (your Reddit password)")
                logger.info("  - secrets/client_id.txt (your Reddit app client ID)")
                logger.info("  - secrets/client_secret.txt (your Reddit app client secret)")
                logger.info("See README.md for setup instructions.")
                return None
            
            # Read all required credentials
            with open('secrets/username.txt', 'r') as f:
                username = f.read().strip()
            
            with open('secrets/pw.txt', 'r') as f:
                pw = f.read().strip()
            
            with open('secrets/client_id.txt', 'r') as f:
                client_id = f.read().strip()
            
            with open('secrets/client_secret.txt', 'r') as f:
                client_secret = f.read().strip()
            
            # Validate that credentials are not empty
            if not all([username, pw, client_id, client_secret]):
                logger.error("Reddit credentials found but some are empty. Please check your secrets files.")
                return None
            
            # Use the authentication logic with user's own credentials
            auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
            data = {
                "grant_type": "password",
                "username": username,
                "password": pw
            }
            headers = {"User-Agent": "StockPredictor/1.0"}
            
            res = requests.post("https://www.reddit.com/api/v1/access_token", 
                              auth=auth, data=data, headers=headers)
            
            if res.status_code == 200:
                token = res.json()['access_token']
                headers = {**headers, **{'Authorization': f"bearer {token}"}}
                
                logger.info(f"Reddit API client initialized successfully for user: {username}")
                return {
                    'headers': headers,
                    'token': token,
                    'auth_method': 'oauth',
                    'username': username
                }
            else:
                logger.error(f"Failed to authenticate with Reddit API: {res.status_code}")
                logger.error("Please check your Reddit credentials in the secrets/ directory.")
                return None
                
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {str(e)}")
            logger.info("Reddit sentiment analysis will be disabled. Check your credentials and try again.")
            return None
    
    def get_analysis_time_range(self) -> Tuple[datetime, datetime]:
        """Get the time range for analysis (4 PM previous day to 8 AM current day)"""
        now = datetime.now()
        
        # If current time is before 8 AM, use previous day as end date
        if now.time() < self.analysis_end_time:
            end_date = now.replace(hour=8, minute=0, second=0, microsecond=0)
            start_date = end_date - timedelta(days=1)
            start_date = start_date.replace(hour=16, minute=0, second=0, microsecond=0)
        else:
            # Use current day 8 AM as end date
            end_date = now.replace(hour=8, minute=0, second=0, microsecond=0)
            start_date = end_date - timedelta(days=1)
            start_date = start_date.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return start_date, end_date
    
    def _fetch_json_with_retry(self, url: str, headers: Dict[str, str], params: Dict, max_retries: int = 3, backoff_seconds: float = 0.8) -> Optional[Dict]:
        """HTTP GET JSON helper with basic retry/backoff and logging."""
        last_status = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=12)
                last_status = resp.status_code
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (429, 500, 502, 503, 504):
                    sleep_time = backoff_seconds * (2 ** attempt)
                    logger.warning(f"HTTP {resp.status_code} from {url}. Backing off {sleep_time:.1f}s (attempt {attempt+1}/{max_retries})")
                    time_module.sleep(sleep_time)
                    continue
                # Non-retryable errors
                logger.warning(f"HTTP {resp.status_code} from {url}. Not retrying.")
                return None
            except Exception as e:
                sleep_time = backoff_seconds * (2 ** attempt)
                logger.warning(f"Exception fetching {url}: {e}. Backing off {sleep_time:.1f}s (attempt {attempt+1}/{max_retries})")
                time_module.sleep(sleep_time)
        logger.error(f"Failed to fetch {url} after {max_retries} attempts (last status={last_status})")
        return None
    
    def get_reddit_stock_mentions(self, start_time: datetime, end_time: datetime) -> Dict[str, List[Dict]]:
        """Get stock mentions from Reddit subreddits using the enhanced script logic"""
        stock_mentions = defaultdict(list)
        
        try:
            use_oauth = bool(self.reddit_client)
            oauth_headers = self.reddit_client['headers'] if use_oauth else None
            public_headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "DNT": "1",
                "Referer": "https://www.reddit.com/"
            }
            
            for subreddit_name in self.stock_subreddits:
                try:
                    kept_count = 0
                    if use_oauth:
                        # OAuth-protected endpoints
                        data = self._fetch_json_with_retry(
                            f"https://oauth.reddit.com/r/{subreddit_name}/new",
                            headers=oauth_headers,
                            params={"limit": 100}
                        )
                        posts = data.get('data', {}).get('children', []) if data else []
                    else:
                        # Public JSON endpoint
                        data = self._fetch_json_with_retry(
                            f"https://api.reddit.com/r/{subreddit_name}/new",
                            headers=public_headers,
                            params={"limit": 100, "raw_json": 1}
                        )
                        if not data:
                            data = self._fetch_json_with_retry(
                                f"https://www.reddit.com/r/{subreddit_name}/new.json",
                                headers=public_headers,
                                params={"limit": 100, "raw_json": 1}
                            )
                        posts = data.get('data', {}).get('children', []) if data else []
                    
                    for post in posts:
                        post_data = post.get('data', {})
                        if not post_data:
                            continue
                        created_utc = post_data.get('created_utc')
                        if created_utc is None:
                            continue
                        post_time = datetime.fromtimestamp(created_utc)
                        
                        if start_time <= post_time <= end_time:
                            title = post_data.get('title', '')
                            selftext = post_data.get('selftext', '')
                            text = f"{title} {selftext}"
                            tickers = self.extract_stock_tickers(text, validate_online=True)
                            
                            for ticker in tickers:
                                stock_mentions[ticker].append({
                                    'source': 'reddit',
                                    'subreddit': subreddit_name,
                                    'title': title,
                                    'score': post_data.get('score', 0),
                                    'comments': post_data.get('num_comments', 0),
                                    'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                    'timestamp': post_time,
                                    'sentiment': self.analyze_sentiment(text)
                                })
                            kept_count += len(tickers)
 
                    # Fetch comments for hot posts for more mentions
                    if use_oauth:
                        hot_data = self._fetch_json_with_retry(
                            f"https://oauth.reddit.com/r/{subreddit_name}/hot",
                            headers=oauth_headers,
                            params={"limit": 20}
                        )
                        hot_posts = hot_data.get('data', {}).get('children', []) if hot_data else []
                    else:
                        hot_data = self._fetch_json_with_retry(
                            f"https://api.reddit.com/r/{subreddit_name}/hot",
                            headers=public_headers,
                            params={"limit": 20, "raw_json": 1}
                        )
                        if not hot_data:
                            hot_data = self._fetch_json_with_retry(
                                f"https://www.reddit.com/r/{subreddit_name}/hot.json",
                                headers=public_headers,
                                params={"limit": 20, "raw_json": 1}
                            )
                        hot_posts = hot_data.get('data', {}).get('children', []) if hot_data else []
 
                    for post in hot_posts:
                        post_data = post.get('data', {})
                        post_id = post_data.get('id', '')
                        if not post_id:
                            continue
                        
                        if use_oauth:
                            comments_data = self._fetch_json_with_retry(
                                f"https://oauth.reddit.com/comments/{post_id}",
                                headers=oauth_headers,
                                params={"limit": 50}
                            )
                            comments_list = comments_data[1]['data']['children'] if isinstance(comments_data, list) and len(comments_data) > 1 else []
                        else:
                            comments_data = self._fetch_json_with_retry(
                                f"https://api.reddit.com/comments/{post_id}",
                                headers=public_headers,
                                params={"limit": 50, "raw_json": 1}
                            )
                            if not comments_data:
                                comments_data = self._fetch_json_with_retry(
                                    f"https://www.reddit.com/comments/{post_id}.json",
                                    headers=public_headers,
                                    params={"limit": 50, "raw_json": 1}
                                )
                            comments_list = comments_data[1]['data']['children'] if isinstance(comments_data, list) and len(comments_data) > 1 else []
                        
                        for comment in comments_list:
                            if comment.get('kind') != 't1':
                                continue
                            comment_data = comment.get('data', {})
                            created_utc = comment_data.get('created_utc')
                            if created_utc is None:
                                continue
                            comment_time = datetime.fromtimestamp(created_utc)
                            if not (start_time <= comment_time <= end_time):
                                continue
                            comment_body = comment_data.get('body', '')
                            tickers = self.extract_stock_tickers(comment_body, validate_online=True)
                            for ticker in tickers:
                                stock_mentions[ticker].append({
                                    'source': 'reddit',
                                    'subreddit': subreddit_name,
                                    'title': f"Comment on: {post_data.get('title', '')}",
                                    'score': comment_data.get('score', 0),
                                    'comments': 0,
                                    'url': f"https://reddit.com{comment_data.get('permalink', '')}",
                                    'timestamp': comment_time,
                                    'sentiment': self.analyze_sentiment(comment_body)
                                })
                            kept_count += len(tickers)

                    # Also fetch comments for newest posts (captures ticker mentions in early discussions)
                    for post in posts[:10]:
                        post_data = post.get('data', {})
                        post_id = post_data.get('id', '')
                        if not post_id:
                            continue
                        if use_oauth:
                            comments_data = self._fetch_json_with_retry(
                                f"https://oauth.reddit.com/comments/{post_id}",
                                headers=oauth_headers,
                                params={"limit": 50}
                            )
                        else:
                            comments_data = self._fetch_json_with_retry(
                                f"https://api.reddit.com/comments/{post_id}",
                                headers=public_headers,
                                params={"limit": 50, "raw_json": 1}
                            )
                            if not comments_data:
                                comments_data = self._fetch_json_with_retry(
                                    f"https://www.reddit.com/comments/{post_id}.json",
                                    headers=public_headers,
                                    params={"limit": 50, "raw_json": 1}
                                )
                        comments_list = comments_data[1]['data']['children'] if isinstance(comments_data, list) and len(comments_data) > 1 else []
                        for comment in comments_list:
                            if comment.get('kind') != 't1':
                                continue
                            comment_data = comment.get('data', {})
                            created_utc = comment_data.get('created_utc')
                            if created_utc is None:
                                continue
                            comment_time = datetime.fromtimestamp(created_utc)
                            if not (start_time <= comment_time <= end_time):
                                continue
                            body = comment_data.get('body', '')
                            tickers = self.extract_stock_tickers(body, validate_online=True)
                            for ticker in tickers:
                                stock_mentions[ticker].append({
                                    'source': 'reddit',
                                    'subreddit': subreddit_name,
                                    'title': f"Comment on: {post_data.get('title', '')}",
                                    'score': comment_data.get('score', 0),
                                    'comments': 0,
                                    'url': f"https://reddit.com{comment_data.get('permalink', '')}",
                                    'timestamp': comment_time,
                                    'sentiment': self.analyze_sentiment(body)
                                })
                            kept_count += len(tickers)

                    logger.info(f"Reddit r/{subreddit_name}: kept {kept_count} validated tickers in window")
                     
                    # Be gentle with rate limits
                    time_module.sleep(0.3)
                
                except Exception as e:
                    logger.warning(f"Error processing subreddit {subreddit_name}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in Reddit analysis: {str(e)}")
        
        return dict(stock_mentions)
    
    def get_alpha_vantage_news(self, start_time: datetime, end_time: datetime) -> Dict[str, List[Dict]]:
        """Get news from Alpha Vantage API"""
        if not self.alpha_vantage_api_key:
            logger.warning("Alpha Vantage API key not found")
            return {}
        
        stock_news = defaultdict(list)
        
        try:
            # Get news for major stock categories
            for category, tickers in self.stock_categories.items():
                for ticker in tickers[:5]:  # Limit to top 5 per category to avoid rate limits
                    try:
                        url = f"https://www.alphavantage.co/query"
                        params = {
                            'function': 'NEWS_SENTIMENT',
                            'tickers': ticker,
                            'apikey': self.alpha_vantage_api_key,
                            'limit': 50
                        }
                        
                        response = requests.get(url, params=params, timeout=10)
                        response.raise_for_status()
                        data = response.json()
                        
                        if 'feed' in data:
                            for article in data['feed']:
                                article_time = datetime.strptime(article['time_published'], '%Y%m%dT%H%M%S')
                                
                                if start_time <= article_time <= end_time:
                                    stock_news[ticker].append({
                                        'source': 'alphavantage',
                                        'title': article['title'],
                                        'summary': article['summary'],
                                        'url': article['url'],
                                        'timestamp': article_time,
                                        'sentiment': float(article.get('overall_sentiment_score', 0)),
                                        'sentiment_label': article.get('overall_sentiment_label', 'neutral'),
                                        'relevance_score': float(article.get('relevance_score', 0))
                                    })
                        
                        # Rate limiting
                        time_module.sleep(0.2)
                    
                    except Exception as e:
                        logger.warning(f"Error getting Alpha Vantage news for {ticker}: {str(e)}")
                        continue
        
        except Exception as e:
            logger.error(f"Error in Alpha Vantage analysis: {str(e)}")
        
        return dict(stock_news)
    
    def get_yahoo_finance_news(self, start_time: datetime, end_time: datetime) -> Dict[str, List[Dict]]:
        """Get news from Yahoo Finance with throttling and fallback if time-filter yields nothing"""
        stock_news = defaultdict(list)
        
        try:
            # Get news for major stock categories
            for category, tickers in self.stock_categories.items():
                for ticker in tickers[:3]:  # Throttle: fewer tickers per category
                    try:
                        stock = yf.Ticker(ticker)
                        news = stock.news
                        
                        # Time-filtered collection
                        for article in news:
                            try:
                                article_time = datetime.fromtimestamp(article['providerPublishTime'])
                            except Exception:
                                continue
                            
                            if start_time <= article_time <= end_time:
                                stock_news[ticker].append({
                                    'source': 'yahoofinance',
                                    'title': article.get('title', ''),
                                    'summary': article.get('summary', ''),
                                    'url': article.get('link', ''),
                                    'timestamp': article_time,
                                    'sentiment': self.analyze_sentiment((article.get('title') or '') + ' ' + (article.get('summary') or '')),
                                    'publisher': article.get('publisher', 'Unknown')
                                })
                        
                        # Gentle sleep to avoid rate limits
                        time_module.sleep(0.35)
                    except Exception as e:
                        logger.warning(f"Error getting Yahoo Finance news for {ticker}: {str(e)}")
                        continue
        
        except Exception as e:
            logger.error(f"Error in Yahoo Finance analysis: {str(e)}")
        
        # Fallback: if nothing collected in window, include most recent headlines without time filter (top 2 per ticker)
        if not stock_news:
            try:
                for category, tickers in self.stock_categories.items():
                    for ticker in tickers[:2]:
                        try:
                            stock = yf.Ticker(ticker)
                            news = stock.news
                            for article in news[:2]:
                                try:
                                    article_time = datetime.fromtimestamp(article['providerPublishTime'])
                                except Exception:
                                    article_time = end_time
                                stock_news[ticker].append({
                                    'source': 'yahoofinance',
                                    'title': article.get('title', ''),
                                    'summary': article.get('summary', ''),
                                    'url': article.get('link', ''),
                                    'timestamp': article_time,
                                    'sentiment': self.analyze_sentiment((article.get('title') or '') + ' ' + (article.get('summary') or '')),
                                    'publisher': article.get('publisher', 'Unknown')
                                })
                            time_module.sleep(0.35)
                        except Exception as e:
                            logger.warning(f"Fallback Yahoo Finance fetch failed for {ticker}: {str(e)}")
                            continue
            except Exception as e:
                logger.error(f"Fallback Yahoo Finance analysis failed: {str(e)}")
        
        return dict(stock_news)
    
    def extract_stock_tickers(self, text: str, validate_online: bool = False) -> List[str]:
        """Extract stock tickers from text"""
        # Build and cache helpers on first use
        if not hasattr(self, '_ticker_validation_cache'):
            self._ticker_validation_cache = {}
        if not hasattr(self, '_ticker_normalization_cache'):
            self._ticker_normalization_cache = {}

        if not text:
            return []

        original_text = text
        text = text.strip()

        # Patterns to capture a wide range of ticker styles
        patterns = [
            # Cashtags like $TSLA, $SHOP.TO
            r'\$[A-Za-z][A-Za-z0-9\.-]{0,9}',
            # Exchange prefixes like NASDAQ:MSFT, TSX:SHOP, NYSE-BABA
            r'\b(?:NASDAQ|NYSE|AMEX|TSX|TSXV|TSX-V|CSE|OTC|LSE|ASX)[:\-\s]([A-Za-z][A-Za-z0-9\.-]{0,9})\b',
            # Tickers with class or exchange suffix (.B, -B, .TO, .V, .CN)
            r'\b[A-Z]{1,5}(?:[\.-](?:[A-Z]{1,3}|TO|V|CN|NE|PA|L|AX))\b'
        ]

        candidates = set()
        for pat in patterns:
            for match in re.findall(pat, text, flags=re.IGNORECASE):
                # Some patterns use a capturing group; normalize match accordingly
                raw = match if isinstance(match, str) else match[0]
                if not raw:
                    continue
                candidates.add(raw)

        # Also handle colon-joined tokens discovered by split (defensive)
        for token in re.split(r'\s+', text):
            if ':' in token:
                parts = re.split(r'[:\-]', token)
                for part in parts:
                    if part:
                        candidates.add(part)

        # Small whitelist of common plain-ticker mentions to catch when people omit '$'
        common_plain_tickers = {
            'AAPL','MSFT','AMZN','TSLA','NVDA','META','GOOG','GOOGL','AMD','NFLX','INTC','BRK-B','BRK-A',
            'JPM','BAC','WFC','GS','MS','C','XOM','CVX','V','MA','NKE','DIS','ADBE','CRM','PYPL','T','VZ','PFE'
        }
        for word in re.findall(r'\b[A-Z]{2,5}\b', text):
            if word in common_plain_tickers:
                candidates.add(word)

        # Common stop words and noise tokens frequently hit by regex
        common_noise = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
            'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD',
            'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE',
            'USA', 'IPO', 'ETF', 'EPS', 'PE', 'EV', 'CEO', 'CFO', 'SEC', 'GDP', 'CPI', 'PPI', 'FED',
            'OTC', 'TSX', 'TSXV', 'NASDAQ', 'NYSE', 'AMEX', 'LSE', 'ASX', 'IMO', 'FYI',
            # Short common words and prepositions/pronouns
            'OF', 'TO', 'IN', 'ON', 'BY', 'IT', 'IS', 'AS', 'AT', 'BE', 'DO', 'GO', 'UP', 'SO', 'MY', 'WE', 'US', 'AM',
            'ME', 'I', 'HE', 'SHE', 'THEY', 'THEM', 'THEIR', 'YOUR', 'OUR', 'ITS',
            # Verbs/common nouns often hit
            'HAVE', 'HAS', 'HAD', 'WILL', 'WOULD', 'SHOULD', 'COULD', 'MAY', 'MIGHT', 'MUST',
            'FROM', 'WITH', 'WITHOUT', 'ABOUT', 'AFTER', 'BEFORE', 'OVER', 'UNDER', 'BETWEEN', 'INTO', 'ONTO', 'OFF', 'OUT', 'BACK',
            'MORE', 'MOST', 'LESS', 'LEAST', 'GOOD', 'BEST', 'BETTER', 'WORST',
            'YEAR', 'YEARS', 'MONTH', 'MONTHS', 'WEEK', 'WEEKS', 'DAYS', 'TODAY', 'YESTERDAY', 'TOMORROW',
            'LIKE', 'RIGHT', 'LEFT', 'TAKE', 'TAKES', 'TAKEN', 'GET', 'GOT', 'GOING', 'HERE', 'THERE',
            'THIS', 'THAT', 'THESE', 'THOSE', 'BUY', 'SELL', 'HOLD', 'PRICE', 'SALES', 'REVENUE', 'EARNINGS', 'REPORT', 'DATE', 'FULL',
            # Months abbreviations
            'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'SEPT', 'OCT', 'NOV', 'DEC',
            # Common countries/orgs that may appear capitalized
            'USA', 'US', 'UK', 'EU', 'JAPAN', 'CHINA', 'SEC'
        }

        def normalize_symbol(raw_symbol: str) -> str:
            s = raw_symbol.strip()
            if s.startswith('$'):
                s = s[1:]
            s = s.upper()
            # Remove surrounding punctuation
            s = s.strip(".,;!?)\"':(")
            # If it looks like EXCHANGE:TICKER, strip leading exchange that slipped through
            s = re.sub(r'^(?:NASDAQ|NYSE|AMEX|TSX|TSXV|TSX\-V|CSE|OTC|LSE|ASX)[:\-\s]+', '', s)
            # Convert class tickers like BRK.B to Yahoo style BRK-B (but keep Canadian suffixes with dot)
            if '.' in s:
                parts = s.split('.')
                if len(parts) == 2 and parts[1] not in {'TO', 'V', 'CN', 'NE', 'PA', 'L', 'AX'}:
                    s = parts[0] + '-' + parts[1]
            # Extra safety: strip any lingering leading '$'
            if s.startswith('$'):
                s = s[1:]
            return s

        def looks_like_ticker(sym: str) -> bool:
            # Allow 1-5 alphanumerics with optional class or exchange suffix
            if sym in common_noise:
                return False
            # Base symbol portion
            m = re.match(r'^[A-Z]{1,5}(?:-[A-Z]{1,3})?(?:\.(?:TO|V|CN|NE|PA|L|AX))?$', sym)
            if not m:
                return False
            # Heuristic: drop most 1-letter tokens except a small allowlist
            if len(sym) == 1 and sym not in {'F', 'T', 'C', 'A', 'B', 'K', 'M'}:
                return False
            return m is not None

        def validate_via_yfinance(sym: str) -> bool:
            if sym in self._ticker_validation_cache:
                return self._ticker_validation_cache[sym]
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="5d")
                is_valid = not hist.empty
            except Exception:
                is_valid = False
            self._ticker_validation_cache[sym] = is_valid
            return is_valid
        
        # Expose validator for reuse
        self._validate_symbol_online = validate_via_yfinance

        normalized = set()
        for cand in candidates:
            sym = normalize_symbol(cand)
            if not sym:
                continue
            if not looks_like_ticker(sym):
                continue
            normalized.add(sym)

        if validate_online:
            validated = []
            for sym in normalized:
                if validate_via_yfinance(sym):
                    validated.append(sym)
            return validated
        else:
            return list(normalized)
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def get_stock_price_data(self, ticker: str) -> Dict:
        """Get current stock price and basic data"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get recent price data
            hist = stock.history(period="5d")
            if hist.empty:
                return {}
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100 if prev_close > 0 else 0
            
            return {
                'current_price': current_price,
                'prev_close': prev_close,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'volume': hist['Volume'].iloc[-1],
                'market_cap': info.get('marketCap', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
        except Exception as e:
            logger.warning(f"Error getting price data for {ticker}: {str(e)}")
            return {}
    
    def calculate_stock_score(self, ticker: str, news_data: List[Dict], price_data: Dict) -> Dict:
        """Calculate a comprehensive score for a stock based on news and price data"""
        if not news_data:
            return {'score': 0, 'reason': 'No news data available'}
        
        # Calculate news metrics
        total_news = len(news_data)
        avg_sentiment = np.mean([item.get('sentiment', 0) for item in news_data])
        
        # Calculate engagement metrics (for Reddit)
        reddit_engagement = 0
        reddit_news = [item for item in news_data if item['source'] == 'reddit']
        if reddit_news:
            total_score = sum(item.get('score', 0) for item in reddit_news)
            total_comments = sum(item.get('comments', 0) for item in reddit_news)
            reddit_engagement = (total_score + total_comments * 2) / len(reddit_news)
        
        # Calculate relevance (for Alpha Vantage)
        avg_relevance = 0
        alphavantage_news = [item for item in news_data if item['source'] == 'alphavantage']
        if alphavantage_news:
            avg_relevance = np.mean([item.get('relevance_score', 0) for item in alphavantage_news])
        
        # Calculate source diversity
        sources = set(item['source'] for item in news_data)
        source_diversity = len(sources) / 3  # Normalize to 0-1
        
        # Calculate price momentum
        price_momentum = 0
        if price_data:
            price_momentum = price_data.get('price_change_pct', 0) / 10  # Normalize to reasonable range
        
        # Calculate final score
        news_score = (total_news * 0.3 + 
                     abs(avg_sentiment) * 0.3 + 
                     reddit_engagement * 0.1 + 
                     avg_relevance * 0.2 + 
                     source_diversity * 0.1)
        
        # Adjust score based on sentiment direction
        if avg_sentiment > 0:
            final_score = news_score * (1 + avg_sentiment)
        else:
            final_score = news_score * (1 - abs(avg_sentiment))
        
        # Add price momentum
        final_score += price_momentum
        
        # Generate reason
        reasons = []
        if total_news >= self.min_news_count:
            reasons.append(f"{total_news} news mentions")
        if abs(avg_sentiment) >= self.min_sentiment_score:
            sentiment_label = "positive" if avg_sentiment > 0 else "negative"
            reasons.append(f"{sentiment_label} sentiment ({avg_sentiment:.2f})")
        if reddit_engagement > 100:
            reasons.append(f"high Reddit engagement ({reddit_engagement:.0f})")
        if avg_relevance > 0.7:
            reasons.append(f"high news relevance ({avg_relevance:.2f})")
        if price_data and abs(price_data.get('price_change_pct', 0)) > 2:
            reasons.append(f"significant price movement ({price_data['price_change_pct']:.1f}%)")
        
        return {
            'score': final_score,
            'total_news': total_news,
            'avg_sentiment': avg_sentiment,
            'reddit_engagement': reddit_engagement,
            'avg_relevance': avg_relevance,
            'source_diversity': source_diversity,
            'price_momentum': price_momentum,
            'reason': '; '.join(reasons) if reasons else 'Limited data available'
        }
    
    def run_morning_analysis(self) -> Dict:
        """Run the complete morning stock analysis"""
        logger.info("Starting morning stock analysis...")
        
        # Get analysis time range
        start_time, end_time = self.get_analysis_time_range()
        logger.info(f"Analyzing news from {start_time} to {end_time}")
        
        # Collect news data from all sources
        all_stock_news = defaultdict(list)
        
        # Run data collection in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_reddit = executor.submit(self.get_reddit_stock_mentions, start_time, end_time)
            future_alphavantage = executor.submit(self.get_alpha_vantage_news, start_time, end_time)
            future_yahoo = executor.submit(self.get_yahoo_finance_news, start_time, end_time)
            
            # Collect results
            reddit_news = future_reddit.result()
            alphavantage_news = future_alphavantage.result()
            yahoo_news = future_yahoo.result()
        
        # Combine all news data
        for ticker, news in reddit_news.items():
            all_stock_news[ticker].extend(news)
        for ticker, news in alphavantage_news.items():
            all_stock_news[ticker].extend(news)
        for ticker, news in yahoo_news.items():
            all_stock_news[ticker].extend(news)
        
        logger.info(f"Collected news data for {len(all_stock_news)} stocks")

        # Prefilter: drop obvious non-tickers and validate online for frequent mentions to reduce noise
        filtered_stock_news = {}
        for ticker, news_items in all_stock_news.items():
            # Skip if mentions are below threshold; they won't be scored anyway
            if len(news_items) < self.min_news_count:
                continue
            # Format and quick pattern check (reuse extraction logic on the symbol itself)
            extracted = self.extract_stock_tickers(ticker, validate_online=False)
            if not extracted or extracted[0] != ticker:
                continue
            # Online validate with cache
            try:
                if hasattr(self, '_validate_symbol_online'):
                    is_valid = self._validate_symbol_online(ticker)
                else:
                    # Fallback to direct validation if helper not set
                    t = yf.Ticker(ticker)
                    is_valid = not t.history(period="5d").empty
            except Exception:
                is_valid = False
            if not is_valid:
                continue
            filtered_stock_news[ticker] = news_items

        all_stock_news = filtered_stock_news
        logger.info(f"After validation, {len(all_stock_news)} tickers remain for scoring")
        
        # Calculate scores for each stock
        stock_scores = {}
        for ticker, news_data in all_stock_news.items():
            price_data = self.get_stock_price_data(ticker)
            score_data = self.calculate_stock_score(ticker, news_data, price_data)
            
            if score_data['score'] > 0:
                stock_scores[ticker] = {
                    'score': score_data['score'],
                    'news_data': news_data,
                    'price_data': price_data,
                    'score_details': score_data,
                    'total_news': len(news_data)
                }
        
        # Sort stocks by score and get top performers
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        top_stocks = sorted_stocks[:self.max_stocks_to_return]
        
        logger.info(f"Identified {len(top_stocks)} top stocks to watch")
        
        return {
            'analysis_time': datetime.now(),
            'time_range': {'start': start_time, 'end': end_time},
            'top_stocks': top_stocks,
            'total_stocks_analyzed': len(all_stock_news),
            'stocks_with_sufficient_news': len(stock_scores)
        }
    
    def format_analysis_report(self, analysis_result: Dict) -> str:
        """Format the analysis results into a readable report with outlook and summary"""
        report = []
        # Header (simplified, fewer emojis)
        report.append("**Morning Stock Analysis Report**")
        report.append(f"Analysis Date: {analysis_result['analysis_time'].strftime('%Y-%m-%d %H:%M')}")
        report.append(f"Time Range: {analysis_result['time_range']['start'].strftime('%m/%d %H:%M')} - {analysis_result['time_range']['end'].strftime('%m/%d %H:%M')}")
        report.append(f"Stocks Analyzed: {analysis_result['total_stocks_analyzed']}")
        report.append(f"Stocks with Sufficient News: {analysis_result['stocks_with_sufficient_news']}")
        report.append("")
        
        if not analysis_result['top_stocks']:
            report.append("No stocks met the minimum criteria for today's watchlist.")
            return "\n".join(report)
        
        report.append("**Top stocks to watch today**")
        report.append("")
        
        for i, (ticker, data) in enumerate(analysis_result['top_stocks'], 1):
            score = data['score']
            price_data = data['price_data']
            score_details = data['score_details']
            news_items = data['news_data']
            
            report.append(f"**{i}. {ticker}** (Score: {score:.2f})")
            
            if price_data:
                current_price = price_data['current_price']
                price_change = price_data['price_change']
                price_change_pct = price_data['price_change_pct']
                price_emoji = "📈" if price_change >= 0 else "📉"
                report.append(f"   💰 Price: ${current_price:.2f} ({price_emoji} {price_change_pct:+.2f}%)")
                if price_data.get('sector'):
                    report.append(f"   🏢 Sector: {price_data['sector']}")
            
            # Core metrics
            report.append(f"   📰 News Count: {data['total_news']}")
            report.append(f"   Sentiment: {score_details['avg_sentiment']:+.2f}")
            report.append(f"   💡 Reason: {score_details['reason']}")
            
            # Expected move and concise summary driven by sentiment and momentum
            avg_sentiment = score_details.get('avg_sentiment', 0.0)
            total_news = data.get('total_news', 0)
            price_change_pct = price_data.get('price_change_pct', 0.0) if price_data else 0.0
            
            if avg_sentiment >= 0.10 or price_change_pct >= 1.0:
                outlook_label = "Likely Rise"
            elif avg_sentiment <= -0.10 or price_change_pct <= -1.0:
                outlook_label = "Likely Fall"
            else:
                outlook_label = "Neutral / Range-bound"
            
            # Source breadth
            distinct_sources = len({item.get('source', 'unknown') for item in news_items})
            source_text = "multi-source coverage" if distinct_sources >= 2 else "limited coverage"
            
            # Momentum descriptor
            if price_change_pct >= 0.5:
                momentum_text = "upward momentum"
            elif price_change_pct <= -0.5:
                momentum_text = "downward momentum"
            else:
                momentum_text = "flat momentum"
            
            report.append(
                f"   Outlook: {outlook_label} — sentiment {avg_sentiment:+.2f}; news {total_news}; momentum {price_change_pct:+.1f}%"
            )
            report.append(
                f"   Summary: {('Positive' if avg_sentiment > 0.05 else 'Negative' if avg_sentiment < -0.05 else 'Mixed')} sentiment with {source_text} and {momentum_text}."
            )
            
            # Top news headlines
            top_news = sorted(news_items, key=lambda x: x.get('score', 0), reverse=True)[:3]
            if top_news:
                report.append("   📋 Top Headlines:")
                for news in top_news:
                    title = news['title'][:80] + "..." if len(news['title']) > 80 else news['title']
                    report.append(f"      • {title}")
            
            report.append("")
        
        report.append("📝 **ANALYSIS SUMMARY**")
        report.append("This analysis is based on:")
        report.append("• Reddit discussions and sentiment")
        report.append("• Alpha Vantage news sentiment")
        report.append("• Yahoo Finance news coverage")
        report.append("• Price momentum and technical indicators")
        report.append("")
        report.append("⚠️ **DISCLAIMER**: This is for informational purposes only. Always do your own research before making investment decisions.")
        
        return "\n".join(report)
    
    def send_slack_notification(self, message: str, channel: str = "#stock-price-alerts") -> None:
        """Send notification to Slack using the same logic as enhanced_stock_alert.py"""
        try:
            if self.slack_bot_token:
                # Use Slack API with bot token to specify channel
                response = requests.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={"Authorization": f"Bearer {self.slack_bot_token}"},
                    json={
                        "channel": channel,
                        "text": message,
                        "username": "Morning Stock Analyzer",
                        "icon_emoji": ":chart_with_upwards_trend:"
                    }
                )
                result = response.json()
                if not result["ok"]:
                    logger.error(f"Failed to send Slack message: {result.get('error', 'Unknown error')}")
                else:
                    logger.info(f"Successfully sent Slack notification to {channel}")
            elif self.slack_webhook_url:
                # Fallback to webhook if bot token not available
                payload = {
                    "text": message,
                    "username": "Morning Stock Analyzer",
                    "icon_emoji": ":chart_with_upwards_trend:"
                }
                response = requests.post(self.slack_webhook_url, json=payload, timeout=10)
                if response.status_code == 200:
                    logger.info(f"Successfully sent Slack webhook notification")
                else:
                    logger.error(f"Failed to send Slack webhook: {response.status_code}")
            else:
                logger.warning("No Slack credentials configured")
                
        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
    
    def save_analysis_to_file(self, analysis_result: Dict, filename: str = "reports/morning_stock_analysis.txt") -> None:
        """Save analysis results to a file (defaulting to reports/)"""
        try:
            report = self.format_analysis_report(analysis_result)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w') as f:
                f.write(report)
            
            logger.info(f"Analysis saved to {filename}")
        
        except Exception as e:
            logger.error(f"Failed to save analysis to file: {str(e)}")
    
    def run_scheduled_analysis(self) -> None:
        """Run the scheduled morning analysis"""
        try:
            logger.info("Running scheduled morning stock analysis...")
            
            # Run the analysis
            analysis_result = self.run_morning_analysis()
            
            # Format the report
            report = self.format_analysis_report(analysis_result)
            
            # Send to Slack
            self.send_slack_notification(report)
            
            # Save to file
            self.save_analysis_to_file(analysis_result)
            
            logger.info("Morning analysis completed successfully")
        
        except Exception as e:
            error_msg = f"Error in scheduled morning analysis: {str(e)}"
            logger.error(error_msg)
            self.send_slack_notification(f"❌ {error_msg}")
    
    def schedule_daily_analysis(self) -> None:
        """Schedule the analysis to run at 8:30 AM every weekday"""
        schedule.every().monday.at("08:30").do(self.run_scheduled_analysis)
        schedule.every().tuesday.at("08:30").do(self.run_scheduled_analysis)
        schedule.every().wednesday.at("08:30").do(self.run_scheduled_analysis)
        schedule.every().thursday.at("08:30").do(self.run_scheduled_analysis)
        schedule.every().friday.at("08:30").do(self.run_scheduled_analysis)
        
        logger.info("Scheduled morning analysis for 8:30 AM every weekday")
        
        # Keep the script running
        while True:
            schedule.run_pending()
            time_module.sleep(60)

def main():
    """Main function to run the morning stock analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Morning Stock Analysis Tool")
    parser.add_argument("--run-once", action="store_true", help="Run analysis once and exit")
    parser.add_argument("--schedule", action="store_true", help="Schedule daily analysis at 8:30 AM")
    parser.add_argument("--config", default="configs/config.json", help="Path to config file")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MorningStockAnalyzer(args.config)
    
    if args.run_once:
        # Run analysis once
        analysis_result = analyzer.run_morning_analysis()
        report = analyzer.format_analysis_report(analysis_result)
        print(report)
        analyzer.send_slack_notification(report)
        analyzer.save_analysis_to_file(analysis_result)
    
    elif args.schedule:
        # Schedule daily analysis
        analyzer.schedule_daily_analysis()
    
    else:
        # Default: run once
        analysis_result = analyzer.run_morning_analysis()
        report = analyzer.format_analysis_report(analysis_result)
        print(report)

if __name__ == "__main__":
    main() 