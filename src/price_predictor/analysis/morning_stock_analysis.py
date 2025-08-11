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
        """Load Alpha Vantage API key using the same logic as enhanced_stock_alert.py"""
        try:
            # Check for Alpha Vantage key in new secrets directory
            if os.path.exists('secrets/alphavantage.txt'):
                with open('secrets/alphavantage.txt', 'r') as f:
                    api_key = f.read().strip()
                logger.info("Alpha Vantage API key loaded from secrets/alphavantage.txt")
                return api_key
            else:
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
        """Initialize Reddit client using the logic from the enhanced_stock_alert.py"""
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
                
                # Use the same authentication logic as the enhanced script
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
    
    def get_reddit_stock_mentions(self, start_time: datetime, end_time: datetime) -> Dict[str, List[Dict]]:
        """Get stock mentions from Reddit subreddits using the enhanced script logic"""
        if not self.reddit_client:
            return {}
        
        stock_mentions = defaultdict(list)
        
        try:
            headers = self.reddit_client['headers']
            
            for subreddit_name in self.stock_subreddits:
                try:
                    # Get posts from the subreddit using the enhanced script logic
                    res = requests.get(f"https://oauth.reddit.com/r/{subreddit_name}/new", 
                                     headers=headers, params={"limit": 100})
                    
                    if res.status_code == 200:
                        posts = res.json()['data']['children']
                        
                        for post in posts:
                            post_data = post['data']
                            post_time = datetime.fromtimestamp(post_data['created_utc'])
                            
                            # Check if post is within our time range
                            if start_time <= post_time <= end_time:
                                # Extract stock tickers from title and content
                                title = post_data.get('title', '')
                                selftext = post_data.get('selftext', '')
                                text = f"{title} {selftext}"
                                tickers = self.extract_stock_tickers(text)
                                
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
                    
                    # Get comments from hot posts
                    res = requests.get(f"https://oauth.reddit.com/r/{subreddit_name}/hot", 
                                     headers=headers, params={"limit": 20})
                    
                    if res.status_code == 200:
                        posts = res.json()['data']['children']
                        
                        for post in posts:
                            post_data = post['data']
                            post_id = post_data.get('id', '')
                            
                            # Get comments for this post
                            comments_res = requests.get(f"https://oauth.reddit.com/comments/{post_id}", 
                                                       headers=headers, params={"limit": 50})
                            
                            if comments_res.status_code == 200:
                                comments_data = comments_res.json()
                                if len(comments_data) > 1:  # Comments are in the second element
                                    comments = comments_data[1]['data']['children']
                                    
                                    for comment in comments:
                                        if comment['kind'] == 't1':  # Regular comment
                                            comment_data = comment['data']
                                            comment_time = datetime.fromtimestamp(comment_data['created_utc'])
                                            
                                            if start_time <= comment_time <= end_time:
                                                comment_body = comment_data.get('body', '')
                                                tickers = self.extract_stock_tickers(comment_body)
                                                
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
        """Get news from Yahoo Finance"""
        stock_news = defaultdict(list)
        
        try:
            # Get news for major stock categories
            for category, tickers in self.stock_categories.items():
                for ticker in tickers[:5]:  # Limit to top 5 per category
                    try:
                        stock = yf.Ticker(ticker)
                        news = stock.news
                        
                        for article in news:
                            article_time = datetime.fromtimestamp(article['providerPublishTime'])
                            
                            if start_time <= article_time <= end_time:
                                stock_news[ticker].append({
                                    'source': 'yahoofinance',
                                    'title': article['title'],
                                    'summary': article.get('summary', ''),
                                    'url': article['link'],
                                    'timestamp': article_time,
                                    'sentiment': self.analyze_sentiment(article['title'] + ' ' + article.get('summary', '')),
                                    'publisher': article.get('publisher', 'Unknown')
                                })
                    
                    except Exception as e:
                        logger.warning(f"Error getting Yahoo Finance news for {ticker}: {str(e)}")
                        continue
        
        except Exception as e:
            logger.error(f"Error in Yahoo Finance analysis: {str(e)}")
        
        return dict(stock_news)
    
    def extract_stock_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text"""
        # Common stock ticker patterns
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        
        # Filter out common words that aren't tickers
        common_words = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
            'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD',
            'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'
        }
        
        # Filter out words that are too short or too long
        valid_tickers = []
        for ticker in potential_tickers:
            if (len(ticker) >= 2 and len(ticker) <= 5 and 
                ticker not in common_words and 
                ticker.isalpha()):
                valid_tickers.append(ticker)
        
        return list(set(valid_tickers))
    
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
        
        # Calculate scores for each stock
        stock_scores = {}
        for ticker, news_data in all_stock_news.items():
            if len(news_data) >= self.min_news_count:
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