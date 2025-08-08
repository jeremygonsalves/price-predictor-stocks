#!/usr/bin/env python3
"""
Real-Time Stock Price Prediction and Trading Alert System
Provides buy/sell recommendations at 12pm based on 4-hour price predictions
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_alerts.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealTimeStockPredictor:
    def __init__(self, ticker: str, lookback_days: int = 60, prediction_hours: int = 4):
        """
        Initialize the real-time stock predictor
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days to look back for training
            prediction_hours: Hours ahead to predict (default 4 hours)
        """
        self.ticker = ticker.upper()
        self.lookback_days = lookback_days
        self.prediction_hours = prediction_hours
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.last_training_date = None
        self.current_open_price = None
        
    def get_historical_data(self, days: int = 365) -> pd.DataFrame:
        """Fetch historical stock data"""
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
            
            # Add technical indicators
            data = self.add_technical_indicators(data)
            
            logger.info(f"Fetched {len(data)} days of historical data for {self.ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset"""
        # Moving averages
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
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
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        
        # Price changes
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Change_5'] = data['Close'].pct_change(periods=5)
        
        # Fill NaN values
        data = data.fillna(method='bfill').fillna(0)
        
        return data
    
    def get_sentiment_data(self) -> pd.DataFrame:
        """Get sentiment data from Reddit and news sources"""
        try:
            # For now, we'll create dummy sentiment data
            # In a real implementation, you would integrate with Reddit API and news APIs
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                end=datetime.now(), freq='D')
            
            sentiment_data = pd.DataFrame({
                'date': dates,
                'reddit_polarity': np.random.normal(0, 0.3, len(dates)),
                'reddit_volume': np.random.randint(10, 100, len(dates)),
                'news_polarity': np.random.normal(0, 0.2, len(dates)),
                'news_volume': np.random.randint(5, 50, len(dates))
            })
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error fetching sentiment data: {str(e)}")
            return pd.DataFrame()
    
    def prepare_data(self, stock_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        try:
            # Select features for training
            feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'MA_5', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal',
                'BB_Upper', 'BB_Lower', 'Volume_Ratio', 'Price_Change', 'Price_Change_5'
            ]
            
            # Normalize price features
            price_features = stock_data[feature_columns].values
            normalized_price = self.price_scaler.fit_transform(price_features)
            
            # Prepare sentiment features
            if not sentiment_data.empty:
                sentiment_features = sentiment_data[['reddit_polarity', 'reddit_volume', 
                                                   'news_polarity', 'news_volume']].values
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
        """Build the LSTM model"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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
            # Check if we need to retrain (only retrain once per day unless forced)
            today = datetime.now().date()
            if (not force_retrain and 
                self.last_training_date == today and 
                self.model is not None):
                logger.info("Model already trained today, skipping training")
                return
            
            logger.info("Starting model training...")
            
            # Get data
            stock_data = self.get_historical_data()
            sentiment_data = self.get_sentiment_data()
            
            # Prepare data
            X, y = self.prepare_data(stock_data, sentiment_data)
            
            if len(X) < 100:  # Need sufficient data
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
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
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
            sentiment_data = self.get_sentiment_data()
            
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
            
            # Define thresholds
            BUY_THRESHOLD = 1.0  # 1% increase
            SELL_THRESHOLD = -1.0  # 1% decrease
            CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for signal
            
            # Determine signal
            if price_change_pct > BUY_THRESHOLD:
                signal = 'BUY'
                confidence = min(abs(price_change_pct) / 5.0, 1.0)  # Scale confidence
            elif price_change_pct < SELL_THRESHOLD:
                signal = 'SELL'
                confidence = min(abs(price_change_pct) / 5.0, 1.0)  # Scale confidence
            else:
                signal = 'HOLD'
                confidence = 0.5
            
            return {
                'signal': signal,
                'reason': f'Predicted {price_change_pct:.2f}% change in {self.prediction_hours} hours',
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
        """Send trading alert"""
        try:
            # Create alert message
            alert_msg = f"""
🚨 STOCK ALERT: {self.ticker} 🚨

Signal: {signal_data['signal']}
Current Price: ${signal_data['current_price']:.2f}
Predicted Price: ${signal_data['predicted_price']:.2f}
Change: {signal_data.get('price_change_pct', 0):.2f}%
Confidence: {signal_data['confidence']:.1%}
Reason: {signal_data['reason']}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
            
            # Log the alert
            logger.info(f"Trading Alert:\n{alert_msg}")
            
            # Save to file
            with open('trading_alerts.txt', 'a') as f:
                f.write(f"\n{alert_msg}\n{'='*50}\n")
            
            # Here you could add integrations for:
            # - Email notifications
            # - SMS alerts
            # - Slack/Discord webhooks
            # - Trading platform APIs
            
            print(alert_msg)
            
        except Exception as e:
            logger.error(f"Error sending alert: {str(e)}")
    
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
    """Main function to run the real-time stock predictor"""
    # Configuration
    TICKER = "NVDA"  # Change this to your desired stock
    LOOKBACK_DAYS = 60
    PREDICTION_HOURS = 4
    
    # Initialize predictor
    predictor = RealTimeStockPredictor(
        ticker=TICKER,
        lookback_days=LOOKBACK_DAYS,
        prediction_hours=PREDICTION_HOURS
    )
    
    # Schedule daily analysis at 12:00 PM
    schedule.every().day.at("12:00").do(predictor.run_daily_analysis)
    
    # Also run immediately for testing
    logger.info("Running initial analysis...")
    predictor.run_daily_analysis()
    
    logger.info(f"Stock predictor started for {TICKER}")
    logger.info("Scheduled to run daily at 12:00 PM")
    logger.info("Press Ctrl+C to stop")
    
    # Keep the script running
    try:
        while True:
            schedule.run_pending()
            time_module.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Stock predictor stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main() 