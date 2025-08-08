#!/usr/bin/env python3
"""
Test script for the Real-Time Stock Prediction System
"""

import sys
import os
import json
from datetime import datetime

# Add src directory to path
sys.path.append('src')

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")
    except ImportError as e:
        print(f"❌ yfinance import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print("✅ tensorflow imported successfully")
    except ImportError as e:
        print(f"❌ tensorflow import failed: {e}")
        return False
    
    try:
        from sklearn.preprocessing import MinMaxScaler
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    
    try:
        from textblob import TextBlob
        print("✅ textblob imported successfully")
    except ImportError as e:
        print(f"❌ textblob import failed: {e}")
        return False
    
    try:
        import schedule
        print("✅ schedule imported successfully")
    except ImportError as e:
        print(f"❌ schedule import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration file"""
    print("\nTesting configuration...")
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        required_keys = ['ticker', 'lookback_days', 'prediction_hours', 'alert_time']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required config key: {key}")
                return False
        
        print("✅ Configuration file loaded successfully")
        print(f"   Ticker: {config['ticker']}")
        print(f"   Lookback days: {config['lookback_days']}")
        print(f"   Prediction hours: {config['prediction_hours']}")
        print(f"   Alert time: {config['alert_time']}")
        
        return True
        
    except FileNotFoundError:
        print("❌ config.json not found")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config.json: {e}")
        return False

def test_stock_data():
    """Test stock data fetching"""
    print("\nTesting stock data fetching...")
    
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        
        # Load config
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        ticker = config['ticker']
        
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = yf.download(
            ticker, 
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )
        
        if data.empty:
            print(f"❌ No data found for {ticker}")
            return False
        
        print(f"✅ Successfully fetched {len(data)} days of data for {ticker}")
        print(f"   Latest close price: ${data['Close'].iloc[-1]:.2f}")
        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fetching stock data: {e}")
        return False

def test_predictor_class():
    """Test the predictor class"""
    print("\nTesting predictor class...")
    
    try:
        from enhanced_stock_alert import EnhancedStockPredictor
        
        # Initialize predictor
        predictor = EnhancedStockPredictor()
        
        print("✅ EnhancedStockPredictor initialized successfully")
        print(f"   Ticker: {predictor.ticker}")
        print(f"   Lookback days: {predictor.lookback_days}")
        print(f"   Prediction hours: {predictor.prediction_hours}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    print("\nTesting sentiment analysis...")
    
    try:
        from textblob import TextBlob
        
        # Test TextBlob
        test_text = "This is a great stock with amazing potential!"
        blob = TextBlob(test_text)
        sentiment = blob.sentiment.polarity
        
        print(f"✅ TextBlob sentiment analysis working")
        print(f"   Test text: '{test_text}'")
        print(f"   Sentiment score: {sentiment:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in sentiment analysis: {e}")
        return False

def test_model_building():
    """Test model building functionality"""
    print("\nTesting model building...")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        # Create a simple test model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(10, 5)),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Test with dummy data
        dummy_data = np.random.random((1, 10, 5))
        prediction = model.predict(dummy_data, verbose=0)
        
        print("✅ TensorFlow model building working")
        print(f"   Model input shape: {dummy_data.shape}")
        print(f"   Model output shape: {prediction.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model building: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Real-Time Stock Prediction System")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_stock_data,
        test_predictor_class,
        test_sentiment_analysis,
        test_model_building
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run: python src/enhanced_stock_alert.py")
        print("2. The system will start and run analysis at 12pm daily")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Run: python setup.py")
        print("2. Check your internet connection")
        print("3. Verify the stock ticker in config.json")

if __name__ == "__main__":
    main() 