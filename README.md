# Real-Time Stock Price Prediction System

A sophisticated real-time stock prediction system that provides buy/sell recommendations at 12pm based on 4-hour price predictions. The system uses LSTM neural networks with sentiment analysis from Reddit and news sources to make accurate predictions.

## Features

- **Real-time Predictions**: Predicts stock prices 4 hours ahead
- **Automated Alerts**: Sends buy/sell signals at 12pm daily
- **Sentiment Analysis**: Integrates Reddit and news sentiment data
- **Technical Indicators**: Uses comprehensive technical analysis
- **LSTM Neural Networks**: Advanced deep learning for price prediction
- **Configurable**: Easy to customize via JSON configuration
- **Logging**: Comprehensive logging and alert history

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd price-predictor-stocks

# Run the setup script
python setup.py
```

### 2. Configuration

Edit `config.json` to customize your settings:

```json
{
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
    "enable_sentiment_analysis": true,
    "enable_technical_indicators": true,
    "log_level": "INFO",
    "save_alerts_to_file": true,
    "alert_file": "trading_alerts.txt",
    "log_file": "stock_alerts.log"
}
```

### 3. Optional: Reddit API Setup

For enhanced sentiment analysis, set up Reddit API credentials:

1. Go to https://www.reddit.com/prefs/apps
2. Create a new app
3. Edit the credential files with your credentials:

```bash
# Edit client_id.txt
echo "YOUR_CLIENT_ID" > client_id.txt

# Edit client_secret.txt  
echo "YOUR_CLIENT_SECRET" > client_secret.txt

# Edit pw.txt
echo "YOUR_REDDIT_PASSWORD" > pw.txt
```

**Note**: The system uses username `InterestingRun2732` by default (as in the original notebook).

### 4. Run the System

```bash
# Run the enhanced version (recommended)
python src/enhanced_stock_alert.py

# Or run the basic version
python src/real_time_stock_alert.py
```

## How It Works

### 1. Data Collection
- **Historical Data**: Fetches daily stock data using Yahoo Finance API
- **Technical Indicators**: Calculates 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Sentiment Data**: Collects sentiment from Reddit and news sources

### 2. Model Training
- **LSTM Neural Network**: Uses Long Short-Term Memory networks for time series prediction
- **Feature Engineering**: Combines price data, technical indicators, and sentiment
- **Sequence Learning**: Learns patterns from historical data sequences

### 3. Prediction Process
- **Real-time Data**: Gets current stock price
- **4-Hour Forecast**: Predicts price 4 hours ahead
- **Signal Generation**: Compares current vs predicted price
- **Alert System**: Sends buy/sell recommendations

### 4. Alert System
- **Daily Schedule**: Runs analysis at 12pm daily
- **Signal Types**: BUY, SELL, or HOLD recommendations
- **Confidence Scoring**: Provides confidence levels for predictions
- **Logging**: Saves all alerts to file

## Configuration Options

### Stock Settings
- `ticker`: Stock symbol (e.g., "NVDA", "AAPL", "TSLA")
- `lookback_days`: Historical data period for training
- `prediction_hours`: Hours ahead to predict (default: 4)

### Alert Settings
- `alert_time`: Time to run daily analysis (format: "HH:MM")
- `buy_threshold`: Minimum % increase for BUY signal
- `sell_threshold`: Minimum % decrease for SELL signal
- `confidence_threshold`: Minimum confidence for signal

### Model Settings
- `training_epochs`: Number of training iterations
- `batch_size`: Training batch size
- `learning_rate`: Model learning rate

### Features
- `enable_sentiment_analysis`: Enable/disable sentiment analysis
- `enable_technical_indicators`: Enable/disable technical indicators
- `save_alerts_to_file`: Save alerts to file

## Technical Indicators Used

- **Moving Averages**: 5, 10, 20, 50, 200-day
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Upper, Lower, Middle bands
- **Volume Analysis**: Volume ratios and trends
- **Price Changes**: Daily and weekly price changes
- **Volatility**: Price volatility measures
- **Support/Resistance**: Dynamic support and resistance levels

## Sentiment Analysis

### Reddit Sentiment
- Analyzes posts from r/stocks, r/investing
- Searches for mentions of the target stock
- Calculates weighted sentiment scores
- Considers upvotes, comments, and post scores

### News Sentiment
- Collects news from Yahoo Finance
- Analyzes article headlines and content
- Provides real-time sentiment scores
- Tracks news volume and sentiment trends

## Output Examples

### Alert Message
```
🟢 STOCK ALERT: NVDA 🟢

📊 Signal: BUY
💰 Current Price: $145.67
🎯 Predicted Price: $148.92
📈 Change: +2.23%
🎯 Confidence: 89.2%
📝 Reason: Predicted +2.23% change in 4 hours

⏰ Time: 2024-01-15 12:00:00
🔧 Model: Enhanced LSTM with Sentiment Analysis
```

### Log File
```
2024-01-15 12:00:01 - INFO - Starting daily analysis for NVDA
2024-01-15 12:00:05 - INFO - Model training completed. Test MAE: 0.0234
2024-01-15 12:00:08 - INFO - Trading Alert: [BUY signal details]
2024-01-15 12:00:08 - INFO - Daily analysis completed for NVDA
```

## File Structure

```
price-predictor-stocks/
├── src/
│   ├── real_time_stock_alert.py      # Basic version
│   ├── enhanced_stock_alert.py       # Enhanced version (recommended)
│   └── stock-price.ipynb             # Original notebook
├── config.json                       # Configuration file
├── requirements.txt                  # Python dependencies
├── setup.py                         # Setup script
├── reddit_credentials.json          # Reddit API credentials (optional)
├── trading_alerts.txt               # Alert history
├── stock_alerts.log                 # System logs
└── README.md                        # This file
```

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **yfinance**: Yahoo Finance API
- **tensorflow**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **textblob**: Sentiment analysis
- **schedule**: Task scheduling
- **praw**: Reddit API client
- **requests**: HTTP requests

## Troubleshooting

### Common Issues

1. **Import Errors**: Run `python setup.py` to install dependencies
2. **Reddit API Errors**: Check your credentials in `reddit_credentials.json`
3. **No Data**: Verify the stock ticker symbol is correct
4. **Model Training Issues**: Increase `lookback_days` or `training_epochs`

### Performance Tips

- Use GPU acceleration for faster training (requires CUDA)
- Adjust `batch_size` based on your system's memory
- Reduce `training_epochs` for quicker testing
- Disable sentiment analysis if not needed

## Disclaimer

⚠️ **Important**: This system is for educational and research purposes only. Stock predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always do your own research and consider consulting with financial advisors before making investment decisions.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
