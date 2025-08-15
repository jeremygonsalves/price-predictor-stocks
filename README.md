# Real-Time Stock Price Prediction System

A sophisticated real-time stock prediction system that provides buy/sell recommendations at 12pm based on 4-hour price predictions. The system uses **config-driven ensemble methods** with LSTM neural networks and sentiment analysis from Reddit and news sources to make accurate predictions.

## Features

- **Real-time Predictions**: Predicts stock prices 4 hours ahead
- **Automated Alerts**: Sends buy/sell signals at 12pm daily
- **Morning Stock Analysis**: Identifies top stocks to watch at 8:30 AM daily
- **Sentiment Analysis**: Integrates Reddit and news sentiment data
- **Technical Indicators**: Uses comprehensive technical analysis
- **LSTM Neural Networks**: Advanced deep learning for price prediction
- **Config-Driven Architecture**: All weights and metrics inherited from config.json
- **Enhanced Price Fetching**: Multiple fallback methods for reliable price data
- **Currency Handling**: Automatic currency detection and formatting
- **Robust Error Handling**: Graceful handling of rate limiting and failures
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

Edit `config.json` to customize your settings. The system now uses **config inheritance** - all weights and metrics are controlled from this file:

```json
{
  "ticker": "MSFT",
  "lookback_days": 10,
  "prediction_hours": 4,
  "alert_time": "12:00",
  "buy_threshold": 0.5,
  "sell_threshold": -0.5,
  "confidence_threshold": 0.7,
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

Advanced settings with **config-driven ensemble weights**:

```json
{
  "min_required_days": 20,
  "min_training_samples": 30,
  "min_prediction_days": 5,
  "min_fallback_days": 2,
  "min_analysis_days": 10,
  "extra_days_for_safety": 20,
  "fallback_period": "1y",
  "current_price_days": 5,
  "date_range_days": 7,
  "recent_days_for_prediction": 15,
  "fallback_days": 5,
  "sentiment_analysis_limit": 50,
  "news_article_limit": 50,
  "use_enhanced_prediction": true,
  "intraday_interval": "5m",
  "intraday_days": 5,
  "ensemble_weights": {
    "technical_analysis": 0.5,
    "sentiment_based": 0.2,
    "microstructure": 0.15,
    "mean_reversion": 0.15
  },
  "real_time_sentiment": {
    "enable_news_sentiment": true,
    "enable_social_sentiment": true,
    "enable_earnings_impact": true,
    "enable_analyst_impact": true,
    "enable_options_flow": true,
    "sentiment_hours": 24
  },
  "microstructure_features": {
    "enable_spread_analysis": true,
    "enable_volume_analysis": true,
    "enable_market_efficiency": true,
    "enable_time_based_adjustments": true
  },
  "prediction_bounds": {
    "max_daily_change": 0.2,
    "max_intraday_change": 0.15,
    "confidence_dampening": 0.5
  }
}
```

### 3. Authentication Setup

#### **Reddit API Setup (Required for Sentiment Analysis)**

For enhanced sentiment analysis, you **must** set up your own Reddit API credentials:

1. Go to https://www.reddit.com/prefs/apps
2. Create a new app (select "script" type)
3. Create the following files in the `secrets/` directory:

```bash
# Create secrets directory
mkdir -p secrets

# Create username file (your Reddit username)
echo "YOUR_REDDIT_USERNAME" > secrets/username.txt

# Create password file (your Reddit password)
echo "YOUR_REDDIT_PASSWORD" > secrets/pw.txt

# Create client ID file (from your Reddit app)
echo "YOUR_CLIENT_ID" > secrets/client_id.txt

# Create client secret file (from your Reddit app)
echo "YOUR_CLIENT_SECRET" > secrets/client_secret.txt
```

**Important**: 
- You must use your own Reddit credentials - the system will not work with default credentials
- Keep your credentials secure and never commit them to version control
- The `secrets/` directory is already in `.gitignore` for security

#### **Slack API Setup (Optional for Notifications)**

For Slack notifications, set up one of the following:

**Option 1: Webhook URL (Simpler)**
```bash
# Add to your ~/.env file
echo "SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL" >> ~/.env
```

**Option 2: Bot Token (More Features)**
```bash
# Add to your ~/.env file
echo "SLACK_BOT_TOKEN=xoxb-YOUR-BOT-TOKEN" >> ~/.env
```

**Note**: If no Slack credentials are provided, the system will still work but won't send Slack notifications.

### 4. Run the System

```bash
# Run the enhanced version (recommended)
python src/price_predictor/alerts/enhanced_stock_alert.py

# Or run the basic version
python src/price_predictor/alerts/real_time_stock_alert.py
```

## How It Works

### 1. Data Collection
- **Historical Data**: Fetches daily stock data using Yahoo Finance API
- **Technical Indicators**: Calculates 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Sentiment Data**: Collects sentiment from Reddit and news sources
- **Enhanced Price Fetching**: Multiple fallback methods for reliable price data
- **Currency Detection**: Automatic currency detection and formatting

### 2. Model Training
- **LSTM Neural Network**: Uses Long Short-Term Memory networks for time series prediction
- **Feature Engineering**: Combines price data, technical indicators, and sentiment
- **Sequence Learning**: Learns patterns from historical data sequences
- **Config-Driven Weights**: All ensemble weights inherited from config.json

### 3. Prediction Process
- **Real-time Data**: Gets current stock price with enhanced methods
- **4-Hour Forecast**: Predicts price 4 hours ahead using config-driven ensemble
- **Signal Generation**: Compares current vs predicted price
- **Alert System**: Sends buy/sell recommendations with currency formatting

### 4. Alert System
- **Daily Schedule**: Runs analysis at 12pm daily
- **Signal Types**: BUY, SELL, or HOLD recommendations
- **Confidence Scoring**: Provides confidence levels for predictions
- **Logging**: Saves all alerts to file
- **Currency Display**: Clear currency indicators in all outputs

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

### Data and Training Thresholds
- `min_required_days`, `min_training_samples`, `min_prediction_days`, `min_fallback_days`, `min_analysis_days`
- `extra_days_for_safety`, `fallback_period`, `fallback_days`

### Price Fetch and Windowing
- `current_price_days`, `date_range_days`, `recent_days_for_prediction`

### Intraday Settings
- `intraday_interval` (e.g., "5m"), `intraday_days`

### Config-Driven Ensemble Weights
- `ensemble_weights.technical_analysis`, `sentiment_based`, `microstructure`, `mean_reversion`

### Config-Driven Real-Time Sentiment Toggles
- `real_time_sentiment.enable_news_sentiment`, `enable_social_sentiment`, `enable_earnings_impact`, `enable_analyst_impact`, `enable_options_flow`, `sentiment_hours`

### Config-Driven Microstructure Features Toggles
- `microstructure_features.enable_spread_analysis`, `enable_volume_analysis`, `enable_market_efficiency`, `enable_time_based_adjustments`

### Config-Driven Prediction Bounds
- `prediction_bounds.max_daily_change`, `max_intraday_change`, `confidence_dampening`

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

### Alert Message with Currency Formatting
```
🟢 STOCK ALERT: NVDA 🟢

📊 Signal: BUY
💰 Current Price: $145.67 USD
🎯 Predicted Price: $148.92 USD
📈 Change: +2.23%
🎯 Confidence: 89.2%
📝 Reason: Predicted +2.23% change in 4 hours

⏰ Time: 2024-01-15 12:00:00
🔧 Model: Enhanced LSTM with Config-Driven Ensemble
```

### Log File
```
2024-01-15 12:00:01 - INFO - Starting daily analysis for NVDA
2024-01-15 12:00:05 - INFO - Model training completed. Test MAE: 0.0234
2024-01-15 12:00:08 - INFO - Trading Alert: [BUY signal details]
2024-01-15 12:00:08 - INFO - Daily analysis completed for NVDA
```

## Morning Stock Analysis

The system now includes a comprehensive morning stock analysis feature that runs at 8:30 AM every weekday. This feature analyzes news from multiple sources to identify the top stocks to watch for the day.

### Features
- **Multi-Source News Analysis**: Reddit, Alpha Vantage, and Yahoo Finance
- **Time-Based Analysis**: Analyzes news from 4 PM previous day to 8 AM current day
- **Sentiment Scoring**: Combines news sentiment with price momentum
- **Automated Scheduling**: Runs automatically at 8:30 AM weekdays
- **Slack Integration**: Sends detailed reports to Slack
- **Comprehensive Scoring**: Considers news volume, sentiment, engagement, and price movement

### Usage

```bash
# Run morning analysis once
python src/price_predictor/analysis/morning_stock_analysis.py --run-once

# Schedule daily analysis at 8:30 AM (runner-based cron)
/Users/jeremygonsalves/python_runner.sh morning_analysis schedule

# Using the python runner script (one-off)
./python_runner.sh morning_analysis

# Add to crontab manually (8:30 AM weekdays)
# crontab -e
# Add: 30 8 * * 1-5 /Users/jeremygonsalves/python_runner.sh morning_analysis >> /Users/jeremygonsalves/logs/morning_analysis.log 2>&1
```

### Sample Output
```
🚀 MORNING STOCK ANALYSIS REPORT 🚀
📅 Analysis Date: 2024-01-15 08:30
⏰ Time Range: 01/14 16:00 - 01/15 08:00
📊 Stocks Analyzed: 45
📈 Stocks with Sufficient News: 12

🔥 TOP STOCKS TO WATCH TODAY 🔥

1. NVDA (Score: 8.45)
   💰 Price: $145.67 USD (📈 +2.34%)
   🏢 Sector: Technology
   📰 News Count: 15
   😊 Sentiment: 0.67
   💡 Reason: 15 news mentions; positive sentiment (0.67); high Reddit engagement (245)
   📋 Top Headlines:
      • NVIDIA Announces New AI Chip Breakthrough
      • Analysts Upgrade NVDA Price Targets
      • Reddit Users Bullish on NVIDIA's Future
```

### Configuration
The morning analysis uses the same configuration file (`config.json`) and environment variables as the main system. Make sure to set up:
- `ALPHA_VANTAGE_API_KEY`: For Alpha Vantage news API
- `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`: For Reddit sentiment analysis
- `SLACK_WEBHOOK_URL`: For Slack notifications

## Recent Enhancements (Latest Updates)

### Config Inheritance Implementation
- ✅ **All weights and metrics now inherited from config.json**
- ✅ **No more hardcoded values in the codebase**
- ✅ **Centralized configuration management**
- ✅ **Easy tuning without code changes**

### Enhanced Price Fetching
- ✅ **Multiple fallback methods for reliable price data**
- ✅ **Improved currency detection and formatting**
- ✅ **Better error handling for rate limiting**
- ✅ **Clear currency indicators in output**

### Robust Error Handling
- ✅ **Graceful handling of Yahoo Finance rate limiting**
- ✅ **Better logging and error messages**
- ✅ **No crashes on temporary failures**
- ✅ **Comprehensive fallback mechanisms**

### Currency Handling Improvements
- ✅ **Automatic currency detection from stock data**
- ✅ **Clear currency formatting (e.g., "$375.50 USD")**
- ✅ **Support for multiple currencies (USD, CAD, EUR, etc.)**
- ✅ **Proper currency symbols and codes**

### Authentication Security Improvements
- ✅ **Requires user's own Reddit credentials - no default credentials**
- ✅ **Removed hardcoded username for security**
- ✅ **Clear error messages when credentials are missing**
- ✅ **Optional Slack integration with proper credential validation**
- ✅ **Secure credential storage in secrets/ directory**

## File Structure

```
price-predictor-stocks/
├── src/
│   └── price_predictor/
│       ├── alerts/
│       │   ├── enhanced_stock_alert.py       # Enhanced version (recommended)
│       │   └── real_time_stock_alert.py      # Basic version
│       └── analysis/
│           └── morning_stock_analysis.py     # Morning stock analysis
├── configs/
│   └── config.json                          # Configuration file
├── secrets/                                 # Authentication credentials (create this)
│   ├── username.txt                         # Your Reddit username
│   ├── pw.txt                              # Your Reddit password
│   ├── client_id.txt                       # Your Reddit app client ID
│   ├── client_secret.txt                   # Your Reddit app client secret
│   └── alphavantage.txt                    # Your Alpha Vantage API key
├── logs/                                   # Log files and alerts
│   ├── stock_alerts.log                    # System logs
│   ├── trading_alerts.txt                  # Alert history
│   ├── morning_stock_analysis.log          # Morning analysis logs
│   └── cron.log                           # Cron job logs
├── data/                                   # Data files
│   └── NVDA.csv                           # Sample stock data
├── reports/                                # Generated reports
│   ├── charts/                            # Generated charts
│   └── morning_stock_analysis.txt         # Morning analysis reports
├── tests/                                  # Test files
│   ├── test_config_inheritance.py         # Config inheritance tests
│   ├── test_morning_analysis.py           # Morning analysis tests
│   ├── test_authentication.py             # Authentication tests
│   └── test_system.py                     # System tests
├── docs/                                   # Documentation
│   ├── LSTM-model.md                      # Technical documentation
│   └── summaries/                         # Implementation summaries
│       └── AUTHENTICATION_CHANGES_SUMMARY.md
├── notebooks/                              # Jupyter notebooks
├── requirements.txt                        # Python dependencies
├── setup.py                               # Setup script
└── README.md                              # This file
```

**Note**: The `secrets/` directory is not included in the repository for security reasons. You must create it and add your own credentials.

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

## Testing

The system includes comprehensive testing for the new config inheritance features:

```bash
# Test config inheritance
python test_config_inheritance.py

# Test morning analysis
python tests/test_morning_analysis.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Run `python setup.py` to install dependencies
2. **Reddit API Errors**: 
   - Check your credentials in the `secrets/` directory
   - Ensure all 4 required files exist: `username.txt`, `pw.txt`, `client_id.txt`, `client_secret.txt`
   - Verify your Reddit app is set to "script" type
   - Check that your Reddit username and password are correct
3. **Slack API Errors**: 
   - Check your environment variables in `~/.env`
   - Verify your webhook URL or bot token is correct
   - Ensure your Slack app has the necessary permissions
4. **No Data**: Verify the stock ticker symbol is correct
5. **Model Training Issues**: Increase `lookback_days` or `training_epochs`
6. **Rate Limiting**: The system now handles Yahoo Finance rate limiting gracefully
7. **Currency Issues**: Currency detection is now automatic and robust
8. **Authentication Required**: The system now requires your own Reddit credentials - it won't work with default credentials

### Performance Tips

- Use GPU acceleration for faster training (requires CUDA)
- Adjust `batch_size` based on your system's memory
- Reduce `training_epochs` for quicker testing
- Disable sentiment analysis if not needed
- Adjust ensemble weights in config.json for optimal performance

## Disclaimer

⚠️ **Important**: This system is for educational and research purposes only. Stock predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always do your own research and consider consulting with financial advisors before making investment decisions.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
