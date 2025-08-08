# Enhanced Stock Alert System - Flowchart Breakdown

## Overview
This flowchart breaks down how the Enhanced Stock Predictor works, from data collection to trading signals.

```mermaid
flowchart TD
    A[Start: Enhanced Stock Predictor] --> B[Load Configuration]
    B --> C[Initialize Components]
    C --> D[Setup Logging]
    D --> E[Initialize Reddit Client]
    
    %% Main Analysis Flow
    E --> F[Run Daily Analysis]
    F --> G[Get Current Price]
    F --> H[Get Historical Data]
    F --> I[Get Sentiment Data]
    
    %% Data Collection Subflows
    G --> G1[Method 1: Recent Data]
    G --> G2[Method 2: Stock Info]
    G --> G3[Method 3: Date Range]
    G1 --> G4[Return Current Price]
    G2 --> G4
    G3 --> G4
    
    H --> H1[Download Stock Data]
    H1 --> H2[Add Technical Indicators]
    H2 --> H3[Calculate Moving Averages]
    H2 --> H4[Calculate RSI]
    H2 --> H5[Calculate MACD]
    H2 --> H6[Calculate Bollinger Bands]
    H2 --> H7[Calculate Volume Metrics]
    
    I --> I1[Reddit Sentiment]
    I --> I2[News Sentiment]
    I --> I3[Yahoo Finance Data]
    I1 --> I4[Analyze Posts & Comments]
    I2 --> I5[Analyze News Articles]
    I3 --> I6[Get Market Cap & Analyst Ratings]
    
    %% Prediction Flow
    G4 --> J[Predict Future Price]
    H7 --> J
    I4 --> J
    I5 --> J
    I6 --> J
    
    J --> J1[Calculate Technical Factors]
    J1 --> J2[Trend Factor: MA5 vs MA20]
    J1 --> J3[Momentum Factor: Daily Returns]
    J1 --> J4[Volatility Factor: Price Changes]
    J1 --> J5[RSI Factor: Overbought/Oversold]
    
    J2 --> J6[Combine All Factors]
    J3 --> J6
    J4 --> J6
    J5 --> J6
    
    J6 --> J7[Apply Conservative Bounds ±2%]
    J7 --> J8[Calculate Predicted Price]
    
    %% Trading Signal Generation
    G4 --> K[Generate Trading Signal]
    J8 --> K
    
    K --> K1[Calculate Price Change %]
    K1 --> K2{Price Change > Buy Threshold?}
    K2 -->|Yes| K3[BUY Signal]
    K2 -->|No| K4{Price Change < Sell Threshold?}
    K4 -->|Yes| K5[SELL Signal]
    K4 -->|No| K6[HOLD Signal]
    
    K3 --> K7[Calculate Confidence]
    K5 --> K7
    K6 --> K7
    
    %% Alert Generation
    K7 --> L[Send Alert]
    L --> L1[Get Market Analysis]
    L --> L2[Get Sentiment Analysis]
    L --> L3[Create Alert Message]
    
    L1 --> L4[Technical Indicators Summary]
    L2 --> L5[Sentiment Summary]
    L3 --> L6[Format Complete Alert]
    
    L4 --> L6
    L5 --> L6
    L6 --> L7[Save to File]
    L6 --> L8[Send to Slack]
    L6 --> L9[Print to Console]
    
    %% Styling
    classDef startEnd fill:#e1f5fe
    classDef dataCollection fill:#f3e5f5
    classDef calculation fill:#e8f5e8
    classDef decision fill:#fff3e0
    classDef output fill:#fce4ec
    
    class A startEnd
    class B,C,D,E dataCollection
    class G,H,I dataCollection
    class J,J1,J2,J3,J4,J5,J6,J7,J8 calculation
    class K,K1,K2,K4,K7 decision
    class L,L1,L2,L3,L4,L5,L6,L7,L8,L9 output
```

## Technical Indicators Breakdown

```mermaid
flowchart LR
    A[Stock Price Data] --> B[Moving Averages]
    A --> C[RSI Calculation]
    A --> D[MACD Calculation]
    A --> E[Bollinger Bands]
    A --> F[Volume Analysis]
    
    B --> B1["MA5: 5-day average"]
    B --> B2["MA10: 10-day average"]
    B --> B3["MA20: 20-day average"]
    B --> B4["MA50: 50-day average"]
    B --> B5["MA200: 200-day average"]
    
    C --> C1[Calculate gains & losses]
    C1 --> C2["14-day average gain"]
    C1 --> C3["14-day average loss"]
    C2 --> C4["RS = Avg Gain / Avg Loss"]
    C3 --> C4
    C4 --> C5["RSI = 100 - 100/(1+RS)"]
    
    D --> D1["12-day EMA"]
    D --> D2["26-day EMA"]
    D1 --> D3["MACD = EMA12 - EMA26"]
    D2 --> D3
    D3 --> D4["Signal = 9-day EMA of MACD"]
    D4 --> D5["Histogram = MACD - Signal"]
    
    E --> E1["20-day SMA"]
    E1 --> E2[Standard Deviation]
    E2 --> E3["Upper Band = SMA + 2*SD"]
    E2 --> E4["Lower Band = SMA - 2*SD"]
    E3 --> E5["BB Position = Price - Lower / Upper - Lower"]
    E4 --> E5
    
    F --> F1["Volume MA: 20-day average"]
    F1 --> F2["Volume Ratio = Current / Average"]
    F2 --> F3[Volume Price Trend]
    
    classDef indicator fill:#e3f2fd
    classDef calculation fill:#f1f8e9
    classDef result fill:#fff8e1
    
    class A indicator
    class B,C,D,E,F calculation
    class B1,B2,B3,B4,B5,C5,D3,D4,D5,E5,F2,F3 result
```

## Sentiment Analysis Flow

```mermaid
flowchart TD
    A[Sentiment Data Collection] --> B[Reddit Sentiment]
    A --> C[News Sentiment]
    A --> D[Market Data]
    
    B --> B1[Search Reddit Posts]
    B1 --> B2[Filter by Ticker Mention]
    B2 --> B3[Analyze Title Sentiment]
    B2 --> B4[Analyze Body Sentiment]
    B3 --> B5["Weighted Sentiment: 70% Title + 30% Body"]
    B4 --> B5
    B5 --> B6[Aggregate Daily Sentiment]
    
    C --> C1[Fetch Yahoo Finance News]
    C1 --> C2[Analyze Article Titles]
    C2 --> C3[Calculate Sentiment Polarity]
    C3 --> C4[Aggregate Daily News Sentiment]
    
    D --> D1[Get Market Cap]
    D --> D2[Get Sector Info]
    D --> D3[Get Analyst Ratings]
    D1 --> D4[Market Context]
    D2 --> D4
    D3 --> D4
    
    B6 --> E[Combine All Sentiment Data]
    C4 --> E
    D4 --> E
    E --> F[Final Sentiment Score]
    
    classDef source fill:#fce4ec
    classDef process fill:#e8f5e8
    classDef result fill:#fff3e0
    
    class A,B,C,D source
    class B1,B2,B3,B4,B5,B6,C1,C2,C3,C4,D1,D2,D3,D4 process
    class E,F result
```

## Prediction Algorithm Breakdown

```mermaid
flowchart TD
    A[Historical Price Data] --> B[Calculate Recent Trends]
    A --> C[Calculate Momentum]
    A --> D[Calculate Volatility]
    A --> E[Calculate RSI Factor]
    A --> F[Get Sentiment Data]
    
    B --> B1[MA5 vs MA20 Comparison]
    B1 --> B2["Trend Factor = MA5-MA20/MA20 * 0.1"]
    
    C --> C1[Daily Returns Calculation]
    C1 --> C2[Average Daily Return]
    C2 --> C3["Momentum Factor = Avg Return * 2"]
    
    D --> D1[Standard Deviation of Returns]
    D1 --> D2["Volatility Factor = SD * 0.05"]
    
    E --> E1[Gains vs Losses Analysis]
    E1 --> E2["RSI Factor = Normalized RSI * 0.02"]
    
    F --> F1[Reddit Sentiment Analysis]
    F --> F2[News Sentiment Analysis]
    F1 --> F3["Reddit Impact = Polarity * 0.05"]
    F2 --> F4["News Impact = Polarity * 0.03"]
    
    B2 --> G[Combine All Factors]
    C3 --> G
    D2 --> G
    E2 --> G
    F3 --> G
    F4 --> G
    
    G --> H[Apply Conservative Bounds ±2%]
    H --> I[Additional Dampening if >1%]
    I --> J[Final Predicted Price]
    
    classDef input fill:#e1f5fe
    classDef calculation fill:#f3e5f5
    classDef factor fill:#e8f5e8
    classDef sentiment fill:#fce4ec
    classDef result fill:#fff3e0
    
    class A input
    class B,C,D,E calculation
    class F sentiment
    class B2,C3,D2,E2 factor
    class F3,F4 sentiment
    class G,H,I,J result
```

## Trading Signal Logic

```mermaid
flowchart TD
    A[Current Price] --> B[Predicted Price]
    A --> C[Calculate Price Change %]
    B --> C
    
    C --> D{Price Change > Buy Threshold?}
    D -->|Yes| E[BUY Signal]
    D -->|No| F{Price Change < Sell Threshold?}
    F -->|Yes| G[SELL Signal]
    F -->|No| H[HOLD Signal]
    
    E --> I[Calculate Confidence]
    G --> I
    H --> I
    
    I --> J{Confidence > Threshold?}
    J -->|Yes| K[High Confidence Signal]
    J -->|No| L[Low Confidence Signal]
    
    K --> M[Send Detailed Alert]
    L --> M
    
    M --> N[Include Technical Analysis]
    M --> O[Include Sentiment Analysis]
    M --> P[Include Market Context]
    
    N --> Q[Final Trading Recommendation]
    O --> Q
    P --> Q
    
    classDef price fill:#e1f5fe
    classDef decision fill:#fff3e0
    classDef signal fill:#e8f5e8
    classDef analysis fill:#f3e5f5
    classDef output fill:#fce4ec
    
    class A,B,C price
    class D,F,J decision
    class E,G,H signal
    class I,K,L analysis
    class M,N,O,P,Q output
```

## Configuration Parameters

```mermaid
mindmap
  root((Config.json))
    Trading Parameters
      ticker
      buy_threshold
      sell_threshold
      confidence_threshold
      alert_time
    Data Parameters
      lookback_days
      prediction_hours
      min_required_days
      min_training_samples
      extra_days_for_safety
    Model Parameters
      training_epochs
      batch_size
      learning_rate
      min_prediction_days
    Sentiment Parameters
      enable_sentiment_analysis
      sentiment_analysis_limit
      news_article_limit
    Technical Parameters
      enable_technical_indicators
      recent_days_for_prediction
      min_analysis_days
    System Parameters
      log_level
      save_alerts_to_file
      alert_file
      log_file
```

## Impact Breakdown Analysis

### **Prediction Factor Weights**

| Factor | Impact Level | Weight | Max Impact | Description |
|--------|-------------|---------|------------|-------------|
| **Moving Averages** | 🔴 **HIGH** | ~35% | ±2% | `trend_factor = ((ma_5 - ma_20) / ma_20) * 0.1` |
| **Momentum** | 🔴 **HIGH** | ~35% | ±2% | `momentum_factor = avg_daily_return * 2` |
| **Reddit Sentiment** | 🟡 **MEDIUM** | ~15% | ±5% | `reddit_impact = polarity * 0.05` |
| **News Sentiment** | 🟡 **MEDIUM** | ~10% | ±3% | `news_impact = polarity * 0.03` |
| **Volatility** | 🟢 **LOW** | ~3% | ±1% | `volatility_factor = volatility * 0.05` |
| **RSI** | 🟢 **LOW** | ~2% | ±0.5% | `rsi_adjustment = (rsi_factor - 0.5) * 0.02` |

### **Sentiment Impact Details**

#### **Reddit Sentiment (15% weight)**
- **Source**: r/stocks, r/investing
- **Calculation**: `reddit_impact = avg_polarity * 0.05`
- **Max Impact**: ±5% price movement
- **Processing**: 70% title weight + 30% body weight
- **Filtering**: Only posts mentioning ticker symbol

#### **News Sentiment (10% weight)**
- **Source**: Yahoo Finance news articles
- **Calculation**: `news_impact = avg_polarity * 0.03`
- **Max Impact**: ±3% price movement
- **Processing**: Title sentiment analysis only
- **Limit**: 50 articles per analysis

### **Technical vs Sentiment Impact**

```mermaid
pie title Prediction Factor Distribution
    "Moving Averages" : 35
    "Momentum" : 35
    "Reddit Sentiment" : 15
    "News Sentiment" : 10
    "Volatility" : 3
    "RSI" : 2
```

### **Combined Prediction Formula**

```python
prediction_change = (
    trend_factor +           # Moving averages (35% weight)
    momentum_factor +        # Daily returns (35% weight)  
    reddit_impact +          # Reddit sentiment (15% weight)
    news_impact +            # News sentiment (10% weight)
    volatility_factor +      # Price volatility (3% weight)
    rsi_adjustment          # RSI factor (2% weight)
)

# Apply conservative bounds
prediction_change = max(-0.02, min(0.02, prediction_change))
```

## Key Features Summary

- **Multi-Source Data**: Combines stock prices, Reddit sentiment, news sentiment, and market data
- **Technical Analysis**: Calculates 15+ technical indicators including RSI, MACD, Bollinger Bands
- **Sentiment Integration**: 25% total weight for sentiment analysis (15% Reddit + 10% News)
- **Conservative Predictions**: Uses dampened factors and bounds to avoid extreme predictions
- **Configurable**: All parameters can be adjusted via config.json
- **Robust Error Handling**: Multiple fallback methods for data collection
- **Comprehensive Alerts**: Includes technical analysis, sentiment analysis, and market context
- **Real-time Capable**: Can run continuously or on-demand
- **Slack Integration**: Sends alerts to Slack channels
- **Logging**: Comprehensive logging for debugging and monitoring 