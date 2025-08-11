# Enhanced Stock Alert System - Flowchart Breakdown

## Overview
This flowchart breaks down how the Enhanced Stock Predictor works, from data collection to trading signals, including the new ensemble prediction methods and intraday data integration.

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
    F --> I[Get Intraday Data]
    F --> J[Get Real-time Sentiment]
    F --> K[Get Market Microstructure]
    
    %% Data Collection Subflows
    G --> G1[Method 1: Recent Data]
    G --> G2[Method 2: Stock Info]
    G --> G3[Method 3: Date Range]
    G1 --> G4[Return Current Price]
    G2 --> G4
    G3 --> G4
    
    H --> H1[Download Daily Stock Data]
    H1 --> H2[Add Technical Indicators]
    H2 --> H3[Calculate Moving Averages]
    H2 --> H4[Calculate RSI]
    H2 --> H5[Calculate MACD]
    H2 --> H6[Calculate Bollinger Bands]
    H2 --> H7[Calculate Volume Metrics]
    
    I --> I1[Download Intraday Data 5m]
    I1 --> I2[Add Intraday Indicators]
    I2 --> I3[Intraday Moving Averages]
    I2 --> I4[Intraday RSI]
    I2 --> I5[Volume Analysis]
    I2 --> I6[Time-based Features]
    
    J --> J1[Recent News Sentiment]
    J --> J2[Social Media Sentiment]
    J --> J3[Earnings Calendar Impact]
    J --> J4[Analyst Rating Changes]
    J --> J5[Options Flow Sentiment]
    
    K --> K1[Spread Analysis]
    K --> K2[Order Flow Indicators]
    K --> K3[Market Efficiency]
    K --> K4[Time-based Adjustments]
    
    %% Enhanced Prediction Flow
    G4 --> L[Enhanced Prediction Engine]
    H7 --> L
    I6 --> L
    J5 --> L
    K4 --> L
    
    L --> L1[Ensemble Prediction Models]
    L1 --> L2[Technical Analysis Model 40%]
    L1 --> L3[Sentiment-Based Model 25%]
    L1 --> L4[Microstructure Model 20%]
    L1 --> L5[Mean Reversion Model 15%]
    
    L2 --> L6[Intraday Momentum]
    L2 --> L7[Intraday RSI]
    L2 --> L8[Volume Trends]
    L2 --> L9[Volatility Analysis]
    
    L3 --> L10[News Sentiment]
    L3 --> L11[Social Media Sentiment]
    L3 --> L12[Earnings Impact]
    L3 --> L13[Analyst Changes]
    
    L4 --> L14[Spread Impact]
    L4 --> L15[Volume Impact]
    L4 --> L16[Market Efficiency]
    L4 --> L17[Time Adjustments]
    
    L5 --> L18[MA5 Reversion]
    L5 --> L19[MA10 Reversion]
    L5 --> L20[MA20 Reversion]
    
    L6 --> M[Weighted Ensemble]
    L7 --> M
    L8 --> M
    L9 --> M
    L10 --> M
    L11 --> M
    L12 --> M
    L13 --> M
    L14 --> M
    L15 --> M
    L16 --> M
    L17 --> M
    L18 --> M
    L19 --> M
    L20 --> M
    
    M --> M1[Confidence Adjustment]
    M1 --> M2[Final Predicted Price]
    
    %% Trading Signal Generation
    G4 --> N[Generate Trading Signal]
    M2 --> N
    
    N --> N1[Calculate Price Change %]
    N1 --> N2{Price Change > Buy Threshold?}
    N2 -->|Yes| N3[BUY Signal]
    N2 -->|No| N4{Price Change < Sell Threshold?}
    N4 -->|Yes| N5[SELL Signal]
    N4 -->|No| N6[HOLD Signal]
    
    N3 --> N7[Calculate Confidence]
    N5 --> N7
    N6 --> N7
    
    %% Alert Generation
    N7 --> O[Send Enhanced Alert]
    O --> O1[Get Market Analysis]
    O --> O2[Get Sentiment Analysis]
    O --> O3[Get Microstructure Analysis]
    O --> O4[Create Alert Message]
    
    O1 --> O5[Technical Indicators Summary]
    O2 --> O6[Sentiment Summary]
    O3 --> O7[Microstructure Summary]
    O4 --> O8[Format Complete Alert]
    
    O5 --> O8
    O6 --> O8
    O7 --> O8
    O8 --> O9[Save to File]
    O8 --> O10[Send to Slack]
    O8 --> O11[Print to Console]
    
    %% Styling
    classDef startEnd fill:#e1f5fe
    classDef dataCollection fill:#f3e5f5
    classDef calculation fill:#e8f5e8
    classDef decision fill:#fff3e0
    classDef output fill:#fce4ec
    classDef ensemble fill:#fff8e1
    
    class A startEnd
    class B,C,D,E dataCollection
    class G,H,I,J,K dataCollection
    class L,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13,L14,L15,L16,L17,L18,L19,L20 ensemble
    class M,M1,M2 calculation
    class N,N1,N2,N4,N7 decision
    class O,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,O11 output
```

## Enhanced Technical Indicators Breakdown

```mermaid
flowchart LR
    A[Stock Price Data] --> B[Daily Indicators]
    A --> C[Intraday Indicators]
    A --> D[Microstructure Features]
    
    B --> B1["MA5, MA10, MA20, MA50, MA200"]
    B --> B2["RSI 14-period"]
    B --> B3["MACD 12/26/9"]
    B --> B4["Bollinger Bands"]
    B --> B5["Volume Analysis"]
    
    C --> C1["Intraday MA3, MA5, MA10, MA20"]
    C --> C2["Intraday RSI 7-period"]
    C --> C3["5-minute Price Changes"]
    C --> C4["Intraday Volume Ratios"]
    C --> C5["High-Low Range Analysis"]
    C --> C6["Time-based Features"]
    
    D --> D1["Spread Proxies"]
    D --> D2["Order Flow Indicators"]
    D --> D3["Market Efficiency Ratios"]
    D --> D4["Volume-Price Impact"]
    D --> D5["Time-based Adjustments"]
    
    classDef indicator fill:#e3f2fd
    classDef calculation fill:#f1f8e9
    classDef result fill:#fff8e1
    classDef microstructure fill:#fce4ec
    
    class A indicator
    class B,C calculation
    class D microstructure
    class B1,B2,B3,B4,B5,C1,C2,C3,C4,C5,C6,D1,D2,D3,D4,D5 result
```

## Enhanced Sentiment Analysis Flow

```mermaid
flowchart TD
    A[Real-time Sentiment Collection] --> B[News Sentiment]
    A --> C[Social Media Sentiment]
    A --> D[Market Events]
    A --> E[Analyst Data]
    A --> F[Options Flow]
    
    B --> B1[Filter Recent News 4h]
    B1 --> B2[Analyze Article Titles]
    B2 --> B3[Calculate Weighted Sentiment]
    B3 --> B4[News Sentiment Score]
    
    C --> C1[Reddit Posts Analysis]
    C --> C2[Time-based Filtering]
    C2 --> C3[Score & Recency Weighting]
    C3 --> C4[Social Sentiment Score]
    
    D --> D1[Earnings Calendar]
    D1 --> D2[Days Until Earnings]
    D2 --> D3[Earnings Impact Score]
    
    E --> E1[Analyst Rating Changes]
    E1 --> E2[Target Price Updates]
    E2 --> E3[Analyst Impact Score]
    
    F --> F1[Options Flow Data]
    F1 --> F2[Call/Put Ratios]
    F2 --> F3[Options Sentiment Score]
    
    B4 --> G[Combine All Sentiment]
    C4 --> G
    D3 --> G
    E3 --> G
    F3 --> G
    G --> H[Final Sentiment Score]
    
    classDef source fill:#fce4ec
    classDef process fill:#e8f5e8
    classDef result fill:#fff3e0
    
    class A,B,C,D,E,F source
    class B1,B2,B3,B4,C1,C2,C3,C4,D1,D2,D3,E1,E2,E3,F1,F2,F3 process
    class G,H result
```

## Ensemble Prediction Algorithm Breakdown

```mermaid
flowchart TD
    A[Multiple Data Sources] --> B[Technical Analysis Model]
    A --> C[Sentiment-Based Model]
    A --> D[Microstructure Model]
    A --> E[Mean Reversion Model]
    
    B --> B1[Intraday Momentum]
    B1 --> B2[Intraday RSI Analysis]
    B2 --> B3[Volume Trend Analysis]
    B3 --> B4[Volatility Adjustment]
    B4 --> B5[Technical Prediction]
    
    C --> C1[News Sentiment 30%]
    C --> C2[Social Media 25%]
    C --> C3[Earnings Impact 20%]
    C --> C4[Analyst Changes 15%]
    C --> C5[Options Flow 10%]
    C5 --> C6[Sentiment Prediction]
    
    D --> D1[Spread Impact Analysis]
    D1 --> D2[Volume Impact Analysis]
    D2 --> D3[Market Efficiency]
    D3 --> D4[Time-based Adjustments]
    D4 --> D5[Microstructure Prediction]
    
    E --> E1[MA5 Reversion 50%]
    E --> E2[MA10 Reversion 30%]
    E --> E3[MA20 Reversion 20%]
    E3 --> E4[Mean Reversion Prediction]
    
    B5 --> F[Ensemble Weighting]
    C6 --> F
    D5 --> F
    E4 --> F
    
    F --> F1[Technical: 40% Weight]
    F --> F2[Sentiment: 25% Weight]
    F --> F3[Microstructure: 20% Weight]
    F --> F4[Mean Reversion: 15% Weight]
    
    F1 --> G[Weighted Average]
    F2 --> G
    F3 --> G
    F4 --> G
    
    G --> H[Confidence Adjustment]
    H --> I[Prediction Consistency Check]
    I --> J[Final Ensemble Prediction]
    
    classDef input fill:#e1f5fe
    classDef model fill:#f3e5f5
    classDef calculation fill:#e8f5e8
    classDef ensemble fill:#fff8e1
    classDef result fill:#fce4ec
    
    class A input
    class B,C,D,E model
    class B1,B2,B3,B4,B5,C1,C2,C3,C4,C5,C6,D1,D2,D3,D4,D5,E1,E2,E3,E4 calculation
    class F,F1,F2,F3,F4,G,H,I ensemble
    class J result
```

## Machine Learning Ensemble Architecture

```mermaid
flowchart TD
    A[Feature Engineering] --> B[Data Preparation]
    B --> C[Model Training]
    
    C --> D[Random Forest]
    C --> E[XGBoost]
    C --> F[Gradient Boosting]
    C --> G[Support Vector Regression]
    C --> H[Linear Regression]
    
    D --> I[RF Predictions]
    E --> J[XGB Predictions]
    F --> K[GB Predictions]
    G --> L[SVR Predictions]
    H --> M[LR Predictions]
    
    I --> N[Performance Evaluation]
    J --> N
    K --> N
    L --> N
    M --> N
    
    N --> O[Calculate R² Scores]
    O --> P[Weight by Performance]
    P --> Q[Ensemble Prediction]
    
    classDef data fill:#e1f5fe
    classDef model fill:#f3e5f5
    classDef prediction fill:#e8f5e8
    classDef ensemble fill:#fff8e1
    
    class A,B data
    class C,D,E,F,G,H model
    class I,J,K,L,M prediction
    class N,O,P,Q ensemble
```

## Enhanced Configuration Parameters

```mermaid
mindmap
  root((Enhanced Config.json))
    Prediction Engine
      use_enhanced_prediction
      ensemble_weights
        technical_analysis
        sentiment_based
        microstructure
        mean_reversion
      prediction_bounds
        max_daily_change
        max_intraday_change
        confidence_dampening
    Data Sources
      intraday_interval
      intraday_days
      lookback_days
      prediction_hours
      min_required_days
      min_training_samples
    Real-time Sentiment
      enable_news_sentiment
      enable_social_sentiment
      enable_earnings_impact
      enable_analyst_impact
      enable_options_flow
      sentiment_hours
    Microstructure Features
      enable_spread_analysis
      enable_volume_analysis
      enable_market_efficiency
      enable_time_based_adjustments
    Trading Parameters
      ticker
      buy_threshold
      sell_threshold
      confidence_threshold
      alert_time
    Model Parameters
      training_epochs
      batch_size
      learning_rate
      min_prediction_days
    System Parameters
      log_level
      save_alerts_to_file
      alert_file
      log_file
```

## Enhanced Impact Breakdown Analysis

### **Ensemble Model Weights & Impact**

| Model | Weight | Max Impact | Key Features | Description |
|-------|--------|------------|--------------|-------------|
| **Technical Analysis** | 🔴 **40%** | ±2% | Intraday momentum, RSI, volume trends | `intraday_momentum * 0.4 + rsi_factor * 0.3 + volume_factor * 0.2 + volatility_factor * 0.1` |
| **Sentiment-Based** | 🟡 **25%** | ±1% | News, social media, earnings, analyst ratings | `news_sentiment * 0.3 + social_sentiment * 0.25 + earnings_impact * 0.2 + analyst_impact * 0.15 + options_sentiment * 0.1` |
| **Microstructure** | 🟡 **20%** | ±1.5% | Spread analysis, order flow, market efficiency | `spread_impact * 0.1 + volume_impact * 0.05 + efficiency_impact * 0.02` |
| **Mean Reversion** | 🟢 **15%** | ±1% | MA5, MA10, MA20 reversion | `weighted_reversion * 0.3` |

### **Intraday Data Impact**

#### **5-Minute Interval Analysis**
- **Data Granularity**: 288 data points per day vs 1 daily point
- **Pattern Recognition**: Intraday momentum, volume spikes, price gaps
- **Time-based Features**: Market hours, end-of-day effects, lunch hour patterns
- **Technical Indicators**: Shorter RSI (7-period), intraday moving averages

#### **Market Microstructure Features**
- **Spread Proxies**: High-low range analysis for bid-ask spread estimation
- **Order Flow**: Volume trends, price impact of volume changes
- **Market Efficiency**: Price change volatility, mean reversion tendencies
- **Time Adjustments**: End-of-day effects, market hour adjustments

### **Real-time Sentiment Sources**

```mermaid
pie title Real-time Sentiment Distribution
    "News Sentiment" : 30
    "Social Media" : 25
    "Earnings Impact" : 20
    "Analyst Changes" : 15
    "Options Flow" : 10
```

### **Enhanced Prediction Formula**

```python
# Ensemble Prediction
ensemble_prediction = (
    technical_prediction * 0.4 +
    sentiment_prediction * 0.25 +
    microstructure_prediction * 0.2 +
    mean_reversion_prediction * 0.15
)

# Technical Analysis Components
technical_prediction = (
    intraday_momentum * 0.4 +
    rsi_factor * 0.3 +
    volume_factor * 0.2 +
    volatility_factor * 0.1
)

# Sentiment Components
sentiment_prediction = (
    news_sentiment * 0.3 +
    social_sentiment * 0.25 +
    earnings_impact * 0.2 +
    analyst_impact * 0.15 +
    options_sentiment * 0.1
)

# Confidence Adjustment
if prediction_std > price_std:
    confidence_factor = price_std / prediction_std
    final_prediction = current_price + (ensemble_prediction - current_price) * confidence_factor
```

## Key Enhanced Features Summary

### **Data Sources**
- **Intraday Data**: 5-minute intervals for granular analysis
- **Market Microstructure**: Spread analysis, order flow, efficiency metrics
- **Real-time Sentiment**: 4-hour filtered news, social media, earnings impact
- **Technical Indicators**: Enhanced with intraday-specific calculations

### **Prediction Methods**
- **Ensemble Approach**: 4 specialized models with weighted combination
- **Machine Learning**: Random Forest, XGBoost, SVR, Gradient Boosting
- **Confidence Adjustment**: Reduces prediction magnitude when models disagree
- **Fallback Mechanisms**: Multiple prediction methods for reliability

### **Advanced Features**
- **Time-based Adjustments**: Market hours, end-of-day effects
- **Volume Analysis**: Intraday volume trends, price impact
- **Sentiment Integration**: Multi-source real-time sentiment analysis
- **Risk Management**: Conservative bounds, confidence scoring

### **Configuration Options**
- **Flexible Weights**: Adjustable ensemble model weights
- **Feature Toggles**: Enable/disable specific analysis components
- **Prediction Bounds**: Configurable maximum change limits
- **Real-time Options**: Control sentiment analysis sources

### **Performance Improvements**
- **Higher Accuracy**: Ensemble methods reduce prediction variance
- **Better Granularity**: Intraday data captures short-term patterns
- **Real-time Updates**: 4-hour sentiment filtering for relevance
- **Robust Fallbacks**: Multiple prediction methods ensure reliability

### **Monitoring & Logging**
- **Detailed Analysis**: Comprehensive technical and sentiment breakdowns
- **Performance Tracking**: Model accuracy and ensemble weights
- **Error Handling**: Graceful degradation with fallback methods
- **Alert System**: Enhanced notifications with detailed analysis 