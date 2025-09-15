
# 🤖 BotTradingFutures

> *A professional futures trading bot with advanced technical analysis, economic data integration, ML signals, and risk management.*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![CCXT](https://img.shields.io/badge/CCXT-Library-green)
![Pandas](https://img.shields.io/badge/Pandas-Data--Analysis-yellow)
![TA-Lib](https://img.shields.io/badge/TA--Lib-Indicators-orange)
![License](https://img.shields.io/badge/License-MIT-red)

---

## 📌 Table of Contents
- [Description](#description)
- [Features](#features)
- [Requirements & Installation](#requirements--installation)
- [Configuration & API Keys](#configuration--api-keys)
- [Usage](#usage)
- [Commands](#commands)
- [Risk Management](#risk-management)
- [Notes](#notes)
- [License](#license)

---

## 📖 Description

**BotTradingFutures** is a fully customizable trading bot designed for **crypto futures markets**.  
It combines **technical analysis**, **economic data feeds**, **market sentiment**, and **risk management** to support professional-grade trading decisions.

---

## 🚀 Features

| Category | Features |
|----------|----------|
| **Technical Analysis** | EMA (20/50/200), MACD, Supertrend, BOS, Smart Money Index, Volume Profile, Bollinger Bands, Fibonacci, ATR |
| **Economic Data** | CPI, Fed Rate, Economic Calendar, Fear & Greed Index, News API integration |
| **Risk Management** | Portfolio risk, daily max risk, leverage control, correlation exposure |
| **Trading Modes** | Scalping, Day Trading, Swing Trading, Position Trading |
| **Safety** | Null-safe ops, error handling, retry, fallback for missing data |
| **Enhancements** | Optional ML enhancer, database logging, websocket support |
| **Reports** | Auto-generate detailed reports for each trade session |

---

## ⚙️ Requirements & Installation

### 📋 Prerequisites
- Python ≥ 3.8
- Internet access
- Exchange API key (if required)
- Optional: ML/DB/WebSocket modules

### 📦 Dependencies
```
ccxt
pandas
numpy
talib
requests
python-telegram-bot
pytz
certifi
urllib3
sqlite3 (built-in)
```

### 💻 Installation
```bash
git clone https://github.com/r00tH3x/BotTradingFutures.git
cd BotTradingFutures
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

If `requirements.txt` is missing:
```bash
pip install ccxt pandas numpy talib requests python-telegram-bot pytz certifi urllib3
```

---

## 🔑 Configuration & API Keys

- Configure your API keys in `.env` file:
  ```ini
  FRED_API_KEY=your_key
  NEWS_API_KEY=your_key
  ```

- Risk parameters (in `TradingConfig`):
  ```python
  max_portfolio_risk = 0.02
  max_daily_risk = 0.06
  min_signal_strength = 6
  min_mtf_confidence = 40
  ```

---

## ▶️ Usage

Run the bot:
```bash
python -m assistant.app
```

### Workflow
1. Configure API keys & trading parameters  
2. Start the bot → connects to exchange & data feeds  
3. Auto-scans market conditions → generates signals  
4. Executes trades based on strategy & risk rules  
5. Generates reports & sends alerts (Telegram, DB, etc.)  

---

## ⌨️ Commands

| Command | Description |
|---------|-------------|
| `/start_scanner` | Start auto scanning for signals |
| `/stop_scanner` | Stop auto scanning |
| `/update_config key value` | Update config dynamically |
| `/status` | Show current status |

---

## 🛡️ Risk Management

- **Max risk per trade**: 2% (default)  
- **Daily max risk**: 6%  
- **Leverage control**: Different per trading style (scalp/swing/etc.)  
- **Correlation check**: Prevent overexposure on correlated assets  

---

## ⚠️ Notes

- Use only with proper risk management  
- Test on **paper trading / demo accounts** first  
- Ensure compliance with exchange ToS  
- Optional modules (ML, DB, Websocket) require extra setup  

---

## 📜 License

This project is licensed under the MIT License.  
Feel free to modify and enhance responsibly.

---

💡 *Use responsibly and ethically — Hack the markets, but with permission!* ⚡
