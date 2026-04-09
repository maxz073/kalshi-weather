# Kalshi Weather Temperature Trading Bot

Automated trading system for KXHIGH (max temperature) contracts on Kalshi across 20 US cities. Polls every 15 minutes, evaluates entry conditions, and buys 100 YES contracts when criteria are met.

## Entry Conditions (all must be true)

- After 4 PM local time in the city
- YES ask price is between 92-96 cents
- Current humidity at the location < 60%
- Current cloud cover at the location < 60%

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Edit .env and set your KALSHI_API_KEY_ID

# 3. Add your Kalshi private key
# Place your kalshi_key.pem file in this directory
```

### .env Configuration

| Variable | Description | Default |
|---|---|---|
| `KALSHI_API_KEY_ID` | Your Kalshi API key ID | (required) |
| `KALSHI_PRIVATE_KEY_PATH` | Path to your PEM key file | `kalshi_key.pem` |
| `TRADING_MODE` | `paper` or `live` | `paper` |
| `BOT_HOST` | Address the bot's control server binds to | `0.0.0.0` |
| `BOT_PORT` | Port for the bot's control server | `8377` |

## Running

```bash
python main.py
```

On startup the bot will:
1. Validate your API credentials with a balance check
2. Print the current trading mode (paper/live)
3. Begin polling all 20 cities every 15 minutes

Logs are written to both stdout and `weather_trading.log`.

Stop the bot with `Ctrl+C`.

## Dashboard

A live terminal dashboard for monitoring positions, portfolio value, and price alerts. Run it on your work laptop while the bot trades on your home laptop.

```bash
# Same machine as the bot
python dashboard.py

# Remote — point to your home laptop's IP
python dashboard.py --bot-host 192.168.1.50

# Custom refresh interval
python dashboard.py --bot-host 192.168.1.50 --interval 10
```

The dashboard shows four panels:

- **Portfolio** — cash balance, portfolio payout, total value
- **Alerts** — warns when any open position's YES price drops below 60 cents
- **Open Positions** — ticker, side, quantity, entry price, current price, P&L, status
- **Today's Trades** — all trades from `trades.json` today with running totals

The header shows real-time bot status (ONLINE/OFFLINE).

### Keyboard Controls

| Key | Action |
|---|---|
| `K` | Kill the bot remotely (sends shutdown signal to `main.py` on your home laptop) |
| `Q` | Quit the dashboard |

### Network Setup

The bot (`main.py`) runs a lightweight control server on port 8377. The dashboard connects to it for status checks and the kill switch. Both machines need:

1. The same `.env` credentials (for Kalshi API access)
2. Network connectivity on port 8377 (both on the same Wi-Fi, or port forwarded)

If your home laptop's local IP is `192.168.1.50`, pass `--bot-host 192.168.1.50` to the dashboard. You can find your home laptop's IP with `ifconfig | grep inet`.

## Paper vs Live Trading

The bot starts in **paper mode** by default. Paper trades are logged to `trades.json` with simulated fills at the current ask price but no real orders are placed.

To switch to live trading after your paper run:

1. Edit `.env` and set `TRADING_MODE=live`
2. Restart the bot

No code changes are needed.

## Trade Limits

- Max **100 contracts per city per day**
- Only **YES side** orders
- **Market orders** (immediate fill)

## Trade Log

All trades (paper and live) are appended to `trades.json` with:

- Timestamp (UTC)
- City and series ticker
- Contract ticker
- Entry price and taker fee (in cents)
- Trading mode at time of execution

## Cities Monitored

New York, Chicago, Miami, Boston, Los Angeles, Austin, San Francisco, Dallas, Philadelphia, Phoenix, Oklahoma City, Denver, Washington DC, San Antonio, Houston, Minneapolis, Atlanta, Seattle, Las Vegas, New Orleans

## Weather Data

Current humidity and cloud cover are fetched from the [Open-Meteo Forecast API](https://api.open-meteo.com/v1/forecast) (free, no API key required).

## File Overview

| File | Purpose |
|---|---|
| `main.py` | Entry point and 15-minute polling loop |
| `config.py` | Credentials, strategy parameters, city definitions |
| `kalshi_client.py` | Synchronous Kalshi API client (RSA-PSS auth) |
| `weather_client.py` | Open-Meteo current weather client |
| `market_discovery.py` | Discover today's markets, find qualifying prices |
| `strategy.py` | Entry condition evaluation |
| `trader.py` | Paper/live trade execution and JSON logging |
| `dashboard.py` | Live terminal dashboard for monitoring |
