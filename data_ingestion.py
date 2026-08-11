import yfinance as yf
import pandas as pd
import numpy as np

def fetch_and_prepare_data(ticker: str = "AAPL", period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch financial data and generate technical indicator features."""
    df = yf.download(ticker, period=period, interval=interval)
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Core Target: Next-day percentage return and binary direction
    df['Target_Return'] = df['Close'].pct_change().shift(-1)
    df['Target_Class'] = (df['Target_Return'] > 0).astype(int)

    # Technical Indicators (Feature Engineering)
    # 1. Exponential Moving Averages & Ratios
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    
    # 2. Relative Strength Index (RSI - 14 period)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 3. Rolling Volatility & Returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility_14'] = df['Log_Return'].rolling(window=14).std()
    
    # Drop na rows caused by indicators and target shift
    df = df.dropna()
    return df

if __name__ == "__main__":
    data = fetch_and_prepare_data("AAPL")
    print(f"Dataset prepared successfully! Shape: {data.shape}")
    print(data[['Close', 'MACD', 'RSI_14', 'Volatility_14', 'Target_Class']].tail())