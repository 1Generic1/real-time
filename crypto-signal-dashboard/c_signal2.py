import pandas as pd
import ccxt
import time
import numpy as np
from datetime import datetime
import sys

# Initialize exchange
exchange = ccxt.binance()

# Consistent signal weights
SIGNAL_WEIGHTS = {
    # TREND SIGNALS
    'STRONG UPTREND': 3,
    'STRONG DOWNTREND': -3,
    'UPTREND': 2,
    'DOWNTREND': -2,
    'RANGING': 0,
    
    # RSI SIGNALS
    'OVERSOLD': 2,
    'OVERBOUGHT': -2,
    'BULLISH MOMENTUM': 1,
    'BEARISH MOMENTUM': -1,
    
    # MACD SIGNALS
    'BULLISH CROSSOVER': 2,
    'BEARISH CROSSOVER': -2,
    'BULLISH MOMENTUM': 1,
    'BEARISH MOMENTUM': -1,
    
    # BOLLINGER BANDS - ADD ALL VARIATIONS
    'AT UPPER BOLLINGER': -1,
    'AT LOWER BOLLINGER': 1,
    'UPPER BOLLINGER': -1,      # ADD THIS for "UPPER BOLLINGER RANGE"
    'LOWER BOLLINGER': 1,       # ADD THIS for "LOWER BOLLINGER RANGE"
    'MIDDLE BOLLINGER': 0,      # ADD THIS for "MIDDLE BOLLINGER RANGE"
    
    # VOLUME
    'HIGH VOLUME': 1,
    'LOW VOLUME': -1,
    
    # SUPPORT/RESISTANCE
    'SUPPORT': 1,
    'RESISTANCE': -1,
    'NEAR STRONG SUPPORT': 1,     # ADD THIS
    'NEAR STRONG RESISTANCE': -1, # ADD THIS
    
    # VOLATILITY
    'HIGH VOLATILITY': 0,
    'LOW VOLATILITY': 0,
}

def typewriter_effect(text, delay=0.03):
    """Simulate typewriter effect for futuristic feel"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def progress_bar(iteration, total, length=50):
    """Display a progress bar"""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '░' * (length - filled_length)
    print(f'\r│ {bar} │ {percent}%', end='', flush=True)
    if iteration == total:
        print()

def animated_loading(text, duration=2, frames=["⡿", "⣟", "⣯", "⣷", "⣾", "⣽", "⣻", "⢿"]):
    """Display animated loading"""
    end_time = time.time() + duration
    frame_index = 0
    while time.time() < end_time:
        print(f'\r{text} {frames[frame_index % len(frames)]}', end='', flush=True)
        frame_index += 1
        time.sleep(0.1)
    print(f'\r{text} ✅', flush=True)

def fetch_price_data(symbol='BTC/USDT', timeframe='1d', limit=200):
    """
    Fetch historical OHLCV data from Binance
    """
    try:
        typewriter_effect("🔗 CONNECTING...FETCHING DATA...", 0.02)
        time.sleep(0.5)
        
        animated_loading("📡 FETCHING MARKET DATA", 1.5)
        
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        # Convert ALL numeric columns to float
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = df[col].astype(float)
        
        print("✅ DATA ACQUISITION COMPLETE")
        return df
    except Exception as e:
        print(f"❌ ERROR FETCHING DATA: {e}")
        return None

def calculate_technical_indicators(df):
    """
    COMPREHENSIVE technical indicator calculation
    """
    typewriter_effect("\n🧮 COMPUTING TECHNICAL INDICATORS...", 0.02)
    
    indicators = [
        "Moving Averages", "RSI", "MACD", "Bollinger Bands", 
        "Volume Analysis", "Support/Resistance", "Volatility (ATR)"
    ]
    
    for i, indicator in enumerate(indicators):
        progress_bar(i + 1, len(indicators))
        time.sleep(0.3)
    
    # Moving Averages
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    df['ma_200'] = df['close'].rolling(window=200).mean()
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_histogram'] = calculate_macd(df['close'])
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'], 20)
    
    # Volume indicators
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Support and Resistance
    df['support'] = calculate_support_resistance(df, 'support')
    df['resistance'] = calculate_support_resistance(df, 'resistance')
    
    # ATR for volatility
    df['atr'] = calculate_atr(df, 14)
    
    print("✅ INDICATORS CALCULATED")
    return df

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    return upper_band, middle_band, lower_band

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = true_range.rolling(period).mean()
    return atr

def calculate_support_resistance(df, level_type='support', lookback=20):
    """Calculate dynamic support/resistance levels"""
    if level_type == 'support':
        return df['low'].rolling(window=lookback).min()
    else:
        return df['high'].rolling(window=lookback).max()

def generate_trading_signals(df):
    """
    Generate comprehensive trading signals
    """
    if len(df) < 50:
        return ["Insufficient data for analysis"]
    
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    signals = []
    
    # 1. TREND ANALYSIS
    price = current['close']
    ma_20 = current['ma_20']
    ma_50 = current['ma_50']
    ma_200 = current['ma_200']
    
    # Trend strength
    if price > ma_20 > ma_50 > ma_200:
        signals.append("📗 STRONG UPTREND (All MAs aligned)")
        trend_strength = "strong_bullish"
    elif price < ma_20 < ma_50 < ma_200:
        signals.append("📕 STRONG DOWNTREND (All MAs aligned)")
        trend_strength = "strong_bearish"
    elif price > ma_20 and ma_20 > ma_50:
        signals.append("📗 UPTREND (Price > 20MA > 50MA)")
        trend_strength = "bullish"
    elif price < ma_20 and ma_20 < ma_50:
        signals.append("📕 DOWNTREND (Price < 20MA < 50MA)")
        trend_strength = "bearish"
    else:
        signals.append("🟡 RANGING (Mixed MA signals)")
        trend_strength = "neutral"
    
    # 2. MOMENTUM ANALYSIS (RSI)
    rsi = current['rsi']
    if not pd.isna(rsi):
        if rsi < 30:
            signals.append("🟢 RSI: OVERSOLD (Potential bounce)")
        elif rsi > 70:
            signals.append("🔴 RSI: OVERBOUGHT (Potential pullback)")
        elif rsi > 50:
            signals.append("📗 RSI: BULLISH MOMENTUM (>50)")
        else:
            signals.append("📕 RSI: BEARISH MOMENTUM (<50)")
    
    # 3. MACD SIGNALS
    macd = current['macd']
    macd_signal = current['macd_signal']
    if not pd.isna(macd) and not pd.isna(macd_signal):
        if macd > macd_signal and previous['macd'] <= previous['macd_signal']:
            signals.append("🚀 MACD: BULLISH CROSSOVER")
        elif macd < macd_signal and previous['macd'] >= previous['macd_signal']:
            signals.append("🔻 MACD: BEARISH CROSSOVER")
        elif macd > macd_signal:
            signals.append("📗 MACD: BULLISH MOMENTUM")
        else:
            signals.append("📕 MACD: BEARISH MOMENTUM")
    
    # 4. BOLLINGER BANDS ANALYSIS
    if not pd.isna(current['bb_upper']) and not pd.isna(current['bb_lower']):
        if price >= current['bb_upper']:
            signals.append("🔴 PRICE: AT UPPER BOLLINGER BAND (Overextended)")
        elif price <= current['bb_lower']:
            signals.append("🟢 PRICE: AT LOWER BOLLINGER BAND (Oversold)")
        else:
            bb_position = (price - current['bb_lower']) / (current['bb_upper'] - current['bb_lower'])
            if bb_position > 0.7:
                signals.append("📗 PRICE: UPPER BOLLINGER RANGE")
            elif bb_position < 0.3:
                signals.append("📕 PRICE: LOWER BOLLINGER RANGE")
            else:
                signals.append("🟡 PRICE: MIDDLE BOLLINGER RANGE")
    
    # 5. VOLUME ANALYSIS
    volume_ratio = current['volume_ratio']
    if not pd.isna(volume_ratio):
        if volume_ratio > 1.5:
            signals.append("📈 HIGH VOLUME: Strong interest")
        elif volume_ratio < 0.7:
            signals.append("📉 LOW VOLUME: Weak interest")
    
    # 6. SUPPORT/RESISTANCE
    support = current['support']
    resistance = current['resistance']
    if not pd.isna(support) and not pd.isna(resistance):
        support_distance = ((price - support) / price) * 100
        resistance_distance = ((resistance - price) / price) * 100
        
        if support_distance < 2:
            signals.append("🛡️ NEAR STRONG SUPPORT (Potential bounce)")
        if resistance_distance < 2:
            signals.append("🚧 NEAR STRONG RESISTANCE (Potential rejection)")
    
    # 7. VOLATILITY ANALYSIS
    atr = current['atr']
    if not pd.isna(atr):
        atr_percentage = (atr / price) * 100
        if atr_percentage > 3:
            signals.append("⚡ HIGH VOLATILITY (Wide stops needed)")
        elif atr_percentage < 1:
            signals.append("🍃 LOW VOLATILITY (Tight consolidation)")
    
    return signals

def debug_signal_scoring(signals):
    """Debug function to show how signals are scored"""
    print("\n🔍 SIGNAL SCORING DEBUG:")
    print("-" * 50)
    
    for i, signal in enumerate(signals, 1):
        score = 0
        signal_type = "Unknown"
        
        # Check each signal type
        for sig_type, weight in SIGNAL_WEIGHTS.items():
            if sig_type in signal:
                score = weight
                signal_type = sig_type
                break
        
        # Fallback scoring
        if score == 0:
            if any(word in signal for word in ['BULLISH', 'UPTREND', 'OVERSOLD', 'SUPPORT']):
                score = 1
                signal_type = "Bullish (fallback)"
            elif any(word in signal for word in ['BEARISH', 'DOWNTREND', 'OVERBOUGHT', 'RESISTANCE']):
                score = -1
                signal_type = "Bearish (fallback)"
        
        emoji = "📗" if score > 0 else "📕" if score < 0 else "📊"
        print(f"{i:2d}. {emoji} {signal[:40]:40} | Score: {score:>+2} | Type: {signal_type}")
    
    print("-" * 50)

def get_overall_recommendation(signals):
    """
    Generate overall recommendation with CONSISTENT scoring
    """
    # Initialize scores
    total_score = 0
    signal_details = []
    
    for signal in signals:
        signal_score = 0
        signal_found = False
        
        # Check each signal type and assign weight
        for signal_type, weight in SIGNAL_WEIGHTS.items():
            if signal_type in signal:
                total_score += weight
                signal_score = weight
                signal_found = True
                break
        
        # If no match found, check for partial matches
        if not signal_found:
            # Default scoring based on emojis/keywords
            if any(word in signal for word in ['BULLISH', 'UPTREND', 'OVERSOLD', 'SUPPORT']):
                signal_score = 1
                total_score += 1
            elif any(word in signal for word in ['BEARISH', 'DOWNTREND', 'OVERBOUGHT', 'RESISTANCE']):
                signal_score = -1
                total_score -= 1
        
        signal_details.append((signal, signal_score))

def display_results(btc_data, signals, recommendation, confidence):
    """Display results in a futuristic, systematic way"""
    
    current = btc_data.iloc[-1]
    
    # Header with animation
    print("\n" + "="*70)
    typewriter_effect("🎯COMPREHENSIVE TECHNICAL ANALYSIS", 0.03)
    typewriter_effect(f"📅 TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0.02)
    print("="*70)
    
    time.sleep(0.5)
    
    # Price display with emphasis
    typewriter_effect(f"\n💰 CURRENT PRICE: ${current['close']:,.2f}", 0.01)
    print("─" * 40)
    
    # Technical Indicators section
    typewriter_effect("\n📊 TECHNICAL INDICATORS:", 0.02)
    time.sleep(0.3)
    
    indicators_data = [
        ("MA20", f"${current['ma_20']:,.2f}"),
        ("MA50", f"${current['ma_50']:,.2f}"),
        ("MA200", f"${current['ma_200']:,.2f}"),
        ("RSI", f"{current['rsi']:.1f}"),
        ("MACD", f"{current['macd']:.2f}"),
        ("Support", f"${current['support']:,.2f}"),
        ("Resistance", f"${current['resistance']:,.2f}")
    ]
    
    for name, value in indicators_data:
        print(f"   └─ {name}: {value}")
        time.sleep(0.1)
    
    # Signals section with dramatic reveal
    typewriter_effect("\n🎯 MARKET SIGNALS ANALYSIS:", 0.02)
    time.sleep(0.5)
    
    for i, signal in enumerate(signals, 1):
        print(f"   {i:2d}. {signal}")
        time.sleep(0.2)
    
    # Recommendation with build-up
    typewriter_effect("\n💡 FINAL ANALYSIS:", 0.03)
    time.sleep(0.5)
    typewriter_effect(f"   └─ RECOMMENDATION: {recommendation}", 0.01)
    typewriter_effect(f"   └─ SIGNAL CONFIDENCE: {confidence}", 0.01)
    
    # Key levels
    typewriter_effect("\n📈 KEY PRICE LEVELS:", 0.02)
    time.sleep(0.3)
    
    print(f"   └─ Current: ${current['close']:,.2f}")
    if not pd.isna(current['support']):
        support_pct = (current['close'] - current['support'])/current['close']*100
        print(f"   └─ Support: ${current['support']:,.2f} ({support_pct:.1f}% below)")
    if not pd.isna(current['resistance']):
        resistance_pct = (current['resistance'] - current['close'])/current['close']*100
        print(f"   └─ Resistance: ${current['resistance']:,.2f} ({resistance_pct:.1f}% above)")
    
    # Recent context
    typewriter_effect("\n📅 RECENT PRICE ACTION (Last 5 sessions):", 0.02)
    time.sleep(0.3)
    recent = btc_data[['close', 'volume', 'rsi']].tail().reset_index()
    print(recent.to_string(index=False))
    
    # Footer
    print("\n" + "="*70)
    typewriter_effect("✅ ANALYSIS COMPLETE - PICK YOUR TRADING POINTS", 0.03)
    print("="*70)

# Main execution
if __name__ == "__main__":
    print("🚀 INITIATING CRYPTO ANALYSIS ENGINE...")
    time.sleep(1)
    
    btc_data = fetch_price_data('BTC/USDT', '1d', 200)
    
    if btc_data is not None:
        btc_data = calculate_technical_indicators(btc_data)
        
        typewriter_effect("\n🔍 ANALYZING MARKET SIGNALS...", 0.02)
        time.sleep(1)
        
        signals = generate_trading_signals(btc_data)
        
        # Debug scoring
        debug_signal_scoring(signals)
        
        recommendation, confidence = get_overall_recommendation(signals)
        
        display_results(btc_data, signals, recommendation, confidence)
    else:
        print("❌ FAILED TO RETRIEVE DATA - PLEASE CHECK CONNECTION")