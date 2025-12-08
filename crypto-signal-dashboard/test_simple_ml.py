# test_simple_ml.py
from advanced_ml_predictorsimple4 import RealisticMLPredictor

# Import your existing modules
try:
    from c_signal import fetch_price_data, calculate_technical_indicators
except ImportError:
    print("❌ Could not import c_signal module")
    # Add fallback functions
    def fetch_price_data(*args, **kwargs):
        print("⚠️  Using mock price data for testing")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='4h')
        data = {
            'close': np.random.normal(2800, 100, 200).cumsum(),
            'volume': np.random.lognormal(10, 1, 200),
            'high': np.random.normal(2850, 100, 200).cumsum(),
            'low': np.random.normal(2750, 100, 200).cumsum(),
        }
        return pd.DataFrame(data, index=dates)
    
    def calculate_technical_indicators(df):
        df['rsi'] = np.random.uniform(30, 70, len(df))
        df['macd'] = np.random.normal(0, 10, len(df))
        df['atr'] = np.random.uniform(20, 50, len(df))
        return df

def test_simple_ml():
    print("🧪 TESTING SIMPLE ML PREDICTOR")
    print("=" * 50)
    
    # Get price data
    print("📊 Fetching price data...")
    price_data = fetch_price_data('ETH/USDT', '4h', 200)
    price_data = calculate_technical_indicators(price_data)
    
    print(f"Data shape: {price_data.shape}")
    print(f"Current price: ${price_data['close'].iloc[-1]:,.2f}")
    
    # Create predictor
    predictor = RealisticMLPredictor(window=14, prediction_horizons=[1, 4, 24])
    
    # Train
    print("\n🤖 Training model...")
    success = predictor.train_simple_model(price_data)
    
    if success:
        print("\n🎯 Making predictions...")
        predictions, confidences = predictor.predict_simple(price_data)
        
        current_price = price_data['close'].iloc[-1]
        print(f"\n📊 FINAL RESULTS:")
        print(f"Current Price: ${current_price:,.2f}")
        print("-" * 50)
        
        timeframes = ['1H', '4H', '24H']
        for tf, pred, conf in zip(timeframes, predictions, confidences):
            change_pct = (pred - current_price) / current_price * 100
            print(f"{tf}: ${pred:,.2f} ({change_pct:+.2f}%) - {conf:.1%} confidence")
        
        # Test signal generation
        print("\n📈 Testing signal generation...")
        ml_signals, ml_boost = predictor.generate_ml_signals(predictions, confidences, current_price)
        print(f"ML Signals: {ml_signals}")
        print(f"ML Boost: {ml_boost}")
        
    else:
        print("❌ Training failed")

if __name__ == "__main__":
    test_simple_ml()
    print("\n" + "✅" * 20)
    print("✅ TEST COMPLETE")
    print("✅" * 20)