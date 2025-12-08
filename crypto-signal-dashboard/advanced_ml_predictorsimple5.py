# advanced_ml_predictorsimple4.py - BEST VERSION WITH PERFECT CONSISTENCY

# ============ SUPPRESS ALL WARNINGS ============
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import random
import numpy as np

# Set seeds BEFORE importing tensorflow
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# NOW import tensorflow (after environment variables are set)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set tensorflow seed
tf.random.set_seed(SEED)

# Other imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

print(f"🤖 ML PREDICTOR INITIALIZED WITH SEED: {SEED} FOR PERFECT CONSISTENCY")
# ============ END OF ADDITION ============

class RealisticMLPredictor:
    def __init__(self, window=14, prediction_horizons=[1, 4, 24]):
        self.window = window
        self.prediction_horizons = prediction_horizons
        self.price_scalers = {}
        self.models = {}
        self.is_trained = False
        self.feature_columns = []
        self.recent_volatility = 0.02  # Default 2% volatility
        self.last_training_time = None  # Add this to track training time
        
    def extract_real_price_features(self, price_data):
        """Use ONLY real price data available from exchanges"""
        df = price_data.copy()
        
        # Start with basic real features
        real_features = []
        
        # 1. Price and volume (guaranteed real from exchange API)
        if 'close' in df.columns:
            real_features.append('close')
        if 'volume' in df.columns:
            real_features.append('volume')
        if 'high' in df.columns:
            real_features.append('high')
        if 'low' in df.columns:
            real_features.append('low')
        
        # 2. Returns (calculated from real price)
        for periods in [1, 3, 5, 10]:
            col_name = f'returns_{periods}'
            df[col_name] = df['close'].pct_change(periods)
            real_features.append(col_name)
        
        # 3. Volume indicators
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10'].replace(0, 1)
        real_features.extend(['volume_sma_10', 'volume_ratio'])
        
        # 4. Price range indicators
        if 'high' in df.columns and 'low' in df.columns:
            df['price_range'] = (df['high'] - df['low']) / df['close'].replace(0, 1)
            high_low_diff = (df['high'] - df['low']).replace(0, 1)
            df['close_position'] = (df['close'] - df['low']) / high_low_diff
            real_features.extend(['price_range', 'close_position'])
        
        # 5. Simple moving averages (from real price)
        for period in [5, 10, 20]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            sma = df[f'sma_{period}'].replace(0, 1)
            df[f'price_sma_ratio_{period}'] = df['close'] / sma
            real_features.extend([f'sma_{period}', f'price_sma_ratio_{period}'])
        
        # 6. Volatility (from real price) - store for dynamic thresholds
        df['volatility_5'] = df['close'].pct_change().rolling(5).std()
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()
        
        # Update recent volatility for signal generation
        if len(df) > 10:
            self.recent_volatility = df['volatility_20'].iloc[-10:].mean()
            if pd.isna(self.recent_volatility) or self.recent_volatility == 0:
                self.recent_volatility = 0.02
        
        real_features.extend(['volatility_5', 'volatility_20'])
        
        # 7. Technical indicators if they exist
        if 'rsi' in df.columns:
            real_features.append('rsi')
        if 'macd' in df.columns:
            real_features.append('macd')
        if 'atr' in df.columns:
            real_features.append('atr')
        
        # Fill NaN values carefully
        df = df[real_features].ffill().bfill()
        df = df.fillna(0)
        
        # Remove any infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        print(f"📊 Using {len(real_features)} REAL price features")
        return df
    
    def create_simple_sequences(self, data_scaled, horizon):
        """Create sequences for prediction"""
        X, y = [], []
        n_samples = len(data_scaled)
        
        for i in range(self.window, n_samples - horizon):
            X.append(data_scaled[i-self.window:i])
            # Predict the close price at horizon
            if i + horizon - 1 < n_samples:
                y.append(data_scaled[i + horizon - 1, 0])  # Index 0 should be 'close'
        
        if len(X) == 0 and n_samples >= self.window:
            # Create at least one sequence
            X.append(data_scaled[-self.window:])
            future_idx = min(n_samples - 1, self.window + horizon - 1)
            y.append(data_scaled[future_idx, 0])
        
        return np.array(X), np.array(y)
    
    def train_simple_model(self, price_data):
        """Train a simple but effective model on REAL price data only"""
        print("🤖 TRAINING SIMPLE PRICE-BASED ML MODEL...")
        self.last_training_time = pd.Timestamp.now()
        
        try:
            # EXTRA CONSISTENCY: Set seeds again for this training session
            np.random.seed(SEED)
            tf.random.set_seed(SEED)
            
            # Use only real price features
            feature_df = self.extract_real_price_features(price_data)
            
            # Store feature names for prediction time
            self.feature_columns = feature_df.columns.tolist()
            
            for horizon in self.prediction_horizons:
                print(f"   📈 Training {horizon}H model...")
                
                # Create scaler for this horizon
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(feature_df)
                self.price_scalers[horizon] = scaler
                
                # Create sequences
                X, y = self.create_simple_sequences(data_scaled, horizon)
                
                if len(X) < 20:  # Need minimum samples
                    print(f"   ⚠️  Need more data for {horizon}H (have {len(X)} sequences)")
                    continue
                
                # Split data - use deterministic split
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                # Simple LSTM model with deterministic settings
                model = Sequential([
                    LSTM(16, input_shape=(X.shape[1], X.shape[2]), dropout=0.1, 
                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                    Dense(8, activation='relu', 
                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                    Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED))
                ])
                
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='mse'
                )
                
                # Simple training with fixed batch order
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=30,
                    batch_size=8,
                    verbose=0,
                    shuffle=False,  # IMPORTANT: Don't shuffle for consistency
                    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
                )
                
                # Evaluate
                y_pred = model.predict(X_val, verbose=0).flatten()
                
                # Calculate error in original price scale
                price_scaler = scaler.scale_[0] if scaler.scale_[0] != 0 else 1
                price_mean = scaler.mean_[0]
                
                y_val_original = y_val * price_scaler + price_mean
                y_pred_original = y_pred * price_scaler + price_mean
                
                mae = mean_absolute_error(y_val_original, y_pred_original)
                mape = np.mean(np.abs((y_val_original - y_pred_original) / y_val_original)) * 100
                
                print(f"   ✅ {horizon}H: MAE=${mae:.2f} ({mape:.1f}% error)")
                
                # Store model
                self.models[horizon] = model
            
            self.is_trained = True
            print("✅ SIMPLE ML MODEL TRAINING COMPLETE")
            return True
            
        except Exception as e:
            print(f"❌ ML Training Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_simple(self, price_data):
        """Make realistic predictions WITH CONSISTENT CONFIDENCE"""
        try:
            if not self.is_trained or len(self.models) == 0:
                current_price = price_data['close'].iloc[-1]
                # Return consistent predictions when not trained
                pred_1h = current_price * 0.99  # -1% consistent
                pred_4h = current_price * 0.96  # -4% consistent  
                pred_24h = current_price * 0.95  # -5% consistent
                return [pred_1h, pred_4h, pred_24h], [0.5, 0.45, 0.4]
            
            # Prepare features
            feature_df = self.extract_real_price_features(price_data)
            
            # Ensure we have the same features as training
            missing_cols = set(self.feature_columns) - set(feature_df.columns)
            for col in missing_cols:
                feature_df[col] = 0
            
            # Reorder columns to match training
            feature_df = feature_df[self.feature_columns]
            
            predictions = []
            confidences = []
            current_price = price_data['close'].iloc[-1]
            
            for horizon in self.prediction_horizons:
                if horizon not in self.price_scalers:
                    # Simple fallback - use consistent values
                    pred = current_price * (1 - (horizon * 0.01))  # -1% per hour
                    confidence = max(0.3, 0.7 - (horizon * 0.05))
                    predictions.append(pred)
                    confidences.append(confidence)
                    continue
                
                # Scale features
                scaler = self.price_scalers[horizon]
                data_scaled = scaler.transform(feature_df)
                
                if len(data_scaled) < self.window:
                    # Not enough data
                    predictions.append(current_price)
                    confidences.append(0.3)
                    continue
                
                # Make prediction - use deterministic prediction
                last_sequence = data_scaled[-self.window:].reshape(1, self.window, -1)
                pred_scaled = self.models[horizon].predict(last_sequence, verbose=0)[0][0]
                
                # Convert back to price
                price_scaler = scaler.scale_[0] if scaler.scale_[0] != 0 else 1
                price_mean = scaler.mean_[0]
                pred_price = pred_scaled * price_scaler + price_mean
                
                # Apply consistent constraints (no randomness)
                max_change_pct = min(0.05, 0.01 * horizon)  # Max 1% per hour, up to 5% max
                max_change = current_price * max_change_pct
                
                if abs(pred_price - current_price) > max_change:
                    # Cap the prediction deterministically
                    direction = 1 if pred_price > current_price else -1
                    pred_price = current_price + (direction * max_change)
                
                predictions.append(float(pred_price))
                
                # Calculate CONSISTENT confidence (no randomness)
                # Higher for short term, lower for long term
                base_confidence = max(0.3, 0.7 - (horizon * 0.05))
                # Add deterministic variation based on prediction quality
                prediction_error = abs(pred_price - current_price) / current_price
                confidence_penalty = min(0.2, prediction_error * 10)
                confidence = max(0.3, base_confidence - confidence_penalty)
                confidences.append(float(confidence))
            
            # Display predictions
            print(f"\n🤖 REALISTIC ML PREDICTIONS:")
            timeframes = ['1H', '4H', '24H']
            for tf, pred, conf in zip(timeframes, predictions, confidences):
                change_pct = (pred - current_price) / current_price * 100
                print(f"   {tf}: ${pred:,.2f} ({change_pct:+.1f}%) - {conf:.1%} confidence")
            
            return predictions, confidences
            
        except Exception as e:
            print(f"❌ ML Prediction Error: {e}")
            current_price = price_data['close'].iloc[-1] if 'close' in price_data.columns else 0
            return [current_price] * 3, [0.5, 0.45, 0.4]
    
    def generate_ml_signals(self, predictions, confidences, current_price):
        """Generate ML signals with PERFECT CONSISTENCY"""
        try:
            if len(predictions) < 3:
                return ["🎯 ML: INSUFFICIENT DATA"], 0
            
            # Calculate average expected return and confidence
            returns = [(p - current_price) / current_price for p in predictions]
            avg_return = np.mean(returns)
            avg_confidence = np.mean(confidences)
            
            # ROUND confidence to nearest 5% for better consistency
            avg_confidence = round(avg_confidence * 20) / 20  # Rounds to nearest 0.05 (5%)
            
            # Dynamic thresholds based on recent volatility
            volatility_factor = max(0.5, min(2.0, self.recent_volatility / 0.02))
            bullish_threshold = 0.01 * volatility_factor  # 1% min, scaled by volatility
            strong_threshold = 0.03 * volatility_factor   # 3% strong, scaled by volatility
            cautious_threshold = 0.005 * volatility_factor  # 0.5% for cautious signals
            
            print(f"🤖 ML Analysis: Avg return = {avg_return:+.2%}, Rounded Confidence = {avg_confidence:.1%}")
            print(f"🤖 Dynamic thresholds (Vol={self.recent_volatility:.2%}):")
            print(f"   Cautious: {cautious_threshold:.2%}, Bullish: {bullish_threshold:.2%}, Strong: {strong_threshold:.2%}")
            
            # CONSISTENT SIGNAL GENERATION SYSTEM
            signals = []
            ml_boost = 0
            
            # HIGH CONFIDENCE (>60%)
            if avg_confidence > 0.6:
                if avg_return > strong_threshold:
                    signals.append(f"🎯 ML: STRONG BULLISH ({avg_return:+.1%} expected)")
                    ml_boost = 3
                elif avg_return > bullish_threshold:
                    signals.append(f"🎯 ML: BULLISH ({avg_return:+.1%} expected)")
                    ml_boost = 2
                elif avg_return < -strong_threshold:
                    signals.append(f"🎯 ML: STRONG BEARISH ({avg_return:+.1%} expected)")
                    ml_boost = -3
                elif avg_return < -bullish_threshold:
                    signals.append(f"🎯 ML: BEARISH ({avg_return:+.1%} expected)")
                    ml_boost = -2
                elif avg_return > cautious_threshold:
                    signals.append(f"🎯 ML: SLIGHTLY BULLISH ({avg_return:+.1%} expected)")
                    ml_boost = 1
                elif avg_return < -cautious_threshold:
                    signals.append(f"🎯 ML: SLIGHTLY BEARISH ({avg_return:+.1%} expected)")
                    ml_boost = -1
                else:
                    signals.append(f"🎯 ML: NEUTRAL ({avg_return:+.1%} expected)")
                    ml_boost = 0
            
            # MODERATE CONFIDENCE (50-60%)
            elif avg_confidence > 0.5:
                # Use more conservative thresholds for moderate confidence
                if avg_return > strong_threshold * 1.2:  # Need 3.6%+ for moderate confidence
                    signals.append(f"🎯 ML: CAUTIOUS BULLISH ({avg_return:+.1%} expected)")
                    ml_boost = 2
                elif avg_return > bullish_threshold * 1.5:  # Need 1.5%+ for moderate confidence
                    signals.append(f"🎯 ML: CAUTIOUS BULLISH ({avg_return:+.1%} expected)")
                    ml_boost = 1
                elif avg_return < -strong_threshold * 1.2:  # Need -3.6%+ for moderate confidence
                    signals.append(f"🎯 ML: CAUTIOUS BEARISH ({avg_return:+.1%} expected)")
                    ml_boost = -2
                elif avg_return < -bullish_threshold * 1.5:  # Need -1.5%+ for moderate confidence
                    signals.append(f"🎯 ML: CAUTIOUS BEARISH ({avg_return:+.1%} expected)")
                    ml_boost = -1
                elif abs(avg_return) > cautious_threshold * 3:  # Need bigger move for moderate confidence
                    direction = "BULLISH" if avg_return > 0 else "BEARISH"
                    signals.append(f"🎯 ML: WEAK {direction} ({avg_return:+.1%} expected)")
                    ml_boost = 1 if avg_return > 0 else -1
                else:
                    signals.append(f"🎯 ML: NEUTRAL ({avg_return:+.1%} expected)")
                    ml_boost = 0
            
            # LOW CONFIDENCE (45-50%) - Only act on STRONG signals
            elif avg_confidence > 0.45:
                if avg_return > strong_threshold * 1.5:  # Need 4.5%+ for low confidence
                    signals.append(f"🎯 ML: VERY CAUTIOUS BULLISH ({avg_return:+.1%} expected)")
                    ml_boost = 1
                elif avg_return < -strong_threshold * 1.5:
                    signals.append(f"🎯 ML: VERY CAUTIOUS BEARISH ({avg_return:+.1%} expected)")
                    ml_boost = -1
                else:
                    signals.append(f"🎯 ML: LOW CONFIDENCE - MONITOR ({avg_return:+.1%} expected)")
                    ml_boost = 0
            
            # VERY LOW CONFIDENCE (<45%) - Ignore
            else:
                signals.append("🎯 ML: VERY LOW CONFIDENCE - IGNORE")
                ml_boost = 0
            
            # Add confidence info
            if 0.45 <= avg_confidence <= 0.6:
                signals[-1] = signals[-1] + f" ({avg_confidence:.0%} confidence)"
            
            print(f"🤖 ML SIGNAL GENERATION: Return={avg_return:+.2%}, Confidence={avg_confidence:.1%}, Boost={ml_boost}")
            return signals, ml_boost
            
        except Exception as e:
            print(f"❌ ML signal error: {e}")
            import traceback
            traceback.print_exc()
            return ["🎯 ML: ERROR"], 0
    
    def display_predictions(self, predictions, confidences, ml_signals, model_timeframe="4H"):
        """Display ML predictions - shows what current price SHOULD be if predictions are correct"""
        print(f"\n{'🤖' * 25}")
        print(f"🤖 ML MODEL CONSISTENCY CHECK")
        print(f"🤖 If predictions are correct, then:")
        print(f"{'🤖' * 25}")
        
        # Calculate what current price SHOULD be for predictions to be consistent
        if len(predictions) > 0:
            if model_timeframe.upper() == "1H":
                if len(predictions) > 1:
                    # Assuming predictions[0]=1H, predictions[1]=4H
                    hourly_return = (predictions[0] - predictions[1]) / predictions[1]
                    implied_current = predictions[0] / (1 + hourly_return)
                else:
                    implied_current = predictions[0] / 1.01
            elif model_timeframe.upper() == "4H":
                if len(predictions) > 1:
                    # Assuming predictions[0]=4H, predictions[1]=8H
                    four_hour_return = (predictions[1] - predictions[0]) / predictions[0]
                    implied_current = predictions[0] / (1 + four_hour_return)
                else:
                    implied_current = predictions[0] / 1.04
            elif model_timeframe.upper() == "24H":
                if len(predictions) > 1:
                    daily_return = (predictions[1] - predictions[0]) / predictions[0]
                    implied_current = predictions[0] / (1 + daily_return)
                else:
                    implied_current = predictions[0] / 1.10
            else:
                implied_current = predictions[0] / 1.01
        else:
            implied_current = 0
        
        print(f"\n📊 LOGICAL CONSISTENCY CHECK:")
        print(f"   For these predictions to be correct:")
        print(f"   ➤ Current price SHOULD be: ${implied_current:,.2f}")
        print(f"   ➤ Based on {model_timeframe} model assumptions")
        print(f"📈 Expected volatility: {self.recent_volatility:.2%}")
        print(f"{'─' * 50}")
        
        print(f"\n🎯 MODEL PREDICTIONS (if current = ${implied_current:,.2f}):")
        
        # Define time horizons
        if model_timeframe.upper() == "1H":
            horizons = ['1H', '4H', '24H']
            hour_multipliers = [1, 4, 24]
        elif model_timeframe.upper() == "4H":
            horizons = ['4H', '8H', '24H']
            hour_multipliers = [4, 8, 24]
        elif model_timeframe.upper() == "24H":
            horizons = ['24H', '48H', '72H']
            hour_multipliers = [24, 48, 72]
        else:
            horizons = [f'{model_timeframe}×{i+1}' for i in range(len(predictions))]
            hour_multipliers = [i+1 for i in range(len(predictions))]
        
        for i, (horizon, pred, conf, hours) in enumerate(zip(horizons, predictions, confidences, hour_multipliers)):
            if i < len(predictions):
                implied_return = (pred - implied_current) / implied_current * 100
                confidence_bar = "▰" * int(conf * 10) + "▱" * (10 - int(conf * 10))
                
                print(f"   {horizon}:  ${pred:,.2f} ({implied_return:+.1f}% from implied current)")
                print(f"        {confidence_bar} {conf:.1%} confidence")
                
                # Show implied hourly rate
                if hours > 1:
                    hourly_rate = ((pred / implied_current) ** (1/hours) - 1) * 100
                    print(f"        ≈ {hourly_rate:+.2f}% per hour")
        
        print(f"\n⚡ MARKET IMPLICATIONS:")
        print(f"   1. If actual price > ${implied_current:,.2f}:")
        print(f"      → Predictions may be too optimistic")
        print(f"   2. If actual price < ${implied_current:,.2f}:")
        print(f"      → Predictions may be conservative")
        print(f"   3. If actual price ≈ ${implied_current:,.2f}:")
        print(f"      → Predictions are well-calibrated")
        
        print(f"\n💡 HOW TO USE:")
        print(f"   1. Get actual market price from exchange")
        print(f"   2. Compare to implied price: ${implied_current:,.2f}")
        print(f"   3. If actual ≫ implied → CAUTION (overvalued?)")
        print(f"   4. If actual ≪ implied → OPPORTUNITY (undervalued?)")
        
        # ML signals
        print(f"\n🔍 ML SIGNALS:")
        if ml_signals and "ERROR" not in ml_signals[0] and "INSUFFICIENT" not in ml_signals[0]:
            for signal in ml_signals[:3]:
                if "BULLISH" in signal:
                    print(f"   🟢 {signal}")
                elif "BEARISH" in signal:
                    print(f"   🔴 {signal}")
                elif "NEUTRAL" in signal:
                    print(f"   🟡 {signal}")
                else:
                    print(f"   ⚪ {signal}")
        else:
            print(f"   ⚪ {ml_signals[0] if ml_signals else 'No signals'}")
        
        # Summary
        if len(predictions) >= 2:
            avg_implied_return = np.mean([(p - implied_current) / implied_current for p in predictions]) * 100
            avg_confidence = np.mean(confidences) * 100
            
            print(f"\n📊 CONSISTENCY SUMMARY:")
            print(f"   Avg implied return: {avg_implied_return:+.1f}%")
            print(f"   Avg model confidence: {avg_confidence:.1f}%")
            
            if abs(implied_current - predictions[0]) / implied_current < 0.01:
                print(f"   ✅ Predictions are internally consistent")
            else:
                print(f"   ⚠️  Some prediction inconsistency detected")
    
    def calculate_confidence_quality(self, confidences):
        """Calculate the quality of confidence scores"""
        if len(confidences) < 3:
            return "POOR"
        
        # Check if confidences decrease with horizon (as expected)
        horizon_decrease = all(confidences[i] >= confidences[i+1] for i in range(len(confidences)-1))
        
        # Check confidence range
        confidence_range = max(confidences) - min(confidences)
        
        if horizon_decrease and confidence_range > 0.1:
            return "GOOD"
        elif not horizon_decrease or confidence_range < 0.05:
            return "QUESTIONABLE"
        else:
            return "FAIR"