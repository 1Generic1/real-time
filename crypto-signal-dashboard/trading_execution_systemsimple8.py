# trading_execution_systemsimple7_animated.py - ANIMATED VERSION
from symtable import Symbol
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple
import math
import sys
import os

# Set global random seeds for reproducibility
SEED = 42  # Fixed seed for consistent results
random.seed(SEED)
np.random.seed(SEED)

# Add this constant for consistent ML training
ML_TRAINING_DATA_LENGTH = 100  # Consistent data length for all ML training

# Add on-chain cache for consistency
ONCHAIN_CACHE = {}
ONCHAIN_CACHE_DURATION = 300  # 5 minutes cache

# Import your existing modules
try:
    from c_signal2 import fetch_price_data, calculate_technical_indicators, generate_trading_signals
except ImportError:
    print("❌ Could not import c_signal module")
    def fetch_price_data(*args, **kwargs): return None
    def calculate_technical_indicators(*args, **kwargs): return None
    def generate_trading_signals(*args, **kwargs): return []

# Import OnChainAnalyzer
try:
    from onchain_analyzer3 import OnChainAnalyzer
    print("✅ Successfully imported OnChainAnalyzer from onchain_analyzer3")
except ImportError as e:
    print(f"❌ Could not import OnChainAnalyzer: {e}")
    class OnChainAnalyzer:
        def __init__(self): self.historical_flows = {}
        def accelerate_data_collection(self, *args): pass
        def get_comprehensive_onchain_analysis(self, *args): return {}
        def analyze_comprehensive_signals(self, *args): return []

# Import ML Predictor
try:
    from advanced_ml_predictorsimple5 import RealisticMLPredictor
    ML_AVAILABLE = True
    print("✅ SIMPLE REALISTIC ML SYSTEM IMPORTED")
except ImportError as e:
    print(f"❌ Advanced ML Import Failed: {e}")
    ML_AVAILABLE = False
    class RealisticMLPredictor:
        def __init__(self): self.is_trained = False
        def train_simple_model(self, *args): print("⚠️  ML not available"); return False
        def predict_simple(self, *args): return [0]*3, [0.5]*3
        def generate_ml_signals(self, *args): return [], 0
        def display_predictions(self, *args): print("🤖 ML Predictions: Not available")

# ==================== ANIMATION FUNCTIONS ====================
class TradingAnimator:
    @staticmethod
    def clear_screen():
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def animate_title(title, delay=0.03):
        """Animate title text"""
        print("\n" + "═" * 60)
        for char in title:
            print(char, end='', flush=True)
            time.sleep(delay)
        print("\n" + "═" * 60)
    
    @staticmethod
    def loading_bar(description, duration=1.5, width=40):
        """Display animated loading bar"""
        print(f"\n⏳ {description}")
        sys.stdout.write("[")
        for i in range(width):
            time.sleep(duration / width)
            sys.stdout.write("▓" if i % 2 == 0 else "▒")
            sys.stdout.flush()
        sys.stdout.write("] 100%\n")
    
    @staticmethod
    def spinning_cursor(description, duration=2):
        """Display spinning cursor animation"""
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        print(f"\n🌀 {description}")
        end_time = time.time() + duration
        while time.time() < end_time:
            for frame in spinner:
                sys.stdout.write(f"\r{frame} Processing... ")
                sys.stdout.flush()
                time.sleep(0.1)
        print("\r✅ Done!" + " " * 20)
    
    @staticmethod
    def flash_text(text, times=3, color_code="\033[93m"):
        """Flash text animation"""
        reset = "\033[0m"
        for _ in range(times):
            print(f"\r{color_code}{text}{reset}", end='', flush=True)
            time.sleep(0.3)
            print(f"\r{' ' * len(text)}", end='', flush=True)
            time.sleep(0.3)
        print(f"\r{text}")
    
    @staticmethod
    def countdown(seconds=3, message="Starting analysis"):
        """Display countdown animation"""
        print(f"\n⏰ {message}")
        for i in range(seconds, 0, -1):
            print(f"\r🚀 {i}... ", end='', flush=True)
            time.sleep(1)
        print("\r🚀 GO!" + " " * 10)
    
    @staticmethod
    def progress_percentage(description, max_value=100, delay=0.02):
        """Display percentage progress animation"""
        print(f"\n📊 {description}")
        for i in range(0, max_value + 1, 2):
            bar = "▓" * int(i / 100 * 30) + "░" * (30 - int(i / 100 * 30))
            print(f"\r[{bar}] {i}%", end='', flush=True)
            time.sleep(delay)
        print(f"\r[{bar}] 100% Complete!")
    
    @staticmethod
    def price_movement_animation(price, change_pct, duration=1):
        """Animate price movement"""
        arrow = "↗️" if change_pct > 0 else "↘️" if change_pct < 0 else "➡️"
        color = "\033[92m" if change_pct > 0 else "\033[91m" if change_pct < 0 else "\033[93m"
        reset = "\033[0m"
        
        print(f"\n📈 Price Animation:")
        print(f"   {color}{arrow} ${price:,.2f} ({change_pct:+.2f}%){reset}")
        
        # Animate movement
        steps = 10
        base_price = price / (1 + change_pct/100)
        for step in range(steps + 1):
            current_price = base_price * (1 + (change_pct/100) * (step/steps))
            current_change = ((current_price - base_price) / base_price) * 100
            
            # Create bar visualization
            bar_length = 30
            if change_pct > 0:
                filled = int(bar_length * (step/steps))
                bar = "🟩" * filled + "⬜" * (bar_length - filled)
            elif change_pct < 0:
                filled = int(bar_length * (step/steps))
                bar = "🟥" * filled + "⬜" * (bar_length - filled)
            else:
                bar = "🟨" * bar_length
            
            print(f"\r   {bar} ${current_price:,.2f} ({current_change:+.2f}%)", end='', flush=True)
            time.sleep(duration/steps)
        print()

# ==================== ANIMATED TRADING SYSTEM ====================

# Add price data cache
price_data_cache = {}

def get_cached_or_fetch_price_data(symbol, timeframe, limit, cache_minutes=5):
    """
    Cache price data to ensure consistency between analyses
    """
    cache_key = f"{symbol}_{timeframe}_{limit}"
    current_time = datetime.now()
    
    if cache_key in price_data_cache:
        cache_time, cached_data = price_data_cache[cache_key]
        time_diff = (current_time - cache_time).total_seconds()
        
        if time_diff < cache_minutes * 60:
            TradingAnimator.flash_text(f"📊 Using cached data ({time_diff:.0f}s ago)", 1)
            return cached_data.copy()
        else:
            print(f"   📊 Cache expired, fetching fresh...")
    
    # Fetch fresh data with animation
    TradingAnimator.loading_bar(f"Fetching {symbol} price data", 1.0)
    data = fetch_price_data(symbol, timeframe, limit)
    if data is not None:
        price_data_cache[cache_key] = (current_time, data.copy())
        TradingAnimator.flash_text(f"✅ Data fetched successfully", 1, "\033[92m")
        return data.copy()
    return None

def get_cached_onchain_data(symbol):
    """Cache on-chain data for consistency"""
    cache_key = f"onchain_{symbol}"
    current_time = datetime.now()
    
    if cache_key in ONCHAIN_CACHE:
        cache_time, cached_data = ONCHAIN_CACHE[cache_key]
        time_diff = (current_time - cache_time).total_seconds()
        
        if time_diff < ONCHAIN_CACHE_DURATION:
            TradingAnimator.flash_text(f"⛓️ Using cached on-chain data", 1)
            return cached_data.copy()
    
    return None

def cache_onchain_data(symbol, data):
    """Cache on-chain data"""
    cache_key = f"onchain_{symbol}"
    ONCHAIN_CACHE[cache_key] = (datetime.now(), data.copy())
    print(f"   ⛓️ Cached on-chain data for {symbol}")

class TradingExecutionSystem:
    def __init__(self, account_balance=1000, risk_per_trade=0.02, max_position_size=0.3):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade  # 2% risk per trade
        self.max_position_size = max_position_size
        self.trading_log = []
        
        # Animation for initialization
        TradingAnimator.animate_title("🏦 TRADING EXECUTION SYSTEM INITIALIZED")
        TradingAnimator.progress_percentage("Loading trading engine", 100, 0.01)
        
        # UPDATED: More comprehensive signal weights
        try:
            from c_signal2 import SIGNAL_WEIGHTS
            self.signal_weights = SIGNAL_WEIGHTS
            print("✅ Imported signal weights from c_signal.py")
        except ImportError:
            print("⚠️  Could not import SIGNAL_WEIGHTS from c_signal.py")
            # Fallback weights - UPDATED FOR BETTER MATCHING
            self.signal_weights = {
                'STRONG UPTREND': 3,
                'STRONG DOWNTREND': -3,
                'UPTREND': 2,
                'DOWNTREND': -2,
                'OVERSOLD': 2,
                'OVERBOUGHT': -2,
                'BULLISH CROSSOVER': 2,
                'BULLISH MOMENTUM': 2,  # ADDED
                'BULLISH DIVERGENCE': 2,  # ADDED
                'BEARISH CROSSOVER': -2,
                'BEARISH MOMENTUM': -2,  # ADDED
                'BEARISH DIVERGENCE': -2,  # ADDED
                'AT UPPER BOLLINGER': -1,
                'UPPER BOLLINGER': -1,
                'AT LOWER BOLLINGER': 1,
                'LOWER BOLLINGER': 1,
                'MIDDLE BOLLINGER': 0,
                'HIGH VOLUME': 1,
                'LOW VOLUME': -1,
                'VOLUME SPIKE': 1,  # ADDED
                'VOLUME DROP': -1,  # ADDED
                'SUPPORT': 1,
                'STRONG SUPPORT': 2,  # ADDED
                'NEAR SUPPORT': 1,  # ADDED
                'NEAR STRONG SUPPORT': 2,
                'RESISTANCE': -1,
                'STRONG RESISTANCE': -2,  # ADDED
                'NEAR RESISTANCE': -1,  # ADDED
                'NEAR STRONG RESISTANCE': -2,
                'RANGING': 0,  # ADDED
                'CONSOLIDATION': 0,  # ADDED
                'MIXED SIGNALS': 0,  # ADDED
                'MIXED MA SIGNALS': 0,  # ADDED
                'BREAKOUT': 2,  # ADDED
                'BREAKDOWN': -2,  # ADDED
                'GOLDEN CROSS': 2,  # ADDED
                'DEATH CROSS': -2,  # ADDED
            }
        
        # REVISED: More realistic leverage recommendations by asset type
        self.asset_volatility_categories = {
            'LOW_VOL': ['BTC/USDT'],  # 2-5% daily moves
            'MED_VOL': ['ETH/USDT'],  # 3-7% daily moves  
            'HIGH_VOL': ['SOL/USDT', 'DOT/USDT', 'AVAX/USDT'],  # 5-12% daily moves
            'EXTREME_VOL': ['ADA/USDT', 'MATIC/USDT', 'LINK/USDT', 'DOGE/USDT', 'SHIB/USDT']  # 8-20% daily moves
        }
        
        # REVISED: Better leverage recommendations based on volatility
        self.leverage_recommendations = {
            'LOW_VOL': {
                'conservative': 3,
                'balanced': 5,
                'aggressive': 8,
                'max_safe': 10,
                'description': 'Low volatility (BTC-like)'
            },
            'MED_VOL': {
                'conservative': 2,
                'balanced': 3,
                'aggressive': 5,
                'max_safe': 7,
                'description': 'Medium volatility (ETH-like)'
            },
            'HIGH_VOL': {
                'conservative': 1.5,
                'balanced': 2,
                'aggressive': 3,
                'max_safe': 5,
                'description': 'High volatility (SOL-like)'
            },
            'EXTREME_VOL': {
                'conservative': 1,
                'balanced': 1.5,
                'aggressive': 2,
                'max_safe': 3,
                'description': 'Extreme volatility (memecoins, alts)'
            }
        }
        
        print(f"💰 Account Balance: ${account_balance:,.2f}")
        print(f"🎯 Risk per Trade: {risk_per_trade*100}%")
        print(f"⚖️ Max Position Size: {max_position_size*100}% of account\n")

    def get_asset_volatility_category(self, symbol):
        """Determine volatility category for asset"""
        for category, symbols in self.asset_volatility_categories.items():
            if symbol in symbols:
                return category
        return 'MED_VOL'  # Default
    
    def calculate_volatility_based_leverage(self, symbol, price_data=None):
        """Calculate recommended leverage based on actual volatility"""
        TradingAnimator.spinning_cursor(f"Calculating volatility for {symbol}", 1)
        category = self.get_asset_volatility_category(symbol)
        
        # If we have price data, calculate actual ATR volatility
        if price_data is not None and 'atr' in price_data.columns and len(price_data) > 20:
            atr = price_data['atr'].iloc[-1]
            current_price = price_data['close'].iloc[-1]
            atr_percentage = (atr / current_price) * 100
            
            # Animate volatility calculation
            print(f"📊 Current Volatility: {atr_percentage:.1f}%")
            
            # Adjust recommendations based on actual volatility
            recommendations = self.leverage_recommendations[category].copy()
            
            # High volatility warning
            if atr_percentage > 10:  # Very high volatility
                recommendations['conservative'] = max(1, recommendations['conservative'] * 0.7)
                recommendations['balanced'] = max(1, recommendations['balanced'] * 0.7)
                recommendations['aggressive'] = max(1.5, recommendations['aggressive'] * 0.7)
                recommendations['max_safe'] = max(2, recommendations['max_safe'] * 0.7)
                TradingAnimator.flash_text(f"⚠️ High volatility detected: {atr_percentage:.1f}% daily ATR", 2, "\033[91m")
            elif atr_percentage > 7:  # Above average volatility
                recommendations['conservative'] = max(1, recommendations['conservative'] * 0.8)
                recommendations['balanced'] = max(1, recommendations['balanced'] * 0.8)
                recommendations['aggressive'] = max(1.5, recommendations['aggressive'] * 0.8)
                recommendations['max_safe'] = max(2, recommendations['max_safe'] * 0.8)
                TradingAnimator.flash_text(f"📈 Above avg volatility: {atr_percentage:.1f}% daily ATR", 2, "\033[93m")
            else:
                recommendations['current_atr'] = f"{atr_percentage:.1f}% daily"
                TradingAnimator.flash_text(f"✅ Normal volatility: {atr_percentage:.1f}% daily ATR", 1, "\033[92m")
            
            return recommendations
        
        return self.leverage_recommendations[category]

    def calculate_position_size(self, entry_price, stop_loss_price, symbol, leverage=None, price_data=None):
        """Calculate position size with risk management - CORRECTED to risk exactly 2% of capital"""
        TradingAnimator.animate_title("📊 POSITION SIZING CALCULATION", 0.02)
        
        # Calculate the risk amount (2% of account balance)
        risk_amount = self.account_balance * self.risk_per_trade
        
        # Calculate the price distance to stop loss
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0, 0, 0, 0, 0, 0, {}
        
        # FIXED: Calculate position size based on RISK AMOUNT, not account balance
        # Formula: Position Size (units) = Risk Amount / Price Risk
        position_size_units = risk_amount / price_risk
        
        # Calculate position value at entry price
        position_value = position_size_units * entry_price
        
        # Apply maximum position size constraint (30% of account)
        max_position_value = self.account_balance * self.max_position_size
        if position_value > max_position_value:
            position_value = max_position_value
            position_size_units = position_value / entry_price
            # Recalculate risk amount based on adjusted position size
            risk_amount = position_size_units * price_risk
        
        # Get recommended leverage based on volatility
        leverage_recommendations = self.calculate_volatility_based_leverage(symbol, price_data)
        
        # Use provided leverage or default to balanced recommendation
        if leverage is None:
            leverage = leverage_recommendations['balanced']
        
        # Calculate margin required (position value divided by leverage)
        margin_required = position_value / leverage
        
        # Animate margin calculation
        TradingAnimator.progress_percentage("Calculating margin requirements", 75, 0.01)
        
        # Calculate liquidation price - FIXED FORMULA
        direction = "LONG" if entry_price > stop_loss_price else "SHORT"
        
        # CORRECTED: Realistic liquidation calculation for Binance USDT-M futures
        maintenance_margin_rate = 0.05  # 5% maintenance margin typical for crypto
        initial_margin_rate = 1.0 / leverage  # e.g., 5x leverage = 20% initial margin
        
        if direction == "LONG":
            # For LONG: liquidation when price drops enough to hit maintenance margin
            liquidation_price = entry_price * (1 - (initial_margin_rate - maintenance_margin_rate))
        else:
            # For SHORT: liquidation when price rises enough to hit maintenance margin
            liquidation_price = entry_price * (1 + (initial_margin_rate - maintenance_margin_rate))
        
        # SAFETY CHECK: Liquidation should be below entry for LONG, above for SHORT
        if direction == "LONG" and liquidation_price > entry_price:
            liquidation_price = entry_price * 0.85  # Fallback: 15% below entry
            
        elif direction == "SHORT" and liquidation_price < entry_price:
            liquidation_price = entry_price * 1.15  # Fallback: 15% above entry
        
        # Calculate buffer to liquidation
        if direction == "LONG":
            liquidation_distance = ((entry_price - liquidation_price) / entry_price) * 100
            stop_distance = ((entry_price - stop_loss_price) / entry_price) * 100
        else:
            liquidation_distance = ((liquidation_price - entry_price) / entry_price) * 100
            stop_distance = ((stop_loss_price - entry_price) / entry_price) * 100
        
        buffer_to_liquidation = liquidation_distance - stop_distance
        
        # Debug info to verify risk
        actual_risk_percentage = (risk_amount / self.account_balance) * 100
        
        # Check if we got full 2% risk
        risk_status = "✅" if abs(actual_risk_percentage - (self.risk_per_trade * 100)) < 0.1 else "❌"
        
        # Animate final calculations
        TradingAnimator.progress_percentage("Finalizing position sizing", 100, 0.01)
        
        return (position_size_units, position_value, margin_required, leverage, 
                liquidation_price, buffer_to_liquidation, leverage_recommendations, 
                risk_amount, actual_risk_percentage, risk_status)

    def calculate_leverage_scenarios(self, entry_price, stop_loss, position_value, symbol, direction, price_data=None):
        """Calculate different leverage scenarios with realistic recommendations"""
        TradingAnimator.animate_title("⚡ LEVERAGE SCENARIO ANALYSIS", 0.02)
        
        scenarios = []
        
        # Get volatility-based recommendations
        leverage_recs = self.calculate_volatility_based_leverage(symbol, price_data)
        
        # Define leverage options to test
        leverage_options = [
            1,  # No leverage
            leverage_recs['conservative'],
            leverage_recs['balanced'],
            leverage_recs['aggressive'],
            leverage_recs['max_safe'],
            5,   # Added: Common moderate leverage
            8,   # Added: Common high leverage
            10,  # Common but often too high
            15,  # Added: Very high leverage
            20,  # Very risky
            25,  # Extremely risky
        ]
        
        # Remove duplicates and sort
        leverage_options = sorted(list(set([round(x, 1) for x in leverage_options])))
        
        # Calculate stop distance percentage
        if direction == "LONG":
            stop_distance = ((entry_price - stop_loss) / entry_price) * 100
        else:
            stop_distance = ((stop_loss - entry_price) / entry_price) * 100
        
        # Animate scenario calculations
        TradingAnimator.loading_bar("Calculating leverage scenarios", 1.5)
        
        for i, leverage in enumerate(leverage_options):
            # Show progress for each scenario
            progress = (i + 1) / len(leverage_options) * 100
            print(f"\r📊 Calculating scenario {i+1}/{len(leverage_options)}... {progress:.0f}%", end='', flush=True)
            
            margin_required = position_value / leverage
            
            # CORRECTED: liquidation calculation
            maintenance_margin_rate = 0.05
            initial_margin_rate = 1.0 / leverage
            
            if direction == "LONG":
                liquidation = entry_price * (1 - (initial_margin_rate - maintenance_margin_rate))
                liquidation_pct = ((entry_price - liquidation) / entry_price) * 100
                buffer_to_liquidation = liquidation_pct - stop_distance
            else:
                liquidation = entry_price * (1 + (initial_margin_rate - maintenance_margin_rate))
                liquidation_pct = ((liquidation - entry_price) / entry_price) * 100
                buffer_to_liquidation = liquidation_pct - stop_distance
            
            # SAFETY CHECK for liquidation
            if direction == "LONG" and liquidation > entry_price:
                liquidation = entry_price * 0.85
                liquidation_pct = ((entry_price - liquidation) / entry_price) * 100
                buffer_to_liquidation = liquidation_pct - stop_distance
            elif direction == "SHORT" and liquidation < entry_price:
                liquidation = entry_price * 1.15
                liquidation_pct = ((liquidation - entry_price) / entry_price) * 100
                buffer_to_liquidation = liquidation_pct - stop_distance
            
            # Determine which recommendation this is
            if abs(leverage - leverage_recs['conservative']) < 0.1:
                recommendation = "🟢 CONSERVATIVE"
                is_recommended = True
            elif abs(leverage - leverage_recs['balanced']) < 0.1:
                recommendation = "🟡 BALANCED"
                is_recommended = True
            elif abs(leverage - leverage_recs['aggressive']) < 0.1:
                recommendation = "🟠 AGGRESSIVE"
                is_recommended = True
            elif abs(leverage - leverage_recs['max_safe']) < 0.1:
                recommendation = "🔴 MAX SAFE"
                is_recommended = False
            elif leverage > leverage_recs['max_safe']:
                recommendation = "⚠️ DANGEROUS"
                is_recommended = False
            else:
                recommendation = "⚪ STANDARD"
                is_recommended = False
            
            # Risk assessment with animated colors
            if buffer_to_liquidation > 30:
                risk_level = "🟢 VERY SAFE"
                risk_color = "\033[92m"
            elif buffer_to_liquidation > 20:
                risk_level = "🟢 SAFE"
                risk_color = "\033[92m"
            elif buffer_to_liquidation > 15:
                risk_level = "🟡 MODERATE"
                risk_color = "\033[93m"
            elif buffer_to_liquidation > 10:
                risk_level = "🟠 HIGH"
                risk_color = "\033[93m"
            elif buffer_to_liquidation > 5:
                risk_level = "🔴 DANGEROUS"
                risk_color = "\033[91m"
            else:
                risk_level = "💀 EXTREME"
                risk_color = "\033[91m"
            
            scenarios.append({
                'leverage': leverage,
                'margin': margin_required,
                'liquidation': liquidation,
                'buffer_pct': buffer_to_liquidation,
                'risk': risk_level,
                'recommendation': recommendation,
                'is_recommended': is_recommended,
                'is_dangerous': leverage > leverage_recs['max_safe'],
                'risk_color': risk_color
            })
        
        print("\r✅ All scenarios calculated!" + " " * 30)
        return scenarios, leverage_recs

    def score_signals(self, ta_signals, onchain_signals):
        """CONSISTENT signal scoring with IMPROVED matching logic"""
        TradingAnimator.animate_title("🎯 SIGNAL SCORING ANALYSIS", 0.03)
        
        score = 0
        
        print("\n🔍 TECHNICAL ANALYSIS SCORING:")
        TradingAnimator.loading_bar("Processing technical signals", 1.0)
        
        # Technical Analysis Scoring
        for signal in ta_signals:
            signal_score = 0
            matched_type = None
            
            # FIXED: Check for exact keyword matching
            signal_upper = signal.upper()
            
            # Try to match exact signal types first
            for signal_type, weight in self.signal_weights.items():
                signal_type_upper = signal_type.upper()
                
                if signal_type_upper in signal_upper:
                    words = signal_type_upper.split()
                    if len(words) > 1:
                        if all(word in signal_upper for word in words):
                            score += weight
                            signal_score = weight
                            matched_type = signal_type
                            break
                    else:
                        score += weight
                        signal_score = weight
                        matched_type = signal_type
                        break
            
            # Fallback scoring for unmatched signals
            if signal_score == 0:
                bullish_keywords = ['BULLISH', 'UPTREND', 'OVERSOLD', 'SUPPORT', 'LOWER BOLLINGER', 'BREAKOUT', 'GOLDEN CROSS']
                bearish_keywords = ['BEARISH', 'DOWNTREND', 'OVERBOUGHT', 'RESISTANCE', 'UPPER BOLLINGER', 'BREAKDOWN', 'DEATH CROSS']
                
                if any(keyword in signal_upper for keyword in bullish_keywords):
                    if 'STRONG' in signal_upper or 'CROSSOVER' in signal_upper or 'BREAKOUT' in signal_upper:
                        score += 2
                        signal_score = 2
                        matched_type = "Strong Bullish (fallback)"
                    else:
                        score += 1
                        signal_score = 1
                        matched_type = "Bullish (fallback)"
                elif any(keyword in signal_upper for keyword in bearish_keywords):
                    if 'STRONG' in signal_upper or 'CROSSOVER' in signal_upper or 'BREAKDOWN' in signal_upper:
                        score -= 2
                        signal_score = -2
                        matched_type = "Strong Bearish (fallback)"
                    else:
                        score -= 1
                        signal_score = -1
                        matched_type = "Bearish (fallback)"
                else:
                    matched_type = "Neutral (no match)"
            
            # Animated score display
            color = "\033[92m" if signal_score > 0 else "\033[91m" if signal_score < 0 else "\033[93m"
            reset = "\033[0m"
            print(f"  {color}{signal_score:+2}{reset} ← {signal[:50]:50}")
        
        print(f"\n📊 TECHNICAL SUBTOTAL: {score:+d}")
        
        # On-Chain Scoring
        print("\n⛓️  ON-CHAIN ANALYSIS:")
        TradingAnimator.loading_bar("Processing on-chain signals", 0.8)
        
        for signal in onchain_signals:
            signal_upper = signal.upper()
            
            if 'BEARISH' in signal_upper and ('🔴' in signal or 'HIGH' in signal_upper or 'CAUTION' in signal_upper):
                score -= 2
                print(f"  \033[91m-2\033[0m ← {signal[:50]:50}")
            elif 'BULLISH' in signal_upper and ('🟢' in signal or 'LOW' in signal_upper):
                score += 2
                print(f"  \033[92m+2\033[0m ← {signal[:50]:50}")
            else:
                print(f"  \033[93m 0\033[0m ← {signal[:50]:50}")
        
        # Animate final score
        TradingAnimator.flash_text(f"🎯 FINAL SCORE: {score:+d}", 3, "\033[95m" if abs(score) >= 3 else "\033[93m")
        
        return score

    def determine_trade_direction(self, signal_score, ta_signals):
        """BALANCED trade direction with SHORT support"""
        TradingAnimator.spinning_cursor("Determining trade direction", 1)
        
        # Check for clear technical direction
        bearish_indicators = sum(1 for s in ta_signals if 'BEARISH' in s or 'DOWNTREND' in s)
        bullish_indicators = sum(1 for s in ta_signals if 'BULLISH' in s or 'UPTREND' in s)
        
        # ADJUSTED THRESHOLDS: Easier to get SHORT signals
        if signal_score >= 3:
            TradingAnimator.flash_text("✅ DIRECTION: LONG", 2, "\033[92m")
            return "LONG"
        elif signal_score <= -3:
            TradingAnimator.flash_text("✅ DIRECTION: SHORT", 2, "\033[91m")
            return "SHORT"
        elif signal_score <= -2 and bearish_indicators >= 3:
            TradingAnimator.flash_text("✅ DIRECTION: SHORT", 2, "\033[91m")
            return "SHORT"
        else:
            TradingAnimator.flash_text("❌ NO TRADE: Insufficient signal", 1, "\033[93m")
            return "NO_TRADE"

    def calculate_levels(self, direction, price_data, current_price, symbol):
        """Calculate entry, stop loss, and take profit for BOTH LONG and SHORT"""
        TradingAnimator.animate_title("📐 PRICE LEVEL CALCULATION", 0.02)
        
        high_20 = price_data['high'].tail(20).max()
        low_20 = price_data['low'].tail(20).min()
        
        if 'atr' in price_data.columns and not pd.isna(price_data['atr'].iloc[-1]):
            atr = price_data['atr'].iloc[-1]
        else:
            atr = current_price * 0.02
        
        # SET MINIMUM STOP DISTANCE BASED ON VOLATILITY
        min_stop_pct = max(0.015, (atr / current_price) * 2)  # Minimum 1.5% or 2x ATR
        
        if direction == "LONG":
            TradingAnimator.flash_text("📈 CALCULATING LONG LEVELS", 1, "\033[92m")
            
            # LONG setup
            support_level = price_data['support'].iloc[-1] if 'support' in price_data.columns else current_price * 0.95
            if not pd.isna(support_level) and current_price <= support_level * 1.02:
                entry_price = min(current_price, support_level * 1.01)
            else:
                entry_price = current_price
            
            # Use MINIMUM stop distance (not too tight)
            atr_stop = entry_price - (atr * 2)  # 2x ATR for safety
            price_based_stop = entry_price * (1 - min_stop_pct)  # Minimum stop percentage
            stop_loss = max(low_20, min(atr_stop, price_based_stop))
            
            # Ensure stop isn't too close
            min_stop_distance = entry_price * min_stop_pct
            if (entry_price - stop_loss) < min_stop_distance:
                stop_loss = entry_price - min_stop_distance
            
            risk_amount = entry_price - stop_loss
            
            take_profits = [
                entry_price + risk_amount * 1,
                entry_price + risk_amount * 2,
                entry_price + risk_amount * 3
            ]
            
        else:  # SHORT setup
            TradingAnimator.flash_text("📉 CALCULATING SHORT LEVELS", 1, "\033[91m")
            
            # SHORT: Entry near resistance
            resistance_level = price_data['resistance'].iloc[-1] if 'resistance' in price_data.columns else current_price * 1.05
            if not pd.isna(resistance_level) and current_price >= resistance_level * 0.98:
                entry_price = max(current_price, resistance_level * 0.99)
            else:
                entry_price = current_price
            
            # SHORT: Use MINIMUM stop distance
            atr_stop = entry_price + (atr * 2)  # 2x ATR for safety
            price_based_stop = entry_price * (1 + min_stop_pct)  # Minimum stop percentage
            stop_loss = min(high_20, max(atr_stop, price_based_stop))
            
            # Ensure stop isn't too close
            min_stop_distance = entry_price * min_stop_pct
            if (stop_loss - entry_price) < min_stop_distance:
                stop_loss = entry_price + min_stop_distance
            
            risk_amount = stop_loss - entry_price
            
            # SHORT: Take profits BELOW entry
            take_profits = [
                entry_price - risk_amount * 1,
                entry_price - risk_amount * 2,
                entry_price - risk_amount * 3
            ]
        
        # Animate level calculation
        print(f"\n📊 Calculated Levels:")
        print(f"   Entry: ${entry_price:,.2f}")
        print(f"   Stop Loss: ${stop_loss:,.2f}")
        print(f"   Take Profits: ${take_profits[0]:,.2f}, ${take_profits[1]:,.2f}, ${take_profits[2]:,.2f}")
        
        return entry_price, stop_loss, take_profits

    def calculate_risk_reward(self, entry, stop_loss, take_profits):
        """Calculate risk-reward ratio"""
        risk = abs(entry - stop_loss)
        if risk == 0:
            return 0
        avg_reward = sum([abs(tp - entry) for tp in take_profits]) / len(take_profits)
        return avg_reward / risk

    def generate_trading_signal(self, ta_signals, onchain_signals, price_data, symbol, custom_leverage=None):
        """Generate trading signals for BOTH LONG and SHORT with custom leverage"""
        if price_data is None or len(price_data) == 0:
            return None
        
        TradingAnimator.animate_title("🚀 GENERATING TRADING SIGNAL", 0.03)
            
        current_price = price_data['close'].iloc[-1]
        
        # Score signals
        signal_score = self.score_signals(ta_signals, onchain_signals)
        
        # Determine trade direction
        direction = self.determine_trade_direction(signal_score, ta_signals)
        
        if direction == "NO_TRADE":
            return None
        
        # Calculate levels with MINIMUM STOP
        entry_price, stop_loss, take_profits = self.calculate_levels(
            direction, price_data, current_price, symbol
        )
        
        # Calculate stop distance percentage
        if direction == "LONG":
            stop_distance_pct = ((entry_price - stop_loss) / entry_price) * 100
        else:
            stop_distance_pct = ((stop_loss - entry_price) / entry_price) * 100
        
        # REJECT trades with stops that are too small
        MIN_STOP_PCT = 1.5  # Minimum 1.5% stop
        if stop_distance_pct < MIN_STOP_PCT:
            TradingAnimator.flash_text(f"⚠️ Stop too small: {stop_distance_pct:.1f}% < {MIN_STOP_PCT}%", 2, "\033[91m")
            return None
        
        # Calculate position sizing with custom leverage
        (position_size, position_value, margin_required, leverage, 
         liquidation_price, buffer_to_liq, leverage_recs, 
         risk_amount, actual_risk_percentage, risk_status) = self.calculate_position_size(
            entry_price, stop_loss, symbol, custom_leverage, price_data
        )
        
        if position_size == 0:
            return None
        
        # Calculate leverage scenarios
        leverage_scenarios, leverage_recommendations = self.calculate_leverage_scenarios(
            entry_price, stop_loss, position_value, symbol, direction, price_data
        )
        
        # Generate trade plan
        trade_signal = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profits': take_profits,
            'position_size': position_size,
            'position_value': position_value,
            'margin_required': margin_required,
            'risk_reward_ratio': self.calculate_risk_reward(entry_price, stop_loss, take_profits),
            'signal_strength': signal_score,
            'timestamp': datetime.now(),
            'leverage': leverage,
            'liquidation_price': liquidation_price,
            'buffer_to_liquidation': buffer_to_liq,
            'leverage_scenarios': leverage_scenarios,
            'leverage_recommendations': leverage_recommendations,
            'account_balance': self.account_balance,
            'risk_amount': risk_amount,
            'risk_percentage': actual_risk_percentage,
            'risk_status': risk_status,
            'asset_category': self.get_asset_volatility_category(symbol),
            'price_at_analysis': current_price,
            'stop_distance_pct': stop_distance_pct
        }
        
        TradingAnimator.flash_text("✅ TRADING SIGNAL GENERATED!", 3, "\033[92m")
        return trade_signal

    def generate_trading_plan(self, trade_signal, show_scenarios=True):
        """Generate detailed trading plan with SHORT support and leverage analysis"""
        TradingAnimator.animate_title("📋 TRADING PLAN", 0.02)
        
        if not trade_signal:
            TradingAnimator.flash_text("❌ NO TRADE: Insufficient signal strength", 2, "\033[91m")
            return "NO TRADE: Insufficient signal strength"
        
        # Get current price to see if signal is still valid
        try:
            current_data = fetch_price_data(trade_signal['symbol'], '1m', 2)
            if current_data is not None:
                current_price = current_data['close'].iloc[-1]
                price_at_signal = trade_signal.get('price_at_analysis', trade_signal['entry_price'])
                price_change = ((current_price - price_at_signal) / price_at_signal) * 100
                TradingAnimator.price_movement_animation(current_price, price_change)
            else:
                current_price = None
                price_change = 0
        except:
            current_price = None
            price_change = 0
        
        direction_emoji = "🟢" if trade_signal['direction'] == "LONG" else "🔴"
        
        # Create animated plan header
        TradingAnimator.flash_text(f"{direction_emoji} TRADING PLAN: {trade_signal['direction']} {trade_signal['symbol']} {direction_emoji}", 2, "\033[96m")
        
        # Calculate price movements
        if trade_signal['direction'] == 'LONG':
            stop_distance = ((trade_signal['entry_price'] - trade_signal['stop_loss']) / trade_signal['entry_price']) * 100
            tp1_distance = ((trade_signal['take_profits'][0] - trade_signal['entry_price']) / trade_signal['entry_price']) * 100
            tp2_distance = ((trade_signal['take_profits'][1] - trade_signal['entry_price']) / trade_signal['entry_price']) * 100
            tp3_distance = ((trade_signal['take_profits'][2] - trade_signal['entry_price']) / trade_signal['entry_price']) * 100
            liquidation_distance = ((trade_signal['entry_price'] - trade_signal['liquidation_price']) / trade_signal['entry_price']) * 100
        else:
            stop_distance = ((trade_signal['stop_loss'] - trade_signal['entry_price']) / trade_signal['entry_price']) * 100
            tp1_distance = ((trade_signal['entry_price'] - trade_signal['take_profits'][0]) / trade_signal['entry_price']) * 100
            tp2_distance = ((trade_signal['entry_price'] - trade_signal['take_profits'][1]) / trade_signal['entry_price']) * 100
            tp3_distance = ((trade_signal['entry_price'] - trade_signal['take_profits'][2]) / trade_signal['entry_price']) * 100
            liquidation_distance = ((trade_signal['liquidation_price'] - trade_signal['entry_price']) / trade_signal['entry_price']) * 100
        
        # Get asset category info
        category_info = trade_signal['leverage_recommendations']
        
        # Build plan with animated sections
        plan_parts = []
        
        # TIMING SECTION
        TradingAnimator.loading_bar("Generating timing section", 0.5)
        plan_parts.append(f"""
⏰ TIMING INFORMATION:
   Analysis Time: {trade_signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
   Price at Analysis: ${trade_signal.get('price_at_analysis', trade_signal['entry_price']):,.2f}""")
        
        if current_price:
            plan_parts.append(f"""
   Current Price: ${current_price:,.2f}
   Price Change: {price_change:+.2f}%""")
            
            if abs(price_change) > 2.0:
                plan_parts.append(f"\n   ⚠️  WARNING: Price moved {price_change:+.2f}% since analysis")
        
        # SYMBOL SECTION
        TradingAnimator.loading_bar("Generating symbol analysis", 0.5)
        plan_parts.append(f"""
📊 SYMBOL: {trade_signal['symbol']}
🏷️ ASSET TYPE: {category_info['description']} ({trade_signal['asset_category']})
💼 ACCOUNT BALANCE: ${trade_signal['account_balance']:,.2f}""")
        
        # TRADE DETAILS SECTION
        TradingAnimator.loading_bar("Generating trade details", 0.5)
        plan_parts.append(f"""
⚡ TRADE DIRECTION: {trade_signal['direction']}
💰 ENTRY PRICE: ${trade_signal['entry_price']:,.2f}
🛡️ STOP LOSS: ${trade_signal['stop_loss']:,.2f} ({stop_distance:+.1f}%)
🎯 TAKE PROFITS:""")
        
        for i, tp in enumerate(trade_signal['take_profits'], 1):
            if i == 1: distance = tp1_distance
            elif i == 2: distance = tp2_distance
            else: distance = tp3_distance
            plan_parts.append(f"\n   TP{i}: ${tp:,.2f} ({distance:+.1f}%)")
        
        # POSITION SIZING SECTION
        TradingAnimator.loading_bar("Generating position sizing", 0.5)
        plan_parts.append(f"""
📈 POSITION SIZING:
   Units: {trade_signal['position_size']:.6f}
   Position Value: ${trade_signal['position_value']:,.2f}
   Margin Required: ${trade_signal['margin_required']:,.2f}
   Leverage: {trade_signal['leverage']:.1f}x
   Liquidation Price: ${trade_signal['liquidation_price']:,.2f} ({liquidation_distance:+.1f}% from entry)""")
        
        # RISK MANAGEMENT SECTION
        TradingAnimator.loading_bar("Generating risk management", 0.5)
        risk_color = "\033[92m" if trade_signal['buffer_to_liquidation'] > 20 else "\033[93m" if trade_signal['buffer_to_liquidation'] > 10 else "\033[91m"
        risk_level = '🟢 SAFE' if trade_signal['buffer_to_liquidation'] > 20 else '🟡 MODERATE' if trade_signal['buffer_to_liquidation'] > 10 else '🔴 DANGEROUS'
        
        plan_parts.append(f"""
⚖️ RISK MANAGEMENT:
   Risk per Trade: {self.risk_per_trade*100}% of account
   Actual Risk: ${trade_signal['risk_amount']:,.2f} ({trade_signal['risk_percentage']:.1f}% of account) {trade_signal['risk_status']}
   R:R Ratio: {trade_signal['risk_reward_ratio']:.2f}:1
   Signal Strength: {trade_signal['signal_strength']}/10
   Buffer to Liquidation: {trade_signal['buffer_to_liquidation']:.1f}%
   Risk Level: {risk_level}""")
        
        # Show volatility warning if present
        if 'volatility_warning' in category_info:
            plan_parts.append(f"\n⚠️  VOLATILITY WARNING: {category_info['volatility_warning']}")
        elif 'current_atr' in category_info:
            plan_parts.append(f"\n📊 CURRENT VOLATILITY: {category_info['current_atr']}")
        
        # LEVERAGE RECOMMENDATIONS SECTION
        TradingAnimator.loading_bar("Generating leverage recommendations", 0.5)
        plan_parts.append(f"""
🎯 LEVERAGE RECOMMENDATIONS for {trade_signal['symbol'].split('/')[0]}:
   🟢 CONSERVATIVE: {category_info['conservative']:.1f}x (Beginner, low risk)
   🟡 BALANCED: {category_info['balanced']:.1f}x (Recommended, moderate risk)
   🟠 AGGRESSIVE: {category_info['aggressive']:.1f}x (Experienced, high risk)
   🔴 MAX SAFE: {category_info['max_safe']:.1f}x (Expert only, very high risk)
   ⚠️  AVOID: >{category_info['max_safe']:.1f}x (Extreme liquidation risk)""")
        
        # Combine all parts
        plan = "".join(plan_parts)
        
        # Animate final plan display
        TradingAnimator.flash_text("✅ TRADING PLAN COMPLETE!", 2, "\033[92m")
        
        return plan

    def execute_spot_trading(self, trade_signal):
        """Execute spot trading (SHORT not available in spot)"""
        if trade_signal['direction'] == 'LONG':
            return self.spot_long_execution(trade_signal)
        else:
            return "❌ SHORT positions not available in spot trading. Use futures."

    def execute_futures_trading(self, trade_signal):
        """Execute futures trading for BOTH LONG and SHORT"""
        if trade_signal['direction'] == 'LONG':
            return self.futures_long_execution(trade_signal)
        else:
            return self.futures_short_execution(trade_signal)

    def spot_long_execution(self, trade_signal):
        """Spot long execution details"""
        execution = f"""
🟢 SPOT LONG EXECUTION:

1. ORDER TYPE: Limit Buy
2. SYMBOL: {trade_signal['symbol'].replace('/USDT', '')}
3. QUANTITY: {trade_signal['position_size']:.6f}
4. PRICE: ${trade_signal['entry_price']:,.4f}
5. TOTAL COST: ${trade_signal['position_value']:,.2f}
6. ACCOUNT USED: {(trade_signal['position_value']/trade_signal['account_balance'])*100:.1f}%

RISK MANAGEMENT:
- Stop Loss: ${trade_signal['stop_loss']:,.4f}
- Risk Amount: ${trade_signal['risk_amount']:,.2f} ({trade_signal['risk_percentage']:.1f}% of account) {trade_signal['risk_status']}
- Take Profit 1: ${trade_signal['take_profits'][0]:,.4f} (Sell 30%)
- Take Profit 2: ${trade_signal['take_profits'][1]:,.4f} (Sell 30%)
- Take Profit 3: ${trade_signal['take_profits'][2]:,.4f} (Sell 40%)
"""
        return execution

    def futures_long_execution(self, trade_signal):
        """Futures long execution details"""
        # Get category info for warnings
        category_info = trade_signal['leverage_recommendations']
        
        execution = f"""
🟢 FUTURES LONG EXECUTION:

1. ORDER TYPE: Limit Buy
2. SYMBOL: {trade_signal['symbol']} Perpetual
3. QUANTITY: {trade_signal['position_size']:.6f}
4. ENTRY PRICE: ${trade_signal['entry_price']:,.4f}
5. LEVERAGE: {trade_signal['leverage']:.1f}x ({'BALANCED' if trade_signal['leverage'] == category_info['balanced'] else 'AGGRESSIVE' if trade_signal['leverage'] == category_info['aggressive'] else 'CONSERVATIVE'})
6. MARGIN: ${trade_signal['margin_required']:,.2f}
7. ACCOUNT USAGE: {(trade_signal['margin_required']/trade_signal['account_balance'])*100:.1f}%

RISK MANAGEMENT:
- Stop Loss: ${trade_signal['stop_loss']:,.4f} ({((trade_signal['entry_price'] - trade_signal['stop_loss'])/trade_signal['entry_price'])*100:.1f}% down)
- Risk Amount: ${trade_signal['risk_amount']:,.2f} ({trade_signal['risk_percentage']:.1f}% of account) {trade_signal['risk_status']}
- Take Profit 1: ${trade_signal['take_profits'][0]:,.4f} (Close 30%)
- Take Profit 2: ${trade_signal['take_profits'][1]:,.4f} (Close 30%)
- Take Profit 3: ${trade_signal['take_profits'][2]:,.4f} (Close 40%)

LIQUIDATION ANALYSIS:
- Liquidation Price: ${trade_signal['liquidation_price']:,.4f} ({((trade_signal['entry_price'] - trade_signal['liquidation_price'])/trade_signal['entry_price'])*100:.1f}% down)
- Stop to Liquidation Buffer: {trade_signal['buffer_to_liquidation']:.1f}%
- Risk Level: {'🟢 LOW' if trade_signal['buffer_to_liquidation'] > 20 else '🟡 MODERATE' if trade_signal['buffer_to_liquidation'] > 10 else '🟠 HIGH' if trade_signal['buffer_to_liquidation'] > 5 else '🔴 DANGEROUS'}

💡 LEVERAGE RECOMMENDATIONS:
   • For {trade_signal['symbol'].split('/')[0]} ({category_info['description']}):
   • Safe: {category_info['conservative']:.1f}x
   • Recommended: {category_info['balanced']:.1f}x  
   • Max Safe: {category_info['max_safe']:.1f}x
   • Avoid: >{category_info['max_safe']:.1f}x
"""
        
        # Add specific warnings for high leverage
        if trade_signal['leverage'] > category_info['max_safe']:
            execution += f"\n⚠️  WARNING: Current leverage ({trade_signal['leverage']:.1f}x) exceeds max safe ({category_info['max_safe']:.1f}x)!"
            execution += "\n   Consider reducing to recommended levels."
        
        return execution

    def futures_short_execution(self, trade_signal):
        """Futures SHORT execution details"""
        # Get category info for warnings
        category_info = trade_signal['leverage_recommendations']
        
        # For shorts, use even more conservative recommendations
        short_safe_leverage = max(1, category_info['conservative'] * 0.7)  # 30% less for shorts
        short_max_leverage = max(2, category_info['max_safe'] * 0.5)  # 50% less for shorts
        
        execution = f"""
🔴 FUTURES SHORT EXECUTION:

1. ORDER TYPE: Limit Sell
2. SYMBOL: {trade_signal['symbol']} Perpetual
3. QUANTITY: {trade_signal['position_size']:.6f}
4. ENTRY PRICE: ${trade_signal['entry_price']:,.4f}
5. LEVERAGE: {trade_signal['leverage']:.1f}x
6. MARGIN: ${trade_signal['margin_required']:,.2f}
7. ACCOUNT USAGE: {(trade_signal['margin_required']/trade_signal['account_balance'])*100:.1f}%

RISK MANAGEMENT:
- Stop Loss: ${trade_signal['stop_loss']:,.4f} ({((trade_signal['stop_loss'] - trade_signal['entry_price'])/trade_signal['entry_price'])*100:.1f}% up)
- Risk Amount: ${trade_signal['risk_amount']:,.2f} ({trade_signal['risk_percentage']:.1f}% of account) {trade_signal['risk_status']}
- Take Profit 1: ${trade_signal['take_profits'][0]:,.4f} (Close 30%)
- Take Profit 2: ${trade_signal['take_profits'][1]:,.4f} (Close 30%)
- Take Profit 3: ${trade_signal['take_profits'][2]:,.4f} (Close 40%)

LIQUIDATION ANALYSIS:
- Liquidation Price: ${trade_signal['liquidation_price']:,.4f} ({((trade_signal['liquidation_price'] - trade_signal['entry_price'])/trade_signal['entry_price'])*100:.1f}% up)
- Stop to Liquidation Buffer: {trade_signal['buffer_to_liquidation']:.1f}%
- Risk Level: {'🟢 LOW' if trade_signal['buffer_to_liquidation'] > 20 else '🟡 MODERATE' if trade_signal['buffer_to_liquidation'] > 10 else '🟠 HIGH' if trade_signal['buffer_to_liquidation'] > 5 else '🔴 DANGEROUS'}

📉 SHORT-SPECIFIC LEVERAGE GUIDELINES:
   • SHORTS REQUIRE MORE CAUTION due to short squeezes
   • Recommended for {trade_signal['symbol'].split('/')[0]} shorts:
   • Very Safe: {short_safe_leverage:.1f}x
   • Balanced: {max(short_safe_leverage * 1.5, 2):.1f}x
   • Max Safe for Shorts: {short_max_leverage:.1f}x
   • AVOID: >{short_max_leverage:.1f}x for shorts

📊 SHORT TRADING TIPS:
1. Check funding rates before entering (ideal: negative or <0.005%)
2. Use lower leverage than long positions
3. Consider wider stops for volatile assets
4. Monitor news for potential squeeze catalysts
5. Consider partial profits at TP1 to reduce risk
"""
        
        # Warning for high leverage on shorts
        if trade_signal['leverage'] > short_max_leverage:
            execution += f"\n⚠️  CRITICAL WARNING: Short leverage ({trade_signal['leverage']:.1f}x) too high!"
            execution += f"\n   Maximum safe for shorts: {short_max_leverage:.1f}x"
            execution += "\n   Strongly consider reducing leverage"
        
        return execution

class MLEnhancedTradingSystem(TradingExecutionSystem):
    def __init__(self, account_balance=1000):
        super().__init__(account_balance)
        self.ml_predictor = RealisticMLPredictor()
        self.ml_trained = False
        self.training_symbol = None  # Track which symbol we trained on
        
    def train_ml_models(self, price_data, symbol=None):
        """Train ML models with animation"""
        if not ML_AVAILABLE:
            TradingAnimator.flash_text("❌ ML dependencies not installed", 2, "\033[91m")
            return False
            
        TradingAnimator.animate_title("🤖 MACHINE LEARNING TRAINING", 0.03)
        
        # If we already trained on this symbol recently, skip
        if (self.ml_trained and self.training_symbol == symbol and 
            hasattr(self.ml_predictor, 'last_training_time')):
            time_since_training = (datetime.now() - self.ml_predictor.last_training_time).total_seconds()
            if time_since_training < 300:
                TradingAnimator.flash_text(f"✅ ML already trained ({time_since_training:.0f}s ago)", 1, "\033[92m")
                return True
        
        TradingAnimator.loading_bar("Preparing training data", 1.0)
        TradingAnimator.loading_bar("Training ML models", 2.0)
        TradingAnimator.loading_bar("Validating model accuracy", 1.0)
        
        success = self.ml_predictor.train_simple_model(price_data)
        self.ml_trained = success
        if symbol:
            self.training_symbol = symbol
        if success:
            TradingAnimator.flash_text("✅ ML MODELS TRAINED SUCCESSFULLY!", 3, "\033[92m")
        else:
            TradingAnimator.flash_text("❌ ML TRAINING FAILED", 2, "\033[91m")
        return success
    
    def generate_enhanced_signal(self, ta_signals, onchain_signals, price_data):
        """Generate signal with ML enhancement"""
        TradingAnimator.animate_title("🤖 ML-ENHANCED SIGNAL GENERATION", 0.03)
        
        traditional_score = self.score_signals(ta_signals, onchain_signals)
        print(f"🔍 TRADITIONAL SCORE: {traditional_score}")

        ml_boost = 0
        ml_signals = []
        
        if self.ml_trained and ML_AVAILABLE:
            try:
                TradingAnimator.loading_bar("Running ML predictions", 1.5)
                print("🤖 GENERATING SIMPLE ML PREDICTIONS...")
                predictions, confidence_scores = self.ml_predictor.predict_simple(price_data)
                current_price = price_data['close'].iloc[-1]
                
                print(f"🤖 PREDICTIONS: {predictions}")
                print(f"🤖 CONFIDENCE: {confidence_scores}")
                
                ml_signals, ml_boost = self.ml_predictor.generate_ml_signals(
                    predictions, confidence_scores, current_price
                )

                print(f"🤖 ML SIGNALS: {ml_signals}")
                print(f"🤖 ML BOOST: {ml_boost}")
                
                self.ml_predictor.display_predictions(predictions, confidence_scores, ml_signals)
                
                # ADD CLEAR LABELS FOR ML PREDICTIONS
                print(f"\n📊 VERIFICATION: Current actual price: ${current_price:,.2f}")
                print(f"📊 ML is showing PREDICTED prices, not current price")
                
            except Exception as e:
                print(f"⚠️  ML Prediction Error: {e}")
                import traceback
                traceback.print_exc()
        
        combined_score = traditional_score + ml_boost
        TradingAnimator.flash_text(f"🎯 FINAL COMBINED SCORE: {traditional_score} + {ml_boost} = {combined_score}", 2, "\033[95m")
        
        return self.determine_trade_direction(combined_score, ta_signals), ml_signals, combined_score

def integrate_and_trade_with_ml(symbol='BTC/USDT', timeframe='4h', account_balance=1000, enable_ml=True, custom_leverage=None):
    """Main function with ML enhancement - WITH ANIMATIONS"""
    
    # CAPTURE START TIME
    analysis_start_time = datetime.now()
    
    trading_system = MLEnhancedTradingSystem(account_balance)
    
    TradingAnimator.animate_title("🚀 ADVANCED TRADING ANALYSIS ENGINE", 0.02)
    
    # Add timestamp to header
    print(f"🕒 ANALYSIS STARTED: {analysis_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 RANDOM SEED: {SEED} (Ensures reproducibility)")
    print(f"📊 ML TRAINING DATA LENGTH: {ML_TRAINING_DATA_LENGTH} candles")
    
    # Get data WITH CACHING - USE CONSISTENT DATA LENGTH
    TradingAnimator.animate_title(f"🔍 ANALYZING {symbol}", 0.03)
    TradingAnimator.loading_bar("Fetching market data", 1.5)
    
    price_data = get_cached_or_fetch_price_data(symbol, timeframe, ML_TRAINING_DATA_LENGTH)
    if price_data is None:
        TradingAnimator.flash_text(f"❌ Could not fetch price data for {symbol}", 2, "\033[91m")
        return
    
    # CAPTURE DATA FETCH TIME
    data_fetch_time = datetime.now()
    
    # Display data timestamp
    print(f"📊 DATA FETCHED: {data_fetch_time.strftime('%H:%M:%S')}")
    print(f"   Price: ${price_data['close'].iloc[-1]:,.2f}")
    print(f"   Data points: {len(price_data)}")
    
    # Calculate time difference
    fetch_duration = (data_fetch_time - analysis_start_time).total_seconds()
    print(f"   Fetch time: {fetch_duration:.1f} seconds")
    
    TradingAnimator.loading_bar("Calculating technical indicators", 1.0)
    price_data = calculate_technical_indicators(price_data)
    ta_signals = generate_trading_signals(price_data)
    
    # Extract technical indicators for market analysis
    current_price = price_data['close'].iloc[-1]
    
    # Check available columns
    available_indicators = []
    
    # Check for common indicators
    if 'MA20' in price_data.columns and not pd.isna(price_data['MA20'].iloc[-1]):
        ma20 = price_data['MA20'].iloc[-1]
        ma20_diff = ((ma20 - current_price) / current_price) * 100
        available_indicators.append(('MA20', ma20, ma20_diff))
    
    if 'MA50' in price_data.columns and not pd.isna(price_data['MA50'].iloc[-1]):
        ma50 = price_data['MA50'].iloc[-1]
        ma50_diff = ((ma50 - current_price) / current_price) * 100
        available_indicators.append(('MA50', ma50, ma50_diff))
    
    if 'MA200' in price_data.columns and not pd.isna(price_data['MA200'].iloc[-1]):
        ma200 = price_data['MA200'].iloc[-1]
        ma200_diff = ((ma200 - current_price) / current_price) * 100
        available_indicators.append(('MA200', ma200, ma200_diff))
    
    # Get RSI
    if 'RSI' in price_data.columns and not pd.isna(price_data['RSI'].iloc[-1]):
        rsi = price_data['RSI'].iloc[-1]
        rsi_status = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
        available_indicators.append(('RSI', rsi, rsi_status))
    
    # Get MACD
    if 'MACD' in price_data.columns and not pd.isna(price_data['MACD'].iloc[-1]):
        macd = price_data['MACD'].iloc[-1]
        macd_signal = price_data['MACD_signal'].iloc[-1] if 'MACD_signal' in price_data.columns and not pd.isna(price_data['MACD_signal'].iloc[-1]) else 0
        macd_status = "BULLISH" if macd > macd_signal else "BEARISH"
        available_indicators.append(('MACD', macd, macd_status))
    
    # Get Bollinger Bands
    if 'BB_upper' in price_data.columns and not pd.isna(price_data['BB_upper'].iloc[-1]):
        bb_upper = price_data['BB_upper'].iloc[-1]
        bb_lower = price_data['BB_lower'].iloc[-1] if 'BB_lower' in price_data.columns else None
        bb_position = "UPPER" if current_price > bb_upper else "LOWER" if bb_lower and current_price < bb_lower else "MIDDLE"
        available_indicators.append(('BB Position', bb_position, None))
    
    # Get support and resistance
    support = price_data['support'].iloc[-1] if 'support' in price_data.columns and not pd.isna(price_data['support'].iloc[-1]) else None
    resistance = price_data['resistance'].iloc[-1] if 'resistance' in price_data.columns and not pd.isna(price_data['resistance'].iloc[-1]) else None
    
    if support is not None:
        support_diff = ((current_price - support) / current_price) * 100
        available_indicators.append(('Support', support, support_diff))
        
    if resistance is not None:
        resistance_diff = ((resistance - current_price) / current_price) * 100
        available_indicators.append(('Resistance', resistance, resistance_diff))
    
    # Get ATR for volatility
    if 'atr' in price_data.columns and not pd.isna(price_data['atr'].iloc[-1]):
        atr = price_data['atr'].iloc[-1]
        atr_percentage = (atr / current_price) * 100
        available_indicators.append(('ATR', atr_percentage, "% daily volatility"))
    
    # On-chain data WITH CACHING
    base_symbol = symbol.split('/')[0]
    onchain_analyzer = OnChainAnalyzer()
    
    # Check cache first
    cached_onchain = get_cached_onchain_data(base_symbol)
    if cached_onchain is not None:
        all_metrics = cached_onchain
        print(f"   ⛓️ Using cached on-chain data")
    else:
        # Fetch fresh data
        TradingAnimator.loading_bar("Fetching on-chain data", 1.0)
        onchain_analyzer.accelerate_data_collection(base_symbol)
        all_metrics = onchain_analyzer.get_comprehensive_onchain_analysis(base_symbol)
        # Cache the result
        cache_onchain_data(base_symbol, all_metrics)
    
    onchain_signals = onchain_analyzer.analyze_comprehensive_signals(all_metrics)
    
    # Train ML (first run) - PASS SYMBOL FOR TRACKING
    if enable_ml and ML_AVAILABLE and not trading_system.ml_trained:
        TradingAnimator.animate_title(f"🤖 TRAINING ML FOR {symbol}", 0.03)
        trading_system.train_ml_models(price_data, symbol)
    
    # Generate enhanced signal
    trade_signal = None
    if enable_ml and ML_AVAILABLE and trading_system.ml_trained:
        trade_direction, ml_signals, combined_score = trading_system.generate_enhanced_signal(
            ta_signals, onchain_signals, price_data
        )
        print(f"\n🎯 COMBINED SCORE: {combined_score} (Traditional + ML)")
        
        if trade_direction != "NO_TRADE":
            trade_signal = trading_system.generate_trading_signal(ta_signals, onchain_signals, price_data, symbol, custom_leverage)
            if trade_signal:
                trade_signal['combined_score'] = combined_score
                trade_signal['ml_signals'] = ml_signals
        else:
            trade_signal = None
    else:
        print("\n🔍 USING TRADITIONAL ANALYSIS ONLY")
        trade_signal = trading_system.generate_trading_signal(ta_signals, onchain_signals, price_data, symbol, custom_leverage)
    
    # Display analysis with improved formatting
    TradingAnimator.animate_title("📊 MARKET ANALYSIS SUMMARY", 0.02)
    
    print(f"\n📊 TECHNICAL INDICATORS:")
    print(f"   └─ CURRENT PRICE: ${current_price:,.2f}")
    
    # Display all available indicators
    for indicator_name, value, diff in available_indicators:
        if indicator_name in ['MA20', 'MA50', 'MA200']:
            direction = "above" if diff > 0 else "below"
            print(f"   └─ {indicator_name}: ${value:,.2f} ({direction} {abs(diff):.1f}%)")
        elif indicator_name == 'RSI':
            print(f"   └─ RSI: {value:.1f} ({diff})")
        elif indicator_name == 'MACD':
            print(f"   └─ MACD: {value:.2f} ({diff})")
        elif indicator_name == 'BB Position':
            print(f"   └─ Bollinger Band: {value}")
        elif indicator_name == 'Support':
            print(f"   └─ Support: ${value:,.2f} ({abs(diff):.1f}% below)")
        elif indicator_name == 'Resistance':
            print(f"   └─ Resistance: ${value:,.2f} ({abs(diff):.1f}% above)")
        elif indicator_name == 'ATR':
            print(f"   └─ ATR (Volatility): {value:.1f}% {diff}")
    
    print(f"\n🎯 MARKET SIGNALS ANALYSIS:")
    for i, signal in enumerate(ta_signals[:6], 1):  # Show first 6 signals
        # Determine emoji based on signal content
        if any(word in signal for word in ['BULLISH', 'UPTREND', 'OVERSOLD', 'SUPPORT']):
            emoji = "📗"
        elif any(word in signal for word in ['BEARISH', 'DOWNTREND', 'OVERBOUGHT', 'RESISTANCE']):
            emoji = "📕"
        elif 'VOLATILITY' in signal:
            emoji = "⚡"
        elif 'VOLUME' in signal:
            if 'LOW' in signal:
                emoji = "📉"
            else:
                emoji = "📈"
        else:
            emoji = "📊"
        
        print(f"    {i}. {emoji} {signal}")
    
    if len(ta_signals) > 6:
        print(f"    ... and {len(ta_signals) - 6} more signals")
    
    print(f"\n⛓️  ON-CHAIN SIGNALS:")
    for i, signal in enumerate(onchain_signals, 1):
        print(f"    {i}. {signal}")
    
    if trade_signal:
        trading_plan = trading_system.generate_trading_plan(trade_signal, show_scenarios=True)
        print(trading_plan)
        
        TradingAnimator.animate_title("⚡ EXECUTION DETAILS", 0.02)
        
        spot_execution = trading_system.execute_spot_trading(trade_signal)
        print(f"\n🪙 SPOT TRADING:")
        print(spot_execution)
        
        futures_execution = trading_system.execute_futures_trading(trade_signal)
        print(f"\n📈 FUTURES TRADING:")
        print(futures_execution)
        
        TradingAnimator.animate_title("🔧 LEVERAGE IMPACT ANALYSIS", 0.02)
        
        print(f"\n📈 POSITION VALUE: ${trade_signal['position_value']:,.2f}")
        print("If you change leverage (ALL scenarios risk exactly 2% on stop loss):")
        print("─" * 80)
        
        # Show MORE scenarios including 2x, 5x, 8x, 15x
        scenarios_to_show = [
            1,                          # No leverage
            trade_signal['leverage_recommendations']['conservative'],  # Conservative
            trade_signal['leverage_recommendations']['balanced'],      # Balanced (current)
            trade_signal['leverage_recommendations']['aggressive'],    # Aggressive
            trade_signal['leverage_recommendations']['max_safe'],      # Max safe
            5,                          # Common moderate
            8,                          # Common high
            10,                         # Very high
            15,                         # Extreme
            20,                         # Dangerous
            25                          # Maximum
        ]
        
        # Remove duplicates and sort
        scenarios_to_show = sorted(list(set([round(x, 1) for x in scenarios_to_show])))
        
        for leverage in scenarios_to_show:
            if leverage > 25:  # Skip anything above 25x
                continue
                
            margin_needed = trade_signal['position_value'] / leverage
            
            # CORRECTED: Liquidation calculation for scenarios
            maintenance_margin_rate = 0.05
            initial_margin_rate = 1.0 / leverage
            
            if trade_signal['direction'] == 'LONG':
                liq_price = trade_signal['entry_price'] * (1 - (initial_margin_rate - maintenance_margin_rate))
                liq_distance = ((trade_signal['entry_price'] - liq_price) / trade_signal['entry_price']) * 100
                stop_distance = ((trade_signal['entry_price'] - trade_signal['stop_loss']) / trade_signal['entry_price']) * 100
            else:
                liq_price = trade_signal['entry_price'] * (1 + (initial_margin_rate - maintenance_margin_rate))
                liq_distance = ((liq_price - trade_signal['entry_price']) / trade_signal['entry_price']) * 100
                stop_distance = ((trade_signal['stop_loss'] - trade_signal['entry_price']) / trade_signal['entry_price']) * 100
            
            buffer = liq_distance - stop_distance
            
            # SAFETY CHECK
            if trade_signal['direction'] == 'LONG' and liq_price > trade_signal['entry_price']:
                liq_price = trade_signal['entry_price'] * 0.85
                liq_distance = ((trade_signal['entry_price'] - liq_price) / trade_signal['entry_price']) * 100
                buffer = liq_distance - stop_distance
            elif trade_signal['direction'] == 'SHORT' and liq_price < trade_signal['entry_price']:
                liq_price = trade_signal['entry_price'] * 1.15
                liq_distance = ((liq_price - trade_signal['entry_price']) / trade_signal['entry_price']) * 100
                buffer = liq_distance - stop_distance
            
            # Determine label
            if leverage == trade_signal['leverage']:
                indicator = "👉 CURRENT"
            elif leverage == 1:
                indicator = "🛡️  NO LEVERAGE"
            elif leverage == trade_signal['leverage_recommendations']['conservative']:
                indicator = "🟢 CONSERVATIVE"
            elif leverage == trade_signal['leverage_recommendations']['balanced']:
                indicator = "🟡 BALANCED"
            elif leverage == trade_signal['leverage_recommendations']['aggressive']:
                indicator = "🟠 AGGRESSIVE"
            elif leverage == trade_signal['leverage_recommendations']['max_safe']:
                indicator = "🔴 MAX SAFE"
            elif leverage <= 5:
                indicator = "⬇️  SAFER"
            elif leverage <= 10:
                indicator = "⬆️  RISKIER"
            else:
                indicator = "💀 EXTREME"
            
            print(f"{indicator} {leverage:>2}x leverage:")
            print(f"   • Margin needed: ${margin_needed:,.0f}")
            print(f"   • Liquidation: ${liq_price:,.0f} ({liq_distance:.1f}% {'down' if trade_signal['direction'] == 'LONG' else 'up'})")
            print(f"   • Buffer to stop: {buffer:.1f}%")
            print(f"   • Risk level: {'🟢 LOW' if buffer > 50 else '🟡 MODERATE' if buffer > 30 else '🟠 HIGH' if buffer > 15 else '🔴 DANGEROUS' if buffer > 5 else '💀 EXTREME'}")
            print(f"   • Stop loss risk: ${trade_signal['risk_amount']:,.2f} ({trade_signal['risk_percentage']:.1f}% of account) {trade_signal['risk_status']}")
            print()
        
    else:
        TradingAnimator.flash_text(f"\n❌ NO TRADING OPPORTUNITY FOUND", 2, "\033[91m")
        print(f"   Signal strength insufficient for {symbol}")
        print(f"   Waiting for better setup...")
    
    # At the end, add completion timestamp
    analysis_end_time = datetime.now()
    total_duration = (analysis_end_time - analysis_start_time).total_seconds()
    
    TradingAnimator.animate_title("📊 ANALYSIS COMPLETED", 0.02)
    print(f"📊 ANALYSIS COMPLETED: {analysis_end_time.strftime('%H:%M:%S')}")
    print(f"📊 TOTAL DURATION: {total_duration:.1f} seconds")

def batch_analyze_cryptos(cryptos=None, account_balance=1000, enable_ml=False):
    """Analyze multiple cryptocurrencies with SEPARATE ML models for each"""
    if cryptos is None:
        cryptos = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT']
    
    TradingAnimator.animate_title("🔍 MULTI-CRYPTO SCANNING MODE", 0.02)
    
    # SCAN START TIME
    scan_start_time = datetime.now()
    print(f"🕒 SCAN STARTED: {scan_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 CRYPTOS TO ANALYZE: {len(cryptos)}")
    print(f"📊 ACCOUNT BALANCE: ${account_balance:,.2f}")
    print(f"🎯 RANDOM SEED: {SEED} (Ensures reproducibility)")
    print(f"📊 ML TRAINING DATA LENGTH: {ML_TRAINING_DATA_LENGTH} candles")
    
    # We'll create separate trading systems for each crypto with ML
    best_opportunities = []
    
    for i, crypto in enumerate(cryptos, 1):
        crypto_start_time = datetime.now()
        
        TradingAnimator.animate_title(f"📈 [{i}/{len(cryptos)}] Analyzing {crypto}", 0.02)
        print(f"   🕒 START: {crypto_start_time.strftime('%H:%M:%S')}")
        
        try:
            # USE CACHED DATA FOR CONSISTENCY - SAME DATA LENGTH
            price_data = get_cached_or_fetch_price_data(crypto, '4h', ML_TRAINING_DATA_LENGTH)
            if price_data is None:
                TradingAnimator.flash_text(f"   ❌ Could not fetch price data for {crypto}", 1, "\033[91m")
                continue
            
            # Create SEPARATE trading system for this crypto
            if enable_ml and ML_AVAILABLE:
                TradingAnimator.loading_bar(f"   Creating ML system for {crypto}", 0.5)
                trading_system = MLEnhancedTradingSystem(account_balance=account_balance)
            else:
                trading_system = TradingExecutionSystem(account_balance=account_balance)
            
            TradingAnimator.loading_bar("   Calculating indicators", 0.5)
            price_data = calculate_technical_indicators(price_data)
            ta_signals = generate_trading_signals(price_data)
            
            # DEBUG: Show key technical indicators
            current_price = price_data['close'].iloc[-1]
            print(f"   📊 Price: ${current_price:,.2f}")
            print(f"   📊 Data points: {len(price_data)}")
            if 'RSI' in price_data.columns:
                rsi = price_data['RSI'].iloc[-1]
                rsi_status = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
                print(f"   📊 RSI: {rsi:.1f} ({rsi_status})")
            
            base_symbol = crypto.split('/')[0]
            onchain_analyzer = OnChainAnalyzer()
            
            # Check cache first
            cached_onchain = get_cached_onchain_data(base_symbol)
            if cached_onchain is not None:
                all_metrics = cached_onchain
                print(f"   ⛓️ Using cached on-chain data")
            else:
                # Fetch fresh data
                TradingAnimator.loading_bar("   Fetching on-chain data", 0.5)
                onchain_analyzer.accelerate_data_collection(base_symbol)
                all_metrics = onchain_analyzer.get_comprehensive_onchain_analysis(base_symbol)
                # Cache the result
                cache_onchain_data(base_symbol, all_metrics)
            
            onchain_signals = onchain_analyzer.analyze_comprehensive_signals(all_metrics)
            
            # FIXED: Train ML SEPARATELY for each crypto
            if enable_ml and ML_AVAILABLE and isinstance(trading_system, MLEnhancedTradingSystem):
                TradingAnimator.loading_bar(f"   Training ML for {crypto}", 1.0)
                trading_system.train_ml_models(price_data, crypto)
            
            # Generate trade signal
            if enable_ml and ML_AVAILABLE and isinstance(trading_system, MLEnhancedTradingSystem):
                # Use ML-enhanced signal generation
                trade_direction, ml_signals, combined_score = trading_system.generate_enhanced_signal(
                    ta_signals, onchain_signals, price_data
                )
                
                if trade_direction != "NO_TRADE":
                    trade_signal = trading_system.generate_trading_signal(
                        ta_signals, onchain_signals, price_data, crypto
                    )
                    if trade_signal:
                        trade_signal['combined_score'] = combined_score
                        trade_signal['ml_signals'] = ml_signals
                        trade_signal['ml_trained_on'] = crypto  # Track which crypto ML was trained on
                else:
                    trade_signal = None
            else:
                # Use traditional signal generation
                trade_signal = trading_system.generate_trading_signal(
                    ta_signals, onchain_signals, price_data, crypto
                )
            
            crypto_end_time = datetime.now()
            crypto_duration = (crypto_end_time - crypto_start_time).total_seconds()
            
            print(f"   🕒 END: {crypto_end_time.strftime('%H:%M:%S')}")
            print(f"   ⏱️  DURATION: {crypto_duration:.1f}s")
            
            if trade_signal:
                # ADD TIMESTAMP TO TRADE SIGNAL
                trade_signal['analysis_timestamp'] = crypto_end_time
                trade_signal['analysis_duration'] = crypto_duration
                trade_signal['price_at_analysis'] = current_price
                
                best_opportunities.append(trade_signal)
                direction_emoji = "🟢" if trade_signal['direction'] == "LONG" else "🔴"
                score_to_show = trade_signal.get('combined_score', trade_signal['signal_strength'])
                TradingAnimator.flash_text(f"   {direction_emoji} TRADE FOUND: {trade_signal['direction']} - Score: {score_to_show}", 1, "\033[92m")
                print(f"   ⏰ Analyzed: {trade_signal['analysis_timestamp'].strftime('%H:%M:%S')}")
                
                # Show if ML was used
                if 'ml_trained_on' in trade_signal:
                    print(f"   🤖 ML trained on: {trade_signal['ml_trained_on']}")
            else:
                TradingAnimator.flash_text(f"   ❌ No trade signal", 1, "\033[93m")
                
        except Exception as e:
            crypto_end_time = datetime.now()
            crypto_duration = (crypto_end_time - crypto_start_time).total_seconds()
            TradingAnimator.flash_text(f"   ⚡ Error analyzing {crypto}: {e}", 1, "\033[91m")
            print(f"   🕒 END: {crypto_end_time.strftime('%H:%M:%S')}")
            print(f"   ⏱️  DURATION: {crypto_duration:.1f}s")
    
    # Display best opportunities WITH TIMESTAMPS
    if best_opportunities:
        scan_end_time = datetime.now()
        total_duration = (scan_end_time - scan_start_time).total_seconds()
        
        TradingAnimator.animate_title("🎯 BEST TRADING OPPORTUNITIES", 0.02)
        print(f"🕒 SCAN COMPLETED: {scan_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  TOTAL SCAN TIME: {total_duration:.1f} seconds")
        
        # Sort by best score
        def get_best_score(trade):
            return trade.get('combined_score', trade['signal_strength'])
        
        best_opportunities.sort(key=lambda x: abs(get_best_score(x)), reverse=True)
        
        for i, opportunity in enumerate(best_opportunities[:3], 1):
            direction_emoji = "🟢" if opportunity['direction'] == "LONG" else "🔴"
            time_ago = (datetime.now() - opportunity['analysis_timestamp']).total_seconds()
            
            TradingAnimator.animate_title(f"#{i} {opportunity['symbol']} - {direction_emoji} {opportunity['direction']}", 0.02)
            print(f"📊 Analysis Time: {opportunity['analysis_timestamp'].strftime('%H:%M:%S')}")
            print(f"⏱️  Analysis Duration: {opportunity['analysis_duration']:.1f}s")
            print(f"🕒 Age: {time_ago:.0f} seconds ago")
            print(f"💰 Price then: ${opportunity['price_at_analysis']:,.2f}")
            
            # Show ML info if used
            if 'combined_score' in opportunity:
                print(f"🎯 ML-Enhanced Score: {opportunity['combined_score']} (Traditional: {opportunity['signal_strength']})")
                if 'ml_trained_on' in opportunity:
                    print(f"🤖 ML trained specifically on: {opportunity['ml_trained_on']}")
            else:
                print(f"🎯 Traditional Score: {opportunity['signal_strength']}")
                
            print(f"⚖️  R:R Ratio: {opportunity['risk_reward_ratio']:.2f}:1")
            print(f"💵 Risk Amount: ${opportunity['risk_amount']:,.2f} ({opportunity['risk_percentage']:.1f}% of account) {opportunity['risk_status']}")
            print(f"⚡ Leverage: {opportunity['leverage']}x")
            print(f"💼 Margin: ${opportunity['margin_required']:,.0f}")
            
            # Show ML signals if available
            if 'ml_signals' in opportunity and opportunity['ml_signals']:
                print(f"🤖 ML Signals: {', '.join(opportunity['ml_signals'][:3])}")
            
            # Check if price has changed significantly
            try:
                current_data = fetch_price_data(opportunity['symbol'], '1m', 2)
                if current_data is not None:
                    current_price = current_data['close'].iloc[-1]
                    price_change = ((current_price - opportunity['price_at_analysis']) / 
                                   opportunity['price_at_analysis']) * 100
                    print(f"📈 Current Price: ${current_price:,.2f}")
                    print(f"📊 Price Change: {price_change:+.2f}%")
                    
                    if abs(price_change) > 1.0:  # If price moved >1%
                        TradingAnimator.flash_text(f"⚠️  SIGNAL AGED: Price moved {price_change:+.2f}%", 1, "\033[93m")
                        print(f"💡 Consider re-analyzing {opportunity['symbol']}")
            except:
                pass
    else:
        scan_end_time = datetime.now()
        total_duration = (scan_end_time - scan_start_time).total_seconds()
        TradingAnimator.flash_text("❌ No strong trading opportunities found", 2, "\033[91m")
        print(f"🕒 SCAN COMPLETED: {scan_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  TOTAL SCAN TIME: {total_duration:.1f} seconds")

def test_scoring_consistency():
    """Test that scoring is consistent"""
    TradingAnimator.animate_title("🧪 SCORING CONSISTENCY TEST", 0.03)
    
    # Clear cache for fresh test
    global ONCHAIN_CACHE
    ONCHAIN_CACHE = {}
    
    # Create consistent test data
    test_ta_signals = [
        "🟡 RANGING (Mixed MA signals)",
        "🔴 RSI: OVERBOUGHT (Potential pullback)",
        "📗 MACD: BULLISH MOMENTUM",
        "📗 PRICE: UPPER BOLLINGER RANGE",
        "📉 LOW VOLUME: Weak interest",
        "🚧 NEAR STRONG RESISTANCE (Potential rejection)"
    ]
    
    test_onchain_signals = [
        "💰 Exchange Flow: 🟡 NEUTRAL: Balanced flows",
        "🏦 Exchange Balance: 🔴 BEARISH (High: 15,386,7",
        "🐋 Whale Ratio: ⚠️ CAUTION (High: 0.839)",
        "⛏️ Miner Flow: 🟢 BULLISH (Low selling: 0)",
        "📈 Funding Rate: ✅ NORMAL (0.2731%)"
    ]
    
    # Test with mock on-chain data for consistency
    from unittest.mock import Mock
    
    # Create mock on-chain analyzer
    mock_analyzer = Mock()
    mock_analyzer.analyze_comprehensive_signals.return_value = test_onchain_signals
    
    system = TradingExecutionSystem()
    
    print("\nRunning 5 identical scoring tests with same inputs...")
    scores = []
    
    for i in range(5):
        TradingAnimator.loading_bar(f"Test {i+1}", 0.3)
        score = system.score_signals(test_ta_signals, test_onchain_signals)
        scores.append(score)
        print(f"   Score: {score}")
    
    print(f"\n📊 Results: {scores}")
    
    # Check consistency
    unique_scores = set(scores)
    if len(unique_scores) == 1:
        TradingAnimator.flash_text(f"✅ PERFECT CONSISTENCY! All scores are identical: {scores[0]}", 2, "\033[92m")
        return True
    else:
        TradingAnimator.flash_text(f"❌ INCONSISTENT SCORING!", 2, "\033[91m")
        print(f"   Unique scores: {list(unique_scores)}")
        print(f"   Max difference: {max(scores) - min(scores)}")
        print(f"   Average score: {sum(scores)/len(scores):.2f}")
        
        # Show breakdown of differences
        for i, score in enumerate(scores):
            print(f"   Test {i+1}: {score}")
        
        return False

def test_final_consistency():
    """Test that ALL systems are now consistent"""
    TradingAnimator.animate_title("✅ FINAL CONSISTENCY VERIFICATION TEST", 0.03)
    
    # Clear cache
    price_data_cache.clear()
    global ONCHAIN_CACHE
    ONCHAIN_CACHE = {}
    
    # Run scoring consistency test first
    scoring_consistent = test_scoring_consistency()
    
    # Test signal matching logic
    TradingAnimator.animate_title("🔍 SIGNAL MATCHING TEST", 0.03)
    
    system = TradingExecutionSystem()
    
    test_cases = [
        ("MACD: BULLISH MOMENTUM", "Should match BULLISH MOMENTUM (+2)"),
        ("RSI: OVERBOUGHT (Potential pullback)", "Should match OVERBOUGHT (-2)"),
        ("PRICE: UPPER BOLLINGER RANGE", "Should match UPPER BOLLINGER (-1)"),
        ("LOW VOLUME: Weak interest", "Should match LOW VOLUME (-1)"),
        ("NEAR STRONG RESISTANCE (Potential rejection)", "Should match NEAR STRONG RESISTANCE (-2)"),
        ("🟡 RANGING (Mixed MA signals)", "Should match RANGING (0)"),
    ]
    
    print("\nTesting signal matching logic:")
    for signal, expected in test_cases:
        # Simulate scoring for single signal
        signal_score = 0
        matched_type = "No match"
        signal_upper = signal.upper()
        
        for signal_type, weight in system.signal_weights.items():
            signal_type_upper = signal_type.upper()
            if signal_type_upper in signal_upper:
                words = signal_type_upper.split()
                if len(words) > 1:
                    if all(word in signal_upper for word in words):
                        signal_score = weight
                        matched_type = signal_type
                        break
                else:
                    signal_score = weight
                    matched_type = signal_type
                    break
        
        color = "\033[92m" if signal_score > 0 else "\033[91m" if signal_score < 0 else "\033[93m"
        reset = "\033[0m"
        print(f"  {color}{signal[:40]:40} → Score: {signal_score:+2} ({matched_type}){reset}")
    
    return scoring_consistent

# Main execution
if __name__ == "__main__":
    TradingAnimator.clear_screen()
    
    # Animated startup sequence
    TradingAnimator.animate_title("🚀 CRYPTO TRADING EXECUTION SYSTEM v7.0", 0.03)
    print("\n" + "="*70)
    
    TradingAnimator.loading_bar("Initializing trading engine", 2.0)
    TradingAnimator.loading_bar("Loading ML modules", 1.5)
    TradingAnimator.loading_bar("Setting up risk management", 1.0)
    
    print("\n" + "="*70)
    TradingAnimator.flash_text("✅ SYSTEM READY FOR TRADING!", 3, "\033[92m")
    
    # Run tests with animations
    TradingAnimator.countdown(2, "Starting analysis tests")
    
    # Clear cache to start fresh
    price_data_cache.clear()
    ONCHAIN_CACHE = {}
    print("🧹 Cleared all caches for fresh analysis")
    
    # Run diagnostic test first
    TradingAnimator.animate_title("🔍 PRICE CONSISTENCY DIAGNOSTIC TEST", 0.03)
    
    # Test 1: Fetch ETH once
    print("Test 1: Fetching COIN data (100 candles)...")
    btc1 = get_cached_or_fetch_price_data('ETH/USDT', '4h', ML_TRAINING_DATA_LENGTH)
    if btc1 is not None:
        print(f"   Price 1: ${btc1['close'].iloc[-1]:,.2f}")
        
        # Test 2: Fetch ETH immediately again (should use cache)
        print(f"\nTest 2: Fetching BTC COIN again (should use cache)...")
        btc2 = get_cached_or_fetch_price_data('ETH/USDT', '4h', ML_TRAINING_DATA_LENGTH)
        if btc2 is not None:
            print(f"   Price 2: ${btc2['close'].iloc[-1]:,.2f}")
            
            # Check if they're the same
            price_diff = abs(btc1['close'].iloc[-1] - btc2['close'].iloc[-1])
            if price_diff < 0.01:  # Less than 1 cent difference
                TradingAnimator.flash_text(f"✅ CACHE WORKING: Prices are identical (difference: ${price_diff:.4f})", 2, "\033[92m")
            else:
                TradingAnimator.flash_text(f"⚠️  CACHE ISSUE: Prices differ by ${price_diff:.2f}", 2, "\033[93m")
    
    TradingAnimator.animate_title("🧪 TEST 1: SINGLE SYMBOL ANALYSIS WITH ML", 0.03)
    integrate_and_trade_with_ml('ETH/USDT', '4h', account_balance=1000, enable_ml=True)
    
    time.sleep(2)
    
    TradingAnimator.animate_title("🧪 TEST 2: BATCH ANALYSIS WITH ML (NOW SHOULD BE CONSISTENT)", 0.03)
    #batch_analyze_cryptos(['BTC/USDT', 'SOL/USDT'], enable_ml=True)
    batch_analyze_cryptos(['BTC/USDT'], enable_ml=True)
    
    time.sleep(2)
    
    TradingAnimator.animate_title("🧪 TEST 3: BATCH ANALYSIS WITHOUT ML FOR COMPARISON", 0.03)
    batch_analyze_cryptos(['BTC/USDT', 'ETH/USDT'], enable_ml=False)
    
    TradingAnimator.animate_title("✅ TRADING EXECUTION SYSTEM READY", 0.03)
    print(f"✅ Random Seed: {SEED} ensures reproducible results")
    print(f"✅ ML Training Data: {ML_TRAINING_DATA_LENGTH} candles (consistent)")
    print(f"✅ On-chain Data Cached: {ONCHAIN_CACHE_DURATION//60} minutes")
    print(f"✅ All systems operational!")