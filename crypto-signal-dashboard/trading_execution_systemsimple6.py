# trading_execution_systemsimple5_fixed_ml.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple
import math

# Import your existing modules
try:
    from c_signal import fetch_price_data, calculate_technical_indicators, generate_trading_signals
except ImportError:
    print("❌ Could not import c_signal module")
    def fetch_price_data(*args, **kwargs): return None
    def calculate_technical_indicators(*args, **kwargs): return None
    def generate_trading_signals(*args, **kwargs): return []

# Import OnChainAnalyzer
try:
    from onchain_analyzer2 import OnChainAnalyzer
    print("✅ Successfully imported OnChainAnalyzer from onchain_analyzer2")
except ImportError as e:
    print(f"❌ Could not import OnChainAnalyzer: {e}")
    class OnChainAnalyzer:
        def __init__(self): self.historical_flows = {}
        def accelerate_data_collection(self, *args): pass
        def get_comprehensive_onchain_analysis(self, *args): return {}
        def analyze_comprehensive_signals(self, *args): return []

# Import ML Predictor
try:
    from advanced_ml_predictorsimple4 import RealisticMLPredictor
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

class TradingExecutionSystem:
    def __init__(self, account_balance=1000, risk_per_trade=0.05, max_position_size=0.1):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade  # 5% risk per trade
        self.max_position_size = max_position_size
        self.trading_log = []
        
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

    def get_asset_volatility_category(self, symbol):
        """Determine volatility category for asset"""
        for category, symbols in self.asset_volatility_categories.items():
            if symbol in symbols:
                return category
        return 'MED_VOL'  # Default
    
    def calculate_volatility_based_leverage(self, symbol, price_data=None):
        """Calculate recommended leverage based on actual volatility"""
        category = self.get_asset_volatility_category(symbol)
        
        # If we have price data, calculate actual ATR volatility
        if price_data is not None and 'atr' in price_data.columns and len(price_data) > 20:
            atr = price_data['atr'].iloc[-1]
            current_price = price_data['close'].iloc[-1]
            atr_percentage = (atr / current_price) * 100
            
            # Adjust recommendations based on actual volatility
            recommendations = self.leverage_recommendations[category].copy()
            
            # High volatility warning
            if atr_percentage > 10:  # Very high volatility
                recommendations['conservative'] = max(1, recommendations['conservative'] * 0.7)
                recommendations['balanced'] = max(1, recommendations['balanced'] * 0.7)
                recommendations['aggressive'] = max(1.5, recommendations['aggressive'] * 0.7)
                recommendations['max_safe'] = max(2, recommendations['max_safe'] * 0.7)
                recommendations['volatility_warning'] = f"⚠️ High volatility detected: {atr_percentage:.1f}% daily ATR"
            elif atr_percentage > 7:  # Above average volatility
                recommendations['conservative'] = max(1, recommendations['conservative'] * 0.8)
                recommendations['balanced'] = max(1, recommendations['balanced'] * 0.8)
                recommendations['aggressive'] = max(1.5, recommendations['aggressive'] * 0.8)
                recommendations['max_safe'] = max(2, recommendations['max_safe'] * 0.8)
                recommendations['volatility_warning'] = f"📈 Above avg volatility: {atr_percentage:.1f}% daily ATR"
            else:
                recommendations['current_atr'] = f"{atr_percentage:.1f}% daily"
            
            return recommendations
        
        return self.leverage_recommendations[category]

    def calculate_position_size(self, entry_price, stop_loss_price, symbol, leverage=None, price_data=None):
        """Calculate position size with risk management - CORRECTED to risk exactly 5% of capital"""
        # Calculate the risk amount (5% of account balance)
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
        
        # Apply maximum position size constraint (10% of account)
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
        
        # Calculate liquidation price with 5% buffer
        direction = "LONG" if entry_price < stop_loss_price else "SHORT"
        
        # More realistic liquidation calculation
        maintenance_margin_rate = 0.05  # 5% maintenance margin typical for crypto
        
        if direction == "LONG":
            liquidation_price = entry_price * (1 - (1/leverage) * (1 - maintenance_margin_rate))
        else:
            liquidation_price = entry_price * (1 + (1/leverage) * (1 - maintenance_margin_rate))
        
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
        
        # Check if we got full 5% risk
        risk_status = "✅" if abs(actual_risk_percentage - (self.risk_per_trade * 100)) < 0.1 else "❌"
        
        return position_size_units, position_value, margin_required, leverage, liquidation_price, buffer_to_liquidation, leverage_recommendations, risk_amount, actual_risk_percentage, risk_status

    def calculate_leverage_scenarios(self, entry_price, stop_loss, position_value, symbol, direction, price_data=None):
        """Calculate different leverage scenarios with realistic recommendations"""
        scenarios = []
        
        # Get volatility-based recommendations
        leverage_recs = self.calculate_volatility_based_leverage(symbol, price_data)
        
        # Define leverage options to test - ADDED MORE OPTIONS
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
        
        for leverage in leverage_options:
            margin_required = position_value / leverage
            
            # Calculate liquidation
            maintenance_margin_rate = 0.05
            
            if direction == "LONG":
                liquidation = entry_price * (1 - (1/leverage) * (1 - maintenance_margin_rate))
                liquidation_pct = ((entry_price - liquidation) / entry_price) * 100
                buffer_to_liquidation = liquidation_pct - stop_distance
            else:
                liquidation = entry_price * (1 + (1/leverage) * (1 - maintenance_margin_rate))
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
            
            # Risk assessment
            if buffer_to_liquidation > 30:
                risk_level = "🟢 VERY SAFE"
            elif buffer_to_liquidation > 20:
                risk_level = "🟢 SAFE"
            elif buffer_to_liquidation > 15:
                risk_level = "🟡 MODERATE"
            elif buffer_to_liquidation > 10:
                risk_level = "🟠 HIGH"
            elif buffer_to_liquidation > 5:
                risk_level = "🔴 DANGEROUS"
            else:
                risk_level = "💀 EXTREME"
            
            scenarios.append({
                'leverage': leverage,
                'margin': margin_required,
                'liquidation': liquidation,
                'buffer_pct': buffer_to_liquidation,
                'risk': risk_level,
                'recommendation': recommendation,
                'is_recommended': is_recommended,
                'is_dangerous': leverage > leverage_recs['max_safe']
            })
        
        return scenarios, leverage_recs

    def score_signals(self, ta_signals, onchain_signals):
        """IMPROVED signal scoring with BETTER SHORT detection"""
        score = 0
        
        # Enhanced Technical Analysis Scoring
        strong_bull_count = 0
        strong_bear_count = 0
        
        for signal in ta_signals:
            # STRONG SIGNALS (Higher weights)
            if 'STRONG UPTREND' in signal:
                score += 4  # Increased from 3
                strong_bull_count += 1
            elif 'STRONG DOWNTREND' in signal:
                score -= 4  # Increased from 3
                strong_bear_count += 1
            elif 'UPTREND' in signal and 'STRONG' not in signal:
                score += 2
            elif 'DOWNTREND' in signal and 'STRONG' not in signal:
                score -= 2
            
            # RSI SIGNALS (Added scoring for all RSI conditions)
            if 'RSI' in signal:
                if 'OVERSOLD' in signal:
                    score += 2
                elif 'OVERBOUGHT' in signal:
                    score -= 2
                elif 'BULLISH' in signal:
                    score += 1
                elif 'BEARISH' in signal:
                    score -= 1
            
            # MACD SIGNALS (Added scoring)
            if 'MACD' in signal:
                if 'BULLISH CROSSOVER' in signal:
                    score += 2
                elif 'BEARISH CROSSOVER' in signal:
                    score -= 2
                elif 'BULLISH MOMENTUM' in signal:
                    score += 1
                elif 'BEARISH MOMENTUM' in signal:
                    score -= 1
            
            # PRICE POSITION SIGNALS
            if 'UPPER BOLLINGER' in signal or 'AT UPPER BOLLINGER' in signal:
                score -= 1  # Overbought
            elif 'LOWER BOLLINGER' in signal or 'AT LOWER BOLLINGER' in signal:
                score += 1  # Oversold
            
            # SUPPORT/RESISTANCE SIGNALS
            if 'SUPPORT' in signal:
                score += 1
            elif 'RESISTANCE' in signal:
                score -= 1
            
            # VOLUME SIGNALS
            if 'HIGH VOLUME' in signal:
                # Bullish volume in uptrend, bearish in downtrend
                if strong_bull_count > strong_bear_count:
                    score += 1
                elif strong_bear_count > strong_bull_count:
                    score -= 1
        
        # Enhanced On-Chain Scoring
        for signal in onchain_signals:
            if 'BULLISH' in signal and '🟢' in signal:
                score += 2
            elif 'BEARISH' in signal and '🔴' in signal:
                score -= 2
            elif 'NEUTRAL' in signal and '🟡' in signal:
                # Neutral doesn't affect score
                pass
            elif 'CAUTION' in signal and '⚠️' in signal:
                score -= 1  # Caution signals are slightly bearish
        
        return score

    def determine_trade_direction(self, signal_score, ta_signals):
        """BALANCED trade direction with SHORT support"""
        # Check for clear technical direction
        bearish_indicators = sum(1 for s in ta_signals if 'BEARISH' in s or 'DOWNTREND' in s)
        bullish_indicators = sum(1 for s in ta_signals if 'BULLISH' in s or 'UPTREND' in s)
        
        # ADJUSTED THRESHOLDS: Easier to get SHORT signals
        if signal_score >= 3:  # Lowered from 4 for more sensitivity
            return "LONG"
        elif signal_score <= -3:  # Changed from -4
            return "SHORT"
        elif signal_score <= -2 and bearish_indicators >= 3:
            return "SHORT"  # Allow SHORT with strong bearish indicators
        else:
            return "NO_TRADE"

    def calculate_levels(self, direction, price_data, current_price, symbol):
        """Calculate entry, stop loss, and take profit for BOTH LONG and SHORT"""
        high_20 = price_data['high'].tail(20).max()
        low_20 = price_data['low'].tail(20).min()
        
        if 'atr' in price_data.columns and not pd.isna(price_data['atr'].iloc[-1]):
            atr = price_data['atr'].iloc[-1]
        else:
            atr = current_price * 0.02
        
        if direction == "LONG":
            # LONG setup
            support_level = price_data['support'].iloc[-1] if 'support' in price_data.columns else current_price * 0.95
            if not pd.isna(support_level) and current_price <= support_level * 1.02:
                entry_price = min(current_price, support_level * 1.01)
            else:
                entry_price = current_price
            
            # Use tighter stop (5% max or ATR-based)
            atr_stop = entry_price - (atr * 1.5)
            price_based_stop = entry_price * 0.95  # 5% max stop
            stop_loss = max(low_20, min(atr_stop, price_based_stop))
            
            # Ensure stop isn't too far
            max_stop_distance = entry_price * 0.07  # 7% maximum stop
            if (entry_price - stop_loss) > max_stop_distance:
                stop_loss = entry_price - max_stop_distance
            
            risk_amount = entry_price - stop_loss
            
            take_profits = [
                entry_price + risk_amount * 1,
                entry_price + risk_amount * 2,
                entry_price + risk_amount * 3
            ]
            
        else:  # SHORT setup
            # SHORT: Entry near resistance
            resistance_level = price_data['resistance'].iloc[-1] if 'resistance' in price_data.columns else current_price * 1.05
            if not pd.isna(resistance_level) and current_price >= resistance_level * 0.98:
                entry_price = max(current_price, resistance_level * 0.99)
            else:
                entry_price = current_price
            
            # SHORT: Stop loss above recent high (tighter)
            atr_stop = entry_price + (atr * 1.5)
            price_based_stop = entry_price * 1.05  # 5% max stop
            stop_loss = min(high_20, max(atr_stop, price_based_stop))
            
            # Ensure stop isn't too far
            max_stop_distance = entry_price * 0.07  # 7% maximum stop
            if (stop_loss - entry_price) > max_stop_distance:
                stop_loss = entry_price + max_stop_distance
            
            risk_amount = stop_loss - entry_price
            
            # SHORT: Take profits BELOW entry
            take_profits = [
                entry_price - risk_amount * 1,
                entry_price - risk_amount * 2,
                entry_price - risk_amount * 3
            ]
        
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
            
        current_price = price_data['close'].iloc[-1]
        
        # Score signals
        signal_score = self.score_signals(ta_signals, onchain_signals)
        
        # Determine trade direction
        direction = self.determine_trade_direction(signal_score, ta_signals)
        
        if direction == "NO_TRADE":
            return None
        
        # Calculate levels
        entry_price, stop_loss, take_profits = self.calculate_levels(
            direction, price_data, current_price, symbol
        )
        
        # Calculate position sizing with custom leverage
        position_size, position_value, margin_required, leverage, liquidation_price, buffer_to_liq, leverage_recs, risk_amount, actual_risk_percentage, risk_status = self.calculate_position_size(
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
            'leverage_recommendations': leverage_recs,
            'account_balance': self.account_balance,
            'risk_amount': risk_amount,
            'risk_percentage': actual_risk_percentage,
            'risk_status': risk_status,
            'asset_category': self.get_asset_volatility_category(symbol),
            'price_at_analysis': current_price
        }
        
        return trade_signal

    def generate_trading_plan(self, trade_signal, show_scenarios=True):
        """Generate detailed trading plan with SHORT support and leverage analysis"""
        if not trade_signal:
            return "NO TRADE: Insufficient signal strength"
        
        # Get current price to see if signal is still valid
        try:
            current_data = fetch_price_data(trade_signal['symbol'], '1m', 2)
            if current_data is not None:
                current_price = current_data['close'].iloc[-1]
                price_at_signal = trade_signal.get('price_at_analysis', trade_signal['entry_price'])
                price_change = ((current_price - price_at_signal) / price_at_signal) * 100
            else:
                current_price = None
                price_change = 0
        except:
            current_price = None
            price_change = 0
        
        direction_emoji = "🟢" if trade_signal['direction'] == "LONG" else "🔴"
        
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
        
        plan = f"""
{direction_emoji} **TRADING PLAN: {trade_signal['direction']} {trade_signal['symbol']}** {direction_emoji}

⏰ TIMING INFORMATION:
   Analysis Time: {trade_signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
   Price at Analysis: ${trade_signal.get('price_at_analysis', trade_signal['entry_price']):,.2f}"""
        
        if current_price:
            plan += f"""
   Current Price: ${current_price:,.2f}
   Price Change: {price_change:+.2f}%"""
            
            if abs(price_change) > 2.0:
                plan += f"\n   ⚠️  WARNING: Price moved {price_change:+.2f}% since analysis"
                if price_change > 2.0 and trade_signal['direction'] == 'LONG':
                    plan += f"\n   💡 Consider waiting for pullback or adjusting entry"
                elif price_change < -2.0 and trade_signal['direction'] == 'SHORT':
                    plan += f"\n   💡 Consider waiting for bounce or adjusting entry"

        plan += f"""

📊 SYMBOL: {trade_signal['symbol']}
🏷️ ASSET TYPE: {category_info['description']} ({trade_signal['asset_category']})
💼 ACCOUNT BALANCE: ${trade_signal['account_balance']:,.2f}

⚡ TRADE DIRECTION: {trade_signal['direction']}
💰 ENTRY PRICE: ${trade_signal['entry_price']:,.2f}
🛡️ STOP LOSS: ${trade_signal['stop_loss']:,.2f} ({stop_distance:+.1f}%)
🎯 TAKE PROFITS:"""
        
        for i, tp in enumerate(trade_signal['take_profits'], 1):
            if i == 1:
                distance = tp1_distance
            elif i == 2:
                distance = tp2_distance
            else:
                distance = tp3_distance
            plan += f"\n   TP{i}: ${tp:,.2f} ({distance:+.1f}%)"
        
        plan += f"""

📈 POSITION SIZING:
   Units: {trade_signal['position_size']:.6f}
   Position Value: ${trade_signal['position_value']:,.2f}
   Margin Required: ${trade_signal['margin_required']:,.2f}
   Leverage: {trade_signal['leverage']:.1f}x ({'BALANCED' if trade_signal['leverage'] == category_info['balanced'] else 'AGGRESSIVE' if trade_signal['leverage'] == category_info['aggressive'] else 'CONSERVATIVE'})
   Liquidation Price: ${trade_signal['liquidation_price']:,.2f} ({liquidation_distance:+.1f}% from entry)

⚖️ RISK MANAGEMENT:
   Risk per Trade: {self.risk_per_trade*100}% of account
   Actual Risk: ${trade_signal['risk_amount']:,.2f} ({trade_signal['risk_percentage']:.1f}% of account) {trade_signal['risk_status']}
   R:R Ratio: {trade_signal['risk_reward_ratio']:.2f}:1
   Signal Strength: {trade_signal['signal_strength']}/10
   Buffer to Liquidation: {trade_signal['buffer_to_liquidation']:.1f}%
   Risk Level: {'🟢 SAFE' if trade_signal['buffer_to_liquidation'] > 20 else '🟡 MODERATE' if trade_signal['buffer_to_liquidation'] > 10 else '🟠 HIGH' if trade_signal['buffer_to_liquidation'] > 5 else '🔴 DANGEROUS'}
"""
        
        # Show volatility warning if present
        if 'volatility_warning' in category_info:
            plan += f"\n⚠️  VOLATILITY WARNING: {category_info['volatility_warning']}"
        elif 'current_atr' in category_info:
            plan += f"\n📊 CURRENT VOLATILITY: {category_info['current_atr']}"
        
        # Leverage recommendations
        plan += f"""
🎯 LEVERAGE RECOMMENDATIONS for {trade_signal['symbol'].split('/')[0]}:
   🟢 CONSERVATIVE: {category_info['conservative']:.1f}x (Beginner, low risk)
   🟡 BALANCED: {category_info['balanced']:.1f}x (Recommended, moderate risk)
   🟠 AGGRESSIVE: {category_info['aggressive']:.1f}x (Experienced, high risk)
   🔴 MAX SAFE: {category_info['max_safe']:.1f}x (Expert only, very high risk)
   ⚠️  AVOID: >{category_info['max_safe']:.1f}x (Extreme liquidation risk)
"""
        
        # Leverage scenarios analysis
        if show_scenarios and 'leverage_scenarios' in trade_signal:
            plan += f"""
🔧 LEVERAGE SCENARIOS ANALYSIS (Position Value: ${trade_signal['position_value']:,.2f}):
┌──────────────┬──────────┬──────────────┬──────────────┬──────────────┬─────────────────────┐
│ Leverage/Rec │  Margin  │ Liquidation  │  Liq. Dist.  │  Stop Dist.  │ Risk & Assessment   │
├──────────────┼──────────┼──────────────┼──────────────┼──────────────┼─────────────────────┤"""
            
            # Show all scenarios up to 25x
            for scenario in trade_signal['leverage_scenarios']:
                if scenario['leverage'] <= 25:  # Show up to 25x
                    star = " ★" if scenario['is_recommended'] else ""
                    warning = " ⚠️" if scenario['is_dangerous'] else ""
                    
                    plan += f"""
│ {scenario['leverage']:>5.1f}x{star:<1}{warning:<1} │ ${scenario['margin']:>7,.0f} │ ${scenario['liquidation']:>11,.0f} │ {scenario['buffer_pct']:>11.1f}% │ {stop_distance:>11.1f}% │ {scenario['risk']:>19} │"""
            
            plan += """
└──────────────┴──────────┴──────────────┴──────────────┴──────────────┴─────────────────────┘
💡 Higher leverage = less margin but liquidation gets closer!
"""
        
        if trade_signal['direction'] == "SHORT":
            plan += "\n⚠️  SHORT TRADING WARNINGS:"
            plan += "\n   - Use LOWER leverage for shorts (max 5x for most assets)"
            plan += "\n   - Higher risk of short squeezes"
            plan += "\n   - Monitor funding rates (avoid >0.01% hourly)"
            plan += "\n   - Consider trailing stop loss after TP1"
            plan += "\n   - Be prepared for sudden price spikes"
        
        plan += """
💡 EXECUTION INSTRUCTIONS:
   1. Set limit order at entry price
   2. Set stop loss immediately after fill
   3. Set take profit orders
   4. Monitor for early exit conditions
   5. Choose leverage based on your risk tolerance
"""
        
        # Add comprehensive leverage adjustment guide
        plan += f"""
⚡ COMPREHENSIVE LEVERAGE GUIDE:
   
   🟢 SAFEST OPTIONS (Beginner):
   • No leverage (1x): ${trade_signal['position_value']/1:,.0f} margin
   • Conservative ({category_info['conservative']:.1f}x): ${trade_signal['position_value']/category_info['conservative']:,.0f} margin
   
   🟡 BALANCED OPTIONS (Intermediate):
   • Current ({trade_signal['leverage']:.1f}x): ${trade_signal['margin_required']:,.0f} margin
   • Balanced ({category_info['balanced']:.1f}x): ${trade_signal['position_value']/category_info['balanced']:,.0f} margin
   
   🟠 AGGRESSIVE OPTIONS (Experienced):
   • Aggressive ({category_info['aggressive']:.1f}x): ${trade_signal['position_value']/category_info['aggressive']:,.0f} margin
   • Max Safe ({category_info['max_safe']:.1f}x): ${trade_signal['position_value']/category_info['max_safe']:,.0f} margin
   
   🔴 DANGEROUS OPTIONS (Experts Only):
   • 10x: ${trade_signal['position_value']/10:,.0f} margin ⚠️ High risk
   • 20x: ${trade_signal['position_value']/20:,.0f} margin 💀 Extreme risk
   • 25x+: ${trade_signal['position_value']/25:,.0f} margin 💀 Almost guaranteed liquidation
"""
        
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
        
    def train_ml_models(self, price_data):
        """Train ML models on price data only"""
        if not ML_AVAILABLE:
            print("❌ ML dependencies not installed")
            return False
            
        print("🤖 INITIATING SIMPLE ML TRAINING (PRICE DATA ONLY)...")
        success = self.ml_predictor.train_simple_model(price_data)
        self.ml_trained = success
        if success:
            print("✅ SIMPLE ML MODELS READY FOR LIVE PREDICTIONS")
        return success
    
    def generate_enhanced_signal(self, ta_signals, onchain_signals, price_data):
        """Generate signal with ML enhancement"""
        traditional_score = self.score_signals(ta_signals, onchain_signals)
        print(f"🔍 TRADITIONAL SCORE: {traditional_score}")

        ml_boost = 0
        ml_signals = []
        
        if self.ml_trained and ML_AVAILABLE:
            try:
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
                
            except Exception as e:
                print(f"⚠️  ML Prediction Error: {e}")
                import traceback
                traceback.print_exc()
        
        combined_score = traditional_score + ml_boost
        print(f"🎯 FINAL COMBINED SCORE: {traditional_score} + {ml_boost} = {combined_score}")
        
        return self.determine_trade_direction(combined_score, ta_signals), ml_signals, combined_score

def integrate_and_trade_with_ml(symbol='BTC/USDT', timeframe='4h', account_balance=1000, enable_ml=True, custom_leverage=None):
    """Main function with ML enhancement - WITH TIMESTAMPS"""
    
    # CAPTURE START TIME
    analysis_start_time = datetime.now()
    
    trading_system = MLEnhancedTradingSystem(account_balance)
    
    print(f"\n{'🤖' * 20}")
    print("🤖 ADVANCED LEVERAGE-AWARE TRADING SYSTEM")
    print(f"{'🤖' * 20}")
    
    # Add timestamp to header
    print(f"🕒 ANALYSIS STARTED: {analysis_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get data
    print(f"\n🔍 ANALYZING {symbol} FOR TRADING OPPORTUNITIES...")
    price_data = fetch_price_data(symbol, timeframe, 500)
    if price_data is None:
        print(f"❌ Could not fetch price data for {symbol}")
        return
    
    # CAPTURE DATA FETCH TIME
    data_fetch_time = datetime.now()
    
    # Display data timestamp
    print(f"📊 DATA FETCHED: {data_fetch_time.strftime('%H:%M:%S')}")
    print(f"   Price: ${price_data['close'].iloc[-1]:,.2f}")
    
    # Calculate time difference
    fetch_duration = (data_fetch_time - analysis_start_time).total_seconds()
    print(f"   Fetch time: {fetch_duration:.1f} seconds")
    
    price_data = calculate_technical_indicators(price_data)
    ta_signals = generate_trading_signals(price_data)
    
    # Extract technical indicators for market analysis - FIXED to handle missing indicators
    current_price = price_data['close'].iloc[-1]
    
    # Check available columns
    available_indicators = []
    indicator_values = {}
    
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
    
    # On-chain data
    base_symbol = symbol.split('/')[0]
    onchain_analyzer = OnChainAnalyzer()
    onchain_analyzer.accelerate_data_collection(base_symbol)
    all_metrics = onchain_analyzer.get_comprehensive_onchain_analysis(base_symbol)
    onchain_signals = onchain_analyzer.analyze_comprehensive_signals(all_metrics)
    
    # Train ML (first run)
    if enable_ml and ML_AVAILABLE and not trading_system.ml_trained:
        print("\n🤖 TRAINING SIMPLE ML MODELS ON PRICE DATA...")
        trading_system.train_ml_models(price_data)
    
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
        else:
            trade_signal = None
    else:
        print("\n🔍 USING TRADITIONAL ANALYSIS ONLY")
        trade_signal = trading_system.generate_trading_signal(ta_signals, onchain_signals, price_data, symbol, custom_leverage)
    
    # Display analysis with improved formatting
    print(f"\n{'📊' * 20}")
    print("📊 MARKET ANALYSIS SUMMARY")
    print(f"{'📊' * 20}")
    
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
        
        print(f"\n{'⚡' * 20}")
        print("⚡ EXECUTION DETAILS")
        print(f"{'⚡' * 20}")
        
        spot_execution = trading_system.execute_spot_trading(trade_signal)
        print(f"\n🪙 SPOT TRADING:")
        print(spot_execution)
        
        futures_execution = trading_system.execute_futures_trading(trade_signal)
        print(f"\n📈 FUTURES TRADING:")
        print(futures_execution)
        
        # Show what happens if leverage changes
        print(f"\n{'🔧' * 20}")
        print("🔧 LEVERAGE IMPACT ANALYSIS")
        print(f"{'🔧' * 20}")
        
        print(f"\n📈 POSITION VALUE: ${trade_signal['position_value']:,.2f}")
        print("If you change leverage (ALL scenarios risk exactly 5% on stop loss):")
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
            
            if trade_signal['direction'] == 'LONG':
                liq_price = trade_signal['entry_price'] * (1 - (1/leverage) * 0.95)
                liq_distance = ((trade_signal['entry_price'] - liq_price) / trade_signal['entry_price']) * 100
                stop_distance = ((trade_signal['entry_price'] - trade_signal['stop_loss']) / trade_signal['entry_price']) * 100
            else:
                liq_price = trade_signal['entry_price'] * (1 + (1/leverage) * 0.95)
                liq_distance = ((liq_price - trade_signal['entry_price']) / trade_signal['entry_price']) * 100
                stop_distance = ((trade_signal['stop_loss'] - trade_signal['entry_price']) / trade_signal['entry_price']) * 100
            
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
        print(f"\n❌ NO TRADING OPPORTUNITY FOUND")
        print(f"   Signal strength insufficient for {symbol}")
        print(f"   Waiting for better setup...")
    
    # At the end, add completion timestamp
    analysis_end_time = datetime.now()
    total_duration = (analysis_end_time - analysis_start_time).total_seconds()
    
    print(f"\n{'📊' * 20}")
    print(f"📊 ANALYSIS COMPLETED: {analysis_end_time.strftime('%H:%M:%S')}")
    print(f"📊 TOTAL DURATION: {total_duration:.1f} seconds")
    print(f"{'📊' * 20}")

def batch_analyze_cryptos(cryptos=None, account_balance=1000, enable_ml=False):
    """Analyze multiple cryptocurrencies with DEDICATED ML for each crypto"""
    if cryptos is None:
        cryptos = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT']
    
    print(f"\n{'🔍' * 20}")
    print("🔍 MULTI-CRYPTO SCANNING MODE")
    print(f"{'🔍' * 20}")
    
    # SCAN START TIME
    scan_start_time = datetime.now()
    print(f"🕒 SCAN STARTED: {scan_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 CRYPTOS TO ANALYZE: {len(cryptos)}")
    print(f"📊 ACCOUNT BALANCE: ${account_balance:,.2f}")
    
    if enable_ml and ML_AVAILABLE:
        print("🤖 ML-ENHANCED SCANNING ENABLED")
        print("✅ DEDICATED ML MODEL FOR EACH CRYPTO (BEST ACCURACY)")
    else:
        print("📊 TRADITIONAL SCANNING ONLY")
    
    best_opportunities = []
    
    for i, crypto in enumerate(cryptos, 1):
        crypto_start_time = datetime.now()
        
        print(f"\n{'─' * 60}")
        print(f"📈 [{i}/{len(cryptos)}] Analyzing {crypto}...")
        print(f"   🕒 START: {crypto_start_time.strftime('%H:%M:%S')}")
        
        try:
            price_data = fetch_price_data(crypto, '4h', 100)
            if price_data is None:
                print(f"   ❌ Could not fetch price data for {crypto}")
                continue
                
            price_data = calculate_technical_indicators(price_data)
            ta_signals = generate_trading_signals(price_data)
            
            base_symbol = crypto.split('/')[0]
            onchain_analyzer = OnChainAnalyzer()
            onchain_analyzer.accelerate_data_collection(base_symbol)
            all_metrics = onchain_analyzer.get_comprehensive_onchain_analysis(base_symbol)
            onchain_signals = onchain_analyzer.analyze_comprehensive_signals(all_metrics)
            
            # Generate trade signal
            trade_signal = None
            
            if enable_ml and ML_AVAILABLE:
                # CREATE DEDICATED ML SYSTEM FOR THIS SPECIFIC CRYPTO
                crypto_ml_system = MLEnhancedTradingSystem(account_balance)
                
                print(f"   🤖 Training dedicated ML model for {crypto}...")
                training_start = time.time()
                crypto_ml_system.train_ml_models(price_data)  # Train on THIS crypto only
                training_time = time.time() - training_start
                
                print(f"   ✅ {crypto} ML trained in {training_time:.1f}s")
                
                # Generate enhanced signal with THIS crypto's dedicated ML
                trade_direction, ml_signals, combined_score = crypto_ml_system.generate_enhanced_signal(
                    ta_signals, onchain_signals, price_data
                )
                
                print(f"   🤖 {crypto} Combined Score: {combined_score} (Traditional + ML)")
                
                if trade_direction != "NO_TRADE":
                    trade_signal = crypto_ml_system.generate_trading_signal(
                        ta_signals, onchain_signals, price_data, crypto
                    )
                    if trade_signal:
                        trade_signal['signal_strength'] = combined_score
                        trade_signal['ml_model'] = f"Dedicated {crypto} ML"
                        trade_signal['ml_training_time'] = training_time
            else:
                # Use traditional system (no ML)
                trading_system = TradingExecutionSystem(account_balance)
                trade_signal = trading_system.generate_trading_signal(
                    ta_signals, onchain_signals, price_data, crypto
                )
            
            crypto_end_time = datetime.now()
            crypto_duration = (crypto_end_time - crypto_start_time).total_seconds()
            
            print(f"   🕒 END: {crypto_end_time.strftime('%H:%M:%S')}")
            print(f"   ⏱️  DURATION: {crypto_duration:.1f}s")
            print(f"   💰 Price: ${price_data['close'].iloc[-1]:,.2f}")
            
            if trade_signal:
                # ADD TIMESTAMP TO TRADE SIGNAL
                trade_signal['analysis_timestamp'] = crypto_end_time
                trade_signal['analysis_duration'] = crypto_duration
                trade_signal['price_at_analysis'] = price_data['close'].iloc[-1]
                
                best_opportunities.append(trade_signal)
                direction_emoji = "🟢" if trade_signal['direction'] == "LONG" else "🔴"
                
                if enable_ml and 'ml_model' in trade_signal:
                    print(f"   {direction_emoji} TRADE FOUND: {trade_signal['direction']} - Score: {trade_signal['signal_strength']}")
                    print(f"   🤖 Using: {trade_signal['ml_model']} ({trade_signal['ml_training_time']:.1f}s training)")
                else:
                    print(f"   {direction_emoji} TRADE FOUND: {trade_signal['direction']} - Score: {trade_signal['signal_strength']}")
                
                print(f"   ⏰ Analyzed: {trade_signal['analysis_timestamp'].strftime('%H:%M:%S')}")
            else:
                print(f"   ❌ No trade signal")
                
        except Exception as e:
            crypto_end_time = datetime.now()
            crypto_duration = (crypto_end_time - crypto_start_time).total_seconds()
            print(f"   ⚡ Error analyzing {crypto}: {e}")
            print(f"   🕒 END: {crypto_end_time.strftime('%H:%M:%S')}")
            print(f"   ⏱️  DURATION: {crypto_duration:.1f}s")
    
    # Display best opportunities WITH TIMESTAMPS
    if best_opportunities:
        scan_end_time = datetime.now()
        total_duration = (scan_end_time - scan_start_time).total_seconds()
        
        print(f"\n{'🎯' * 20}")
        print("🎯 BEST TRADING OPPORTUNITIES")
        print(f"🕒 SCAN COMPLETED: {scan_end_time.strftime('%H:%M:%S')}")
        print(f"⏱️  TOTAL SCAN TIME: {total_duration:.1f} seconds")
        print(f"{'🎯' * 20}")
        
        # Sort by signal strength
        best_opportunities.sort(key=lambda x: abs(x['signal_strength']), reverse=True)
        
        for i, opportunity in enumerate(best_opportunities[:5], 1):  # Show top 5
            direction_emoji = "🟢" if opportunity['direction'] == "LONG" else "🔴"
            time_ago = (datetime.now() - opportunity['analysis_timestamp']).total_seconds()
            
            print(f"\n#{i} {opportunity['symbol']} - {direction_emoji} {opportunity['direction']}")
            print(f"   📊 Analysis Time: {opportunity['analysis_timestamp'].strftime('%H:%M:%S')}")
            print(f"   ⏱️  Analysis Duration: {opportunity['analysis_duration']:.1f}s")
            print(f"   🕒 Age: {time_ago:.0f} seconds ago")
            print(f"   💰 Price then: ${opportunity['price_at_analysis']:,.2f}")
            print(f"   🎯 Signal Strength: {opportunity['signal_strength']}")
            print(f"   ⚖️  R:R Ratio: {opportunity['risk_reward_ratio']:.2f}:1")
            print(f"   💵 Risk Amount: ${opportunity['risk_amount']:,.2f} ({opportunity['risk_percentage']:.1f}% of account) {opportunity['risk_status']}")
            print(f"   ⚡ Leverage: {opportunity['leverage']}x")
            print(f"   💼 Margin: ${opportunity['margin_required']:,.0f}")
            
            if 'ml_model' in opportunity:
                print(f"   🤖 ML Model: {opportunity['ml_model']}")
            
            # Check if price has changed significantly
            try:
                current_data = fetch_price_data(opportunity['symbol'], '1m', 2)
                if current_data is not None:
                    current_price = current_data['close'].iloc[-1]
                    price_change = ((current_price - opportunity['price_at_analysis']) / 
                                   opportunity['price_at_analysis']) * 100
                    print(f"   📈 Current Price: ${current_price:,.2f}")
                    print(f"   📊 Price Change: {price_change:+.2f}%")
                    
                    if abs(price_change) > 1.0:  # If price changed more than 1%
                        if opportunity['direction'] == 'LONG' and price_change > 1.0:
                            print(f"   ⚠️  WARNING: Price up {price_change:.1f}% - Consider waiting for pullback")
                        elif opportunity['direction'] == 'SHORT' and price_change < -1.0:
                            print(f"   ⚠️  WARNING: Price down {abs(price_change):.1f}% - Consider waiting for bounce")
            except:
                pass
            
            # Quick action recommendation
            print(f"\n   🚀 QUICK ACTION:")
            if opportunity['signal_strength'] >= 5:
                print(f"   🟢 STRONG SIGNAL - Consider entering soon")
            elif opportunity['signal_strength'] >= 3:
                print(f"   🟡 MODERATE SIGNAL - Wait for confirmation")
            else:
                print(f"   🟠 WEAK SIGNAL - Monitor for improvement")
            
            # Show if this uses ML
            if enable_ml and ML_AVAILABLE:
                print(f"   🤖 ML-ENHANCED SIGNAL")
    else:
        scan_end_time = datetime.now()
        total_duration = (scan_end_time - scan_start_time).total_seconds()
        
        print(f"\n{'❌' * 20}")
        print("❌ NO TRADING OPPORTUNITIES FOUND")
        print(f"🕒 SCAN COMPLETED: {scan_end_time.strftime('%H:%M:%S')}")
        print(f"⏱️  TOTAL SCAN TIME: {total_duration:.1f} seconds")
        print(f"{'❌' * 20}")
        print("💡 SUGGESTIONS:")
        print("1. Try different cryptocurrencies")
        print("2. Check different timeframes")
        print("3. Enable ML for enhanced signals")
        print("4. Wait for market conditions to change")

def real_time_monitoring(symbols=None, update_interval=300, account_balance=1000, enable_ml=True):
    """Real-time monitoring with continuous updates"""
    if symbols is None:
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    print(f"\n{'📡' * 20}")
    print("📡 REAL-TIME MONITORING MODE")
    print(f"{'📡' * 20}")
    print(f"🕒 STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 MONITORING {len(symbols)} SYMBOLS")
    print(f"⏰ UPDATE INTERVAL: {update_interval} seconds")
    print(f"🤖 ML ENABLED: {enable_ml}")
    print(f"💼 ACCOUNT: ${account_balance:,.2f}")
    
    iteration = 1
    
    try:
        while True:
            print(f"\n{'=' * 60}")
            print(f"🔄 ITERATION #{iteration} - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'=' * 60}")
            
            for symbol in symbols:
                try:
                    print(f"\n📊 CHECKING {symbol}...")
                    
                    # Quick analysis for real-time
                    price_data = fetch_price_data(symbol, '15m', 50)
                    if price_data is not None:
                        price_data = calculate_technical_indicators(price_data)
                        current_price = price_data['close'].iloc[-1]
                        
                        # Quick signal check
                        if enable_ml and ML_AVAILABLE:
                            # Quick ML check
                            ml_system = MLEnhancedTradingSystem(account_balance)
                            ml_system.train_ml_models(price_data)
                            
                            ta_signals = generate_trading_signals(price_data)
                            base_symbol = symbol.split('/')[0]
                            onchain_analyzer = OnChainAnalyzer()
                            all_metrics = onchain_analyzer.get_comprehensive_onchain_analysis(base_symbol)
                            onchain_signals = onchain_analyzer.analyze_comprehensive_signals(all_metrics)
                            
                            trade_direction, ml_signals, combined_score = ml_system.generate_enhanced_signal(
                                ta_signals, onchain_signals, price_data
                            )
                            
                            if trade_direction != "NO_TRADE":
                                direction_emoji = "🟢" if trade_direction == "LONG" else "🔴"
                                print(f"   {direction_emoji} {trade_direction} SIGNAL DETECTED!")
                                print(f"   🤖 ML Score: {combined_score}")
                                print(f"   💰 Price: ${current_price:,.2f}")
                                
                                # Generate quick trade plan
                                trade_signal = ml_system.generate_trading_signal(
                                    ta_signals, onchain_signals, price_data, symbol
                                )
                                if trade_signal:
                                    print(f"   🎯 Entry: ${trade_signal['entry_price']:,.2f}")
                                    print(f"   🛡️  Stop: ${trade_signal['stop_loss']:,.2f}")
                                    print(f"   ⚡ Leverage: {trade_signal['leverage']}x")
                                    print(f"   ⚖️  R:R: {trade_signal['risk_reward_ratio']:.2f}:1")
                        else:
                            # Traditional quick check
                            print(f"   💰 Price: ${current_price:,.2f}")
                            
                            # Check simple conditions
                            if 'RSI' in price_data.columns:
                                rsi = price_data['RSI'].iloc[-1]
                                if rsi < 30:
                                    print(f"   📉 RSI OVERSOLD: {rsi:.1f}")
                                elif rsi > 70:
                                    print(f"   📈 RSI OVERBOUGHT: {rsi:.1f}")
                            
                            if 'MA20' in price_data.columns:
                                ma20 = price_data['MA20'].iloc[-1]
                                ma_diff = ((current_price - ma20) / ma20) * 100
                                if ma_diff > 5:
                                    print(f"   ⚡ {ma_diff:+.1f}% above MA20 - Overextended")
                                elif ma_diff < -5:
                                    print(f"   ⚡ {ma_diff:+.1f}% below MA20 - Oversold")
                except Exception as e:
                    print(f"   ❌ Error checking {symbol}: {str(e)[:50]}...")
            
            print(f"\n⏰ Next update in {update_interval} seconds...")
            print(f"🕒 Current time: {datetime.now().strftime('%H:%M:%S')}")
            
            time.sleep(update_interval)
            iteration += 1
            
    except KeyboardInterrupt:
        print(f"\n\n{'🛑' * 20}")
        print("🛑 REAL-TIME MONITORING STOPPED BY USER")
        print(f"🕒 ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔄 TOTAL ITERATIONS: {iteration-1}")
        print(f"{'🛑' * 20}")

def interactive_trading_planner():
    """Interactive trading planner with user input"""
    print(f"\n{'🎮' * 20}")
    print("🎮 INTERACTIVE TRADING PLANNER")
    print(f"{'🎮' * 20}")
    
    print("\n📊 WELCOME TO ADVANCED TRADING PLANNER")
    print("Choose your trading mode:")
    print("1. 📈 Single Crypto Analysis")
    print("2. 🔍 Multi-Crypto Scanner")
    print("3. 📡 Real-time Monitoring")
    print("4. 🎯 Custom Trading Plan")
    print("5. 💼 Account Simulator")
    print("6. 📚 Educational Guide")
    print("7. 🚪 Exit")
    
    try:
        choice = int(input("\nEnter choice (1-7): "))
    except:
        print("❌ Invalid choice. Exiting.")
        return
    
    if choice == 1:
        symbol = input("Enter symbol (e.g., BTC/USDT): ") or "BTC/USDT"
        timeframe = input("Enter timeframe (1m, 5m, 15m, 1h, 4h, 1d) [4h]: ") or "4h"
        balance = float(input("Account balance [1000]: ") or "1000")
        ml_enabled = input("Enable ML? (y/n) [y]: ").lower() != 'n'
        custom_leverage = input("Custom leverage (press Enter for auto): ") or None
        if custom_leverage:
            custom_leverage = float(custom_leverage)
        
        print(f"\n{'🚀' * 20}")
        print(f"🚀 STARTING ANALYSIS: {symbol} @ {timeframe}")
        print(f"{'🚀' * 20}")
        
        integrate_and_trade_with_ml(
            symbol=symbol,
            timeframe=timeframe,
            account_balance=balance,
            enable_ml=ml_enabled,
            custom_leverage=custom_leverage
        )
    
    elif choice == 2:
        print("\n🔍 MULTI-CRYPTO SCANNER")
        print("Enter cryptocurrencies (comma-separated, e.g., BTC/USDT,ETH/USDT,SOL/USDT)")
        cryptos_input = input("Cryptos [BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT,DOT/USDT]: ") or "BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT,DOT/USDT"
        cryptos = [c.strip() for c in cryptos_input.split(',')]
        balance = float(input("Account balance [1000]: ") or "1000")
        ml_enabled = input("Enable ML? (y/n) [y]: ").lower() != 'n'
        
        batch_analyze_cryptos(
            cryptos=cryptos,
            account_balance=balance,
            enable_ml=ml_enabled
        )
    
    elif choice == 3:
        print("\n📡 REAL-TIME MONITORING")
        symbols_input = input("Enter symbols to monitor (comma-separated) [BTC/USDT,ETH/USDT,SOL/USDT]: ") or "BTC/USDT,ETH/USDT,SOL/USDT"
        symbols = [s.strip() for s in symbols_input.split(',')]
        interval = int(input("Update interval in seconds [300]: ") or "300")
        balance = float(input("Account balance [1000]: ") or "1000")
        ml_enabled = input("Enable ML? (y/n) [y]: ").lower() != 'n'
        
        real_time_monitoring(
            symbols=symbols,
            update_interval=interval,
            account_balance=balance,
            enable_ml=ml_enabled
        )
    
    elif choice == 4:
        print("\n🎯 CUSTOM TRADING PLAN GENERATOR")
        symbol = input("Symbol (e.g., BTC/USDT): ") or "BTC/USDT"
        direction = input("Direction (LONG/SHORT): ").upper() or "LONG"
        entry_price = float(input("Entry price: "))
        stop_loss = float(input("Stop loss: "))
        account_balance = float(input("Account balance [1000]: ") or "1000")
        leverage = float(input("Leverage [3]: ") or "3")
        
        trading_system = TradingExecutionSystem(account_balance)
        
        # Create custom trade signal
        if direction == "LONG":
            take_profits = [
                entry_price + (entry_price - stop_loss) * 1,
                entry_price + (entry_price - stop_loss) * 2,
                entry_price + (entry_price - stop_loss) * 3
            ]
        else:
            take_profits = [
                entry_price - (stop_loss - entry_price) * 1,
                entry_price - (stop_loss - entry_price) * 2,
                entry_price - (stop_loss - entry_price) * 3
            ]
        
        # Calculate position size with custom leverage
        position_size, position_value, margin_required, used_leverage, liquidation_price, buffer_to_liq, leverage_recs, risk_amount, actual_risk_percentage, risk_status = trading_system.calculate_position_size(
            entry_price, stop_loss, symbol, leverage
        )
        
        trade_signal = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profits': take_profits,
            'position_size': position_size,
            'position_value': position_value,
            'margin_required': margin_required,
            'leverage': used_leverage,
            'liquidation_price': liquidation_price,
            'buffer_to_liquidation': buffer_to_liq,
            'leverage_recommendations': leverage_recs,
            'account_balance': account_balance,
            'risk_amount': risk_amount,
            'risk_percentage': actual_risk_percentage,
            'risk_status': risk_status,
            'asset_category': trading_system.get_asset_volatility_category(symbol),
            'price_at_analysis': entry_price,
            'risk_reward_ratio': trading_system.calculate_risk_reward(entry_price, stop_loss, take_profits),
            'signal_strength': 0  # Custom plan, no signal score
        }
        
        print(f"\n{'📋' * 20}")
        print("📋 CUSTOM TRADING PLAN")
        print(f"{'📋' * 20}")
        
        plan = trading_system.generate_trading_plan(trade_signal)
        print(plan)
        
        # Show execution details
        if direction == "LONG":
            execution = trading_system.execute_futures_long_execution(trade_signal)
        else:
            execution = trading_system.execute_futures_short_execution(trade_signal)
        
        print(execution)
    
    elif choice == 5:
        print("\n💼 ACCOUNT SIMULATOR")
        print("Simulate different account sizes and risk parameters")
        
        initial_balance = float(input("Initial account balance [1000]: ") or "1000")
        risk_per_trade = float(input("Risk per trade (e.g., 0.05 for 5%) [0.05]: ") or "0.05")
        num_trades = int(input("Number of trades to simulate [100]: ") or "100")
        win_rate = float(input("Win rate (e.g., 0.55 for 55%) [0.55]: ") or "0.55")
        avg_win_rr = float(input("Average win R:R ratio (e.g., 1.5) [1.5]: ") or "1.5")
        avg_loss_rr = float(input("Average loss R:R ratio (e.g., 1.0) [1.0]: ") or "1.0")
        
        trading_system = TradingExecutionSystem(initial_balance)
        trading_system.risk_per_trade = risk_per_trade
        
        print(f"\n{'📈' * 20}")
        print("📈 SIMULATION RESULTS")
        print(f"{'📈' * 20}")
        
        balance = initial_balance
        wins = 0
        losses = 0
        max_drawdown = 0
        peak_balance = balance
        
        print(f"\n🧪 SIMULATION PARAMETERS:")
        print(f"   Initial Balance: ${initial_balance:,.2f}")
        print(f"   Risk per Trade: {risk_per_trade*100}%")
        print(f"   Win Rate: {win_rate*100}%")
        print(f"   Win R:R: {avg_win_rr}:1")
        print(f"   Loss R:R: {avg_loss_rr}:1")
        print(f"   Total Trades: {num_trades}")
        
        print(f"\n📊 TRADE BY TRADE:")
        for i in range(num_trades):
            risk_amount = balance * risk_per_trade
            is_win = np.random.random() < win_rate
            
            if is_win:
                profit = risk_amount * avg_win_rr
                balance += profit
                wins += 1
                result = "WIN"
                result_emoji = "🟢"
            else:
                loss = risk_amount * avg_loss_rr
                balance -= loss
                losses += 1
                result = "LOSS"
                result_emoji = "🔴"
            
            # Update drawdown
            if balance > peak_balance:
                peak_balance = balance
            
            current_drawdown = ((peak_balance - balance) / peak_balance) * 100
            max_drawdown = max(max_drawdown, current_drawdown)
            
            if (i + 1) % 10 == 0:
                print(f"   Trade {i+1:3d}: {result_emoji} {result:4s} | Balance: ${balance:,.0f} | Drawdown: {current_drawdown:.1f}%")
        
        print(f"\n📈 FINAL RESULTS:")
        print(f"   Final Balance: ${balance:,.2f}")
        print(f"   Total Return: {(balance/initial_balance-1)*100:.1f}%")
        print(f"   Wins: {wins} ({wins/num_trades*100:.1f}%)")
        print(f"   Losses: {losses} ({losses/num_trades*100:.1f}%)")
        print(f"   Max Drawdown: {max_drawdown:.1f}%")
        
        # Calculate expected value
        expected_value = (win_rate * avg_win_rr * risk_per_trade) - ((1-win_rate) * avg_loss_rr * risk_per_trade)
        print(f"   Expected Value per Trade: {expected_value*100:.2f}%")
        
        if expected_value > 0:
            print(f"   🟢 POSITIVE EXPECTANCY SYSTEM")
        else:
            print(f"   🔴 NEGATIVE EXPECTANCY SYSTEM")
    
    elif choice == 6:
        print("\n📚 TRADING EDUCATION GUIDE")
        print(f"\n{'📖' * 20}")
        print("📖 TRADING FUNDAMENTALS")
        print(f"{'📖' * 20}")
        
        print("\n1. 📊 RISK MANAGEMENT:")
        print("   • Never risk more than 1-2% of account per trade")
        print("   • Use stop losses on EVERY trade")
        print("   • Risk-Reward ratio should be at least 1:1.5")
        
        print("\n2. ⚡ LEVERAGE GUIDELINES:")
        print("   • BTC: 3-5x max (low volatility)")
        print("   • ETH: 2-4x max (medium volatility)")
        print("   • Altcoins: 1-2x max (high volatility)")
        print("   • Memecoins: Avoid leverage or 1x only")
        
        print("\n3. 🎯 ENTRY STRATEGIES:")
        print("   • Wait for confirmation before entering")
        print("   • Enter near support/resistance levels")
        print("   • Use limit orders, not market orders")
        
        print("\n4. 💼 POSITION SIZING:")
        print("   • Calculate based on stop loss distance")
        print("   • Consider correlation between assets")
        print("   • Diversify across different sectors")
        
        print("\n5. 🧠 PSYCHOLOGY:")
        print("   • Stick to your trading plan")
        print("   • Don't chase losses")
        print("   • Take breaks to avoid burnout")
        print("   • Keep a trading journal")
        
        print("\n6. 🔧 USING THIS SYSTEM:")
        print("   • Start with conservative leverage")
        print("   • Paper trade first")
        print("   • Enable ML for enhanced signals")
        print("   • Use multi-crypto scanner for best opportunities")
        
        print(f"\n{'💡' * 20}")
        print("💡 PRO TIPS")
        print(f"{'💡' * 20}")
        print("• Trade during high volume hours (US/London overlap)")
        print("• Monitor funding rates for futures trading")
        print("• Use trailing stops after hitting first profit target")
        print("• Check on-chain metrics for long-term trends")
        print("• Combine multiple timeframes for confirmation")
    
    elif choice == 7:
        print("\n👋 Thank you for using the Advanced Trading System!")
        print("Stay safe and trade wisely! 🎯")
    
    else:
        print("❌ Invalid choice. Please run again.")

if __name__ == "__main__":
    print(f"\n{'🤖' * 30}")
    print("🤖 ADVANCED TRADING EXECUTION SYSTEM v2.0")
    print("🤖 WITH ML ENHANCEMENT & LEVERAGE MANAGEMENT")
    print(f"{'🤖' * 30}")
    
    print(f"\n📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"🕒 Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"🤖 ML Available: {ML_AVAILABLE}")
    
    # Check for required modules
    try:
        from c_signal import fetch_price_data, calculate_technical_indicators, generate_trading_signals
        print("✅ c_signal module loaded")
    except ImportError:
        print("⚠️  c_signal module not found - using mock data")
    
    try:
        from onchain_analyzer2 import OnChainAnalyzer
        print("✅ onchain_analyzer2 module loaded")
    except ImportError:
        print("⚠️  onchain_analyzer2 module not found - using mock on-chain data")
    
    # Interactive mode
    interactive_trading_planner()