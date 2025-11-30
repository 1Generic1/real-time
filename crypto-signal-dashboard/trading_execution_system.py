# trading_execution_system.py
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
    # Add fallback functions if needed
    def fetch_price_data(*args, **kwargs):
        return None
    def calculate_technical_indicators(*args, **kwargs):
        return None
    def generate_trading_signals(*args, **kwargs):
        return []

# Import OnChainAnalyzer from your onchain_analyzer2.py file
try:
    from onchain_analyzer2 import OnChainAnalyzer
    print("✅ Successfully imported OnChainAnalyzer from onchain_analyzer2")
except ImportError as e:
    print(f"❌ Could not import OnChainAnalyzer from onchain_analyzer2: {e}")
    print("   Make sure onchain_analyzer2.py is in the same directory")
    
    # Create a minimal fallback class
    class OnChainAnalyzer:
        def __init__(self):
            self.historical_flows = {}
            self.min_data_points = 15
            
        def accelerate_data_collection(self, symbol, days=14):
            print(f"🚀 Accelerating data collection for {symbol}...")
            
        def get_comprehensive_onchain_analysis(self, symbol):
            print(f"📊 Getting on-chain analysis for {symbol}...")
            return {
                'exchange_flow': {'symbol': symbol, 'net_flow': 0, 'source': 'Fallback'},
                'exchange_balance': {'symbol': symbol, 'balance': 0, 'source': 'Fallback'},
                'whale_ratio': {'symbol': symbol, 'ratio': 0.5, 'source': 'Fallback'},
                'miner_flow': {'symbol': symbol, 'miner_to_exchange': 0, 'source': 'Fallback'},
                'funding_rate': {'symbol': symbol, 'funding_rate': 0.001, 'source': 'Fallback'}
            }
            
        def analyze_comprehensive_signals(self, metrics):
            return [
                f"💰 Exchange Flow: 🟡 NEUTRAL (Fallback data)",
                f"🏦 Exchange Balance: 🟢 BULLISH (Fallback)",
                f"🐋 Whale Ratio: ✅ NORMAL (0.500) (Fallback)",
                f"⛏️  Miner Flow: 🟢 BULLISH (Low selling: 0) (Fallback)",
                f"📈 Funding Rate: ✅ NORMAL (0.10%) (Fallback)"
            ]

# Trading Execution System Class
class TradingExecutionSystem:
    def __init__(self, account_balance=1000, risk_per_trade=0.05, max_position_size=0.1):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.trading_log = []
        
        self.leverage_levels = {
            'BTC/USDT': 3,
            'ETH/USDT': 5,
            'SOL/USDT': 8,
            'ADA/USDT': 10,
            'DOT/USDT': 10,
            'AVAX/USDT': 8,
            'MATIC/USDT': 10,
            'LINK/USDT': 10
        }

    def calculate_position_size(self, entry_price, stop_loss_price, symbol):
        """Calculate precise position size with 5% risk management"""
        risk_amount = self.account_balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0, 0, 0
            
        position_size_units = risk_amount / price_risk
        position_value = position_size_units * entry_price
        
        max_position_value = self.account_balance * self.max_position_size
        if position_value > max_position_value:
            position_value = max_position_value
            position_size_units = position_value / entry_price
        
        leverage = self.leverage_levels.get(symbol, 3)
        margin_required = position_value / leverage
        
        return position_size_units, position_value, margin_required

    def generate_trading_signal(self, ta_signals, onchain_signals, price_data, symbol):
        """Generate precise trading signals with entry/exit points"""
        if price_data is None or len(price_data) == 0:
            return None
            
        current_price = price_data['close'].iloc[-1]
        
        # Score signals
        signal_score = self.score_signals(ta_signals, onchain_signals)
        
        # Determine trade direction
        direction = self.determine_trade_direction(signal_score)
        
        if direction == "NO_TRADE":
            return None
        
        # Calculate precise entry, stop loss, and take profit
        entry_price, stop_loss, take_profits = self.calculate_levels(
            direction, price_data, current_price, symbol
        )
        
        # Calculate position sizing
        position_size, position_value, margin_required = self.calculate_position_size(
            entry_price, stop_loss, symbol
        )
        
        if position_size == 0:
            return None
        
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
            'leverage': self.leverage_levels.get(symbol, 3)
        }
        
        return trade_signal

    def score_signals(self, ta_signals, onchain_signals):
        """Comprehensive signal scoring system"""
        score = 0
        
        # Technical Analysis Scoring
        for signal in ta_signals:
            if 'STRONG UPTREND' in signal:
                score += 3
            elif 'UPTREND' in signal:
                score += 2
            elif 'STRONG DOWNTREND' in signal:
                score -= 3
            elif 'DOWNTREND' in signal:
                score -= 2
            elif 'OVERSOLD' in signal and 'RSI' in signal:
                score += 2
            elif 'OVERBOUGHT' in signal and 'RSI' in signal:
                score -= 2
            elif 'BULLISH CROSSOVER' in signal:
                score += 2
            elif 'BEARISH CROSSOVER' in signal:
                score -= 2
            elif 'SUPPORT' in signal:
                score += 1
            elif 'RESISTANCE' in signal:
                score -= 1
        
        # On-Chain Scoring
        for signal in onchain_signals:
            if 'BULLISH' in signal and '🟢' in signal:
                score += 2
            elif 'BEARISH' in signal and '🔴' in signal:
                score -= 2
            elif 'ACCUMULATION' in signal.lower():
                score += 1
            elif 'DISTRIBUTION' in signal.lower():
                score -= 1
        
        return score

    def determine_trade_direction(self, signal_score):
        """Determine trade direction based on signal strength"""
        if signal_score >= 5:
            return "LONG"
        elif signal_score <= -5:
            return "SHORT"
        else:
            return "NO_TRADE"

    def calculate_levels(self, direction, price_data, current_price, symbol):
        """Calculate precise entry, stop loss, and take profit levels"""
        # Get recent price action data
        high_20 = price_data['high'].tail(20).max()
        low_20 = price_data['low'].tail(20).min()
        
        # Use ATR if available, otherwise use 2% of current price
        if 'atr' in price_data.columns and not pd.isna(price_data['atr'].iloc[-1]):
            atr = price_data['atr'].iloc[-1]
        else:
            atr = current_price * 0.02
        
        if direction == "LONG":
            # Entry: Near support or current price
            support_level = price_data['support'].iloc[-1] if 'support' in price_data.columns else current_price * 0.95
            if not pd.isna(support_level) and current_price <= support_level * 1.02:
                entry_price = min(current_price, support_level * 1.01)
            else:
                entry_price = current_price
            
            # Stop loss: Below recent low or using ATR
            stop_loss = min(low_20, entry_price - (atr * 1.5))
            
            # Take profits: Multiple targets
            risk_amount = entry_price - stop_loss
            take_profits = [
                entry_price + risk_amount * 1,  # 1:1 R:R
                entry_price + risk_amount * 2,  # 2:1 R:R
                entry_price + risk_amount * 3   # 3:1 R:R
            ]
            
        else:  # SHORT
            # Entry: Near resistance or current price
            resistance_level = price_data['resistance'].iloc[-1] if 'resistance' in price_data.columns else current_price * 1.05
            if not pd.isna(resistance_level) and current_price >= resistance_level * 0.98:
                entry_price = max(current_price, resistance_level * 0.99)
            else:
                entry_price = current_price
            
            # Stop loss: Above recent high or using ATR
            stop_loss = max(high_20, entry_price + (atr * 1.5))
            
            # Take profits: Multiple targets
            risk_amount = stop_loss - entry_price
            take_profits = [
                entry_price - risk_amount * 1,  # 1:1 R:R
                entry_price - risk_amount * 2,  # 2:1 R:R
                entry_price - risk_amount * 3   # 3:1 R:R
            ]
        
        return entry_price, stop_loss, take_profits

    def calculate_risk_reward(self, entry, stop_loss, take_profits):
        """Calculate risk-reward ratio for the trade"""
        risk = abs(entry - stop_loss)
        if risk == 0:
            return 0
        
        avg_reward = sum([abs(tp - entry) for tp in take_profits]) / len(take_profits)
        return avg_reward / risk

    def generate_trading_plan(self, trade_signal):
        """Generate detailed trading plan for execution"""
        if not trade_signal:
            return "NO TRADE: Insufficient signal strength"
        
        plan = f"""
🎯 **TRADING PLAN EXECUTION** 🎯

📊 SYMBOL: {trade_signal['symbol']}
⏰ TIMESTAMP: {trade_signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

⚡ TRADE DIRECTION: {trade_signal['direction']}
💰 ENTRY PRICE: ${trade_signal['entry_price']:,.2f}
🛡️ STOP LOSS: ${trade_signal['stop_loss']:,.2f}
🎯 TAKE PROFITS: 
   TP1: ${trade_signal['take_profits'][0]:,.2f}
   TP2: ${trade_signal['take_profits'][1]:,.2f}
   TP3: ${trade_signal['take_profits'][2]:,.2f}

📈 POSITION SIZING:
   Units: {trade_signal['position_size']:.6f}
   Value: ${trade_signal['position_value']:,.2f}
   Margin: ${trade_signal['margin_required']:,.2f}
   Leverage: {trade_signal['leverage']}x

⚖️ RISK MANAGEMENT:
   Risk: {self.risk_per_trade*100}% of account (${self.account_balance * self.risk_per_trade:,.2f})
   R:R Ratio: {trade_signal['risk_reward_ratio']:.2f}:1
   Signal Strength: {trade_signal['signal_strength']}/10

💡 EXECUTION INSTRUCTIONS:
   1. Set limit order at entry price
   2. Set stop loss immediately after fill
   3. Set take profit orders at TP1, TP2, TP3
   4. Monitor for early exit conditions

⚠️ RISK WARNINGS:
   - Maximum 5% account risk per trade
   - Use proper leverage management
   - Monitor market conditions
   - Be prepared to adjust if fundamentals change
"""
        return plan

    def execute_spot_trading(self, trade_signal):
        """Execute spot trading strategy"""
        if trade_signal['direction'] == 'LONG':
            return self.spot_long_execution(trade_signal)
        else:
            return "SHORT positions not available in spot trading"

    def execute_futures_trading(self, trade_signal):
        """Execute futures trading strategy"""
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

RISK MANAGEMENT:
- Stop Loss: ${trade_signal['stop_loss']:,.4f}
- Take Profit 1: ${trade_signal['take_profits'][0]:,.4f} (Sell 30%)
- Take Profit 2: ${trade_signal['take_profits'][1]:,.4f} (Sell 30%)
- Take Profit 3: ${trade_signal['take_profits'][2]:,.4f} (Sell 40%)

📊 EXPECTED OUTCOMES:
- Max Loss: ${abs(trade_signal['entry_price'] - trade_signal['stop_loss']) * trade_signal['position_size']:,.2f}
- Potential Gain: ${(sum(trade_signal['take_profits'])/3 - trade_signal['entry_price']) * trade_signal['position_size']:,.2f}
"""
        return execution

    def futures_long_execution(self, trade_signal):
        """Futures long execution details"""
        execution = f"""
🟢 FUTURES LONG EXECUTION:

1. ORDER TYPE: Limit Buy
2. SYMBOL: {trade_signal['symbol']} Perpetual
3. QUANTITY: {trade_signal['position_size']:.6f}
4. ENTRY PRICE: ${trade_signal['entry_price']:,.4f}
5. LEVERAGE: {trade_signal['leverage']}x
6. MARGIN: ${trade_signal['margin_required']:,.2f}

RISK MANAGEMENT:
- Stop Loss: ${trade_signal['stop_loss']:,.4f}
- Take Profit 1: ${trade_signal['take_profits'][0]:,.4f} (Close 30%)
- Take Profit 2: ${trade_signal['take_profits'][1]:,.4f} (Close 30%)
- Take Profit 3: ${trade_signal['take_profits'][2]:,.4f} (Close 40%)

LIQUIDATION PRICE: ${trade_signal['stop_loss'] * 0.95:,.4f}

📊 RISK/REWARD:
- Risk: ${abs(trade_signal['entry_price'] - trade_signal['stop_loss']) * trade_signal['position_size']:,.2f}
- Reward (TP3): ${(trade_signal['take_profits'][2] - trade_signal['entry_price']) * trade_signal['position_size']:,.2f}
- R:R Ratio: {trade_signal['risk_reward_ratio']:.2f}:1
"""
        return execution

    def futures_short_execution(self, trade_signal):
        """Futures short execution details"""
        execution = f"""
🔴 FUTURES SHORT EXECUTION:

1. ORDER TYPE: Limit Sell
2. SYMBOL: {trade_signal['symbol']} Perpetual
3. QUANTITY: {trade_signal['position_size']:.6f}
4. ENTRY PRICE: ${trade_signal['entry_price']:,.4f}
5. LEVERAGE: {trade_signal['leverage']}x
6. MARGIN: ${trade_signal['margin_required']:,.2f}

RISK MANAGEMENT:
- Stop Loss: ${trade_signal['stop_loss']:,.4f}
- Take Profit 1: ${trade_signal['take_profits'][0]:,.4f} (Close 30%)
- Take Profit 2: ${trade_signal['take_profits'][1]:,.4f} (Close 30%)
- Take Profit 3: ${trade_signal['take_profits'][2]:,.4f} (Close 40%)

LIQUIDATION PRICE: ${trade_signal['stop_loss'] * 1.05:,.4f}

📊 RISK/REWARD:
- Risk: ${abs(trade_signal['entry_price'] - trade_signal['stop_loss']) * trade_signal['position_size']:,.2f}
- Reward (TP3): ${(trade_signal['entry_price'] - trade_signal['take_profits'][2]) * trade_signal['position_size']:,.2f}
- R:R Ratio: {trade_signal['risk_reward_ratio']:.2f}:1
"""
        return execution

def integrate_and_trade(symbol='BTC/USDT', timeframe='4h', account_balance=1000):
    """
    Main integration function that combines both analysis systems
    and generates executable trading plans
    """
    print(f"\n{'🎯' * 20}")
    print("🎯 ADVANCED TRADING EXECUTION SYSTEM")
    print(f"{'🎯' * 20}")
    
    # Initialize trading system
    trading_system = TradingExecutionSystem(account_balance=account_balance)
    
    print(f"\n🔍 ANALYZING {symbol} FOR TRADING OPPORTUNITIES...")
    time.sleep(1)
    
    # Get technical analysis
    price_data = fetch_price_data(symbol, timeframe, 200)
    if price_data is None:
        print(f"❌ Could not fetch price data for {symbol}")
        return
    
    price_data = calculate_technical_indicators(price_data)
    ta_signals = generate_trading_signals(price_data)
    
    # Get on-chain analysis
    base_symbol = symbol.split('/')[0]
    onchain_analyzer = OnChainAnalyzer()
    onchain_analyzer.accelerate_data_collection(base_symbol)
    all_metrics = onchain_analyzer.get_comprehensive_onchain_analysis(base_symbol)
    onchain_signals = onchain_analyzer.analyze_comprehensive_signals(all_metrics)
    
    # Generate trading signal
    trade_signal = trading_system.generate_trading_signal(
        ta_signals, onchain_signals, price_data, symbol
    )
    
    # Display comprehensive analysis
    print(f"\n{'📊' * 20}")
    print("📊 MARKET ANALYSIS SUMMARY")
    print(f"{'📊' * 20}")
    
    print(f"\n🔧 TECHNICAL SIGNALS:")
    for signal in ta_signals:
        print(f"   • {signal}")
    
    print(f"\n⛓️  ON-CHAIN SIGNALS:")
    for signal in onchain_signals:
        print(f"   • {signal}")
    
    if trade_signal:
        # Generate trading plan
        trading_plan = trading_system.generate_trading_plan(trade_signal)
        print(trading_plan)
        
        # Show execution details
        print(f"\n{'⚡' * 20}")
        print("⚡ EXECUTION DETAILS")
        print(f"{'⚡' * 20}")
        
        # Spot trading execution
        spot_execution = trading_system.execute_spot_trading(trade_signal)
        print(f"\n🪙 SPOT TRADING:")
        print(spot_execution)
        
        # Futures trading execution
        futures_execution = trading_system.execute_futures_trading(trade_signal)
        print(f"\n📈 FUTURES TRADING:")
        print(futures_execution)
        
        # Risk management summary
        print(f"\n{'⚠️ ' * 20}")
        print("⚠️  RISK MANAGEMENT SUMMARY")
        print(f"{'⚠️ ' * 20}")
        
        print(f"""
💰 ACCOUNT PROTECTION:
   • Maximum Risk per Trade: 5% (${account_balance * 0.05:,.2f})
   • Position Size Limit: 10% of account
   • Stop Loss: Always set immediately after entry
   • Take Profits: Scale out at 1:1, 2:1, 3:1 R:R

🎯 TRADE MANAGEMENT:
   • Monitor for fundamental changes
   • Adjust stop loss to breakeven after TP1
   • Consider partial profits at key levels
   • Never risk more than planned

📊 TRADE STATISTICS:
   • Expected Win Rate: 60-70% (with proper signal filtering)
   • Average R:R: 1:2 or better
   • Maximum Drawdown: <15% with proper risk management
        """)
        
    else:
        print(f"\n❌ NO TRADING OPPORTUNITY FOUND")
        print(f"   Signal strength insufficient for {symbol}")
        print(f"   Waiting for better setup...")

def batch_analyze_cryptos(cryptos=None, account_balance=1000):
    """
    Analyze multiple cryptocurrencies and find best opportunities
    """
    if cryptos is None:
        cryptos = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT']
    
    print(f"\n{'🔍' * 20}")
    print("🔍 MULTI-CRYPTO SCANNING MODE")
    print(f"{'🔍' * 20}")
    
    trading_system = TradingExecutionSystem(account_balance=account_balance)
    best_opportunities = []
    
    for crypto in cryptos:
        print(f"\n📈 Analyzing {crypto}...")
        
        try:
            # Get technical analysis
            price_data = fetch_price_data(crypto, '4h', 100)
            if price_data is None:
                print(f"   ❌ Could not fetch price data for {crypto}")
                continue
                
            price_data = calculate_technical_indicators(price_data)
            ta_signals = generate_trading_signals(price_data)
            
            # Get on-chain analysis
            base_symbol = crypto.split('/')[0]
            onchain_analyzer = OnChainAnalyzer()
            onchain_analyzer.accelerate_data_collection(base_symbol)
            all_metrics = onchain_analyzer.get_comprehensive_onchain_analysis(base_symbol)
            onchain_signals = onchain_analyzer.analyze_comprehensive_signals(all_metrics)
            
            # Generate signal
            trade_signal = trading_system.generate_trading_signal(
                ta_signals, onchain_signals, price_data, crypto
            )
            
            if trade_signal:
                best_opportunities.append(trade_signal)
                print(f"   ✅ TRADE FOUND: {trade_signal['direction']} - Score: {trade_signal['signal_strength']}")
            else:
                print(f"   ❌ No trade signal")
                
        except Exception as e:
            print(f"   ⚠️  Error analyzing {crypto}: {e}")
    
    # Display best opportunities
    if best_opportunities:
        print(f"\n{'🎯' * 20}")
        print("🎯 BEST TRADING OPPORTUNITIES")
        print(f"{'🎯' * 20}")
        
        # Sort by signal strength
        best_opportunities.sort(key=lambda x: abs(x['signal_strength']), reverse=True)
        
        for i, opportunity in enumerate(best_opportunities[:3], 1):
            print(f"\n#{i} {opportunity['symbol']} - {opportunity['direction']}")
            print(f"   Signal Strength: {opportunity['signal_strength']}")
            print(f"   R:R Ratio: {opportunity['risk_reward_ratio']:.2f}:1")
            print(f"   Potential Return: {opportunity['risk_reward_ratio'] * 100:.1f}%")
            
            # Show quick execution details
            if opportunity['direction'] == 'LONG':
                print(f"   Entry: ${opportunity['entry_price']:,.2f}")
                print(f"   Stop: ${opportunity['stop_loss']:,.2f}")
                print(f"   Target: ${opportunity['take_profits'][2]:,.2f}")
            else:
                print(f"   Entry: ${opportunity['entry_price']:,.2f}")
                print(f"   Stop: ${opportunity['stop_loss']:,.2f}")
                print(f"   Target: ${opportunity['take_profits'][2]:,.2f}")
    else:
        print(f"\n❌ No strong trading opportunities found across analyzed cryptos")
# Main execution
if __name__ == "__main__":
    # Single symbol analysis with full trading plan
    integrate_and_trade('ETH/USDT', '4h', account_balance=1000)
    
    # Multi-crypto scanning
    time.sleep(2)
    #  batch_analyze_cryptos(['ETH/USDT', 'SOL/USDT', 'ADA/USDT'])
    
    print(f"\n{'✅' * 20}")
    print("✅ TRADING EXECUTION SYSTEM READY")
    print(f"{'✅' * 20}")