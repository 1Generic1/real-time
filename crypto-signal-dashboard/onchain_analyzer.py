from c_signal import fetch_price_data, calculate_signals, analyze_current_signal
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from collections import defaultdict
import time

load_dotenv()

class OnChainAnalyzer:
    def __init__(self):
        self.cryptoquant_key = os.getenv('CRYPTOQUANT_API_KEY', '')
        self.base_url = "https://api.cryptoquant.com/v1"

        # Map symbols to their on-chain identifiers
        self.symbol_map = {
            'BTC': 'btc',
            'ETH': 'eth',
            'SOL': 'sol',
            'ADA': 'ada',
            'DOT': 'dot',
            'AVAX': 'avax',
            'MATIC': 'matic',
            'LINK': 'link'
        }

        # CryptoQuant-specific metric mapping - NOW FULLY IMPLEMENTED
        self.metric_endpoints = {
            'exchange_flow': '/exchange-flows',
            'exchange_balance': '/exchange-balance',
            'whale_ratio': '/whale-ratio',
            'miner_flow': '/miner-flows',
            'funding_rate': '/funding-rates'
        }

        # COMPLETE Dynamic threshold system
        self.historical_flows = defaultdict(list)
        self.min_data_points = 15
        self.volatility_data = defaultdict(list)

    def make_cryptoquant_request(self, endpoint, symbol, params=None):
        """Make authenticated request to CryptoQuant API"""
        if not self.cryptoquant_key:
            raise ValueError("CryptoQuant API key not found")
        
        url = f"{self.base_url}/{symbol}{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.cryptoquant_key}',
            'Accept': 'application/json'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ CryptoQuant API error: {e}")
            return None

    # NEW: IMPLEMENT ALL METRIC FUNCTIONS
    def get_exchange_balance(self, symbol='BTC'):
        """Get total coins on exchanges - bearish when high"""
        api_symbol = self.symbol_map.get(symbol.upper(), symbol.lower())
        
        try:
            data = self.make_cryptoquant_request('/exchange-balance', api_symbol, {
                'limit': 1
            })
            
            if data and 'result' in data and 'data' in data['result'] and data['result']['data']:
                balance_data = data['result']['data'][0]
                return {
                    'symbol': symbol,
                    'balance': balance_data.get('balance', 0),
                    'timestamp': datetime.now(),
                    'source': 'CryptoQuant'
                }
        except Exception as e:
            print(f"Error fetching exchange balance for {symbol}: {e}")
        
        return {'symbol': symbol, 'balance': 0, 'source': 'Error'}

    def get_whale_ratio(self, symbol='BTC'):
        """Get whale transaction ratio - high ratio can indicate manipulation"""
        api_symbol = self.symbol_map.get(symbol.upper(), symbol.lower())
        
        try:
            data = self.make_cryptoquant_request('/whale-ratio', api_symbol, {
                'limit': 1
            })
            
            if data and 'result' in data and 'data' in data['result'] and data['result']['data']:
                whale_data = data['result']['data'][0]
                return {
                    'symbol': symbol,
                    'ratio': whale_data.get('ratio', 0),
                    'timestamp': datetime.now(),
                    'source': 'CryptoQuant'
                }
        except Exception as e:
            print(f"Error fetching whale ratio for {symbol}: {e}")
        
        return {'symbol': symbol, 'ratio': 0, 'source': 'Error'}

    def get_miner_flow(self, symbol='BTC'):
        """Get miner flows - miner selling can be bearish"""
        api_symbol = self.symbol_map.get(symbol.upper(), symbol.lower())
        
        try:
            data = self.make_cryptoquant_request('/miner-flows', api_symbol, {
                'limit': 1
            })
            
            if data and 'result' in data and 'data' in data['result'] and data['result']['data']:
                miner_data = data['result']['data'][0]
                return {
                    'symbol': symbol,
                    'miner_flow': miner_data.get('miner_flow', 0),
                    'miner_to_exchange': miner_data.get('miner_to_exchange_flow', 0),
                    'timestamp': datetime.now(),
                    'source': 'CryptoQuant'
                }
        except Exception as e:
            print(f"Error fetching miner flow for {symbol}: {e}")
        
        return {'symbol': symbol, 'miner_flow': 0, 'miner_to_exchange': 0, 'source': 'Error'}

    def get_funding_rate(self, symbol='BTC'):
        """Get funding rates - extreme rates can indicate market sentiment"""
        api_symbol = self.symbol_map.get(symbol.upper(), symbol.lower())
        
        try:
            data = self.make_cryptoquant_request('/funding-rates', api_symbol, {
                'limit': 1,
                'exchange': 'binance'  # Use Binance as default
            })
            
            if data and 'result' in data and 'data' in data['result'] and data['result']['data']:
                funding_data = data['result']['data'][0]
                return {
                    'symbol': symbol,
                    'funding_rate': funding_data.get('funding_rate', 0),
                    'exchange': funding_data.get('exchange', 'binance'),
                    'timestamp': datetime.now(),
                    'source': 'CryptoQuant'
                }
        except Exception as e:
            print(f"Error fetching funding rate for {symbol}: {e}")
        
        return {'symbol': symbol, 'funding_rate': 0, 'source': 'Error'}

    def get_exchange_flow(self, symbol='BTC', window=30):
        """Get REAL exchange flow data from CryptoQuant"""
        api_symbol = self.symbol_map.get(symbol.upper(), symbol.lower())
        
        if not self.cryptoquant_key:
            print(f"❌ No CryptoQuant API key found. Please add CRYPTOQUANT_API_KEY to your .env file")
            return self.get_mock_flow_data(symbol)

        print(f"📡 Fetching real exchange flow data for {symbol} from CryptoQuant...")
        
        try:
            data = self.make_cryptoquant_request('/exchange-flows', api_symbol, {
                'window': '1d',
                'limit': 1
            })
            
            if not data or 'result' not in data or 'data' not in data['result']:
                print(f"⚠️  No real data returned for {symbol}, using mock data")
                return self.get_mock_flow_data(symbol)
            
            flow_data = data['result']['data']
            if not flow_data:
                print(f"⚠️  Empty data for {symbol}, using mock data")
                return self.get_mock_flow_data(symbol)
            
            latest_flow = flow_data[0]
            
            real_flow_data = {
                'symbol': symbol,
                'net_flow': latest_flow.get('netflow', 0),
                'inflow': latest_flow.get('inflow', 0),
                'outflow': latest_flow.get('outflow', 0),
                'timestamp': datetime.now(),
                'source': 'CryptoQuant'
            }

            # Store for COMPLETE dynamic threshold calculation
            self.add_flow_data(real_flow_data)
            
            print(f"✅ Successfully fetched real data for {symbol}")
            return real_flow_data

        except Exception as e:
            print(f"❌ Error fetching real data for {symbol}: {e}")
            return self.get_mock_flow_data(symbol)

    def get_comprehensive_onchain_analysis(self, symbol='BTC'):
        """NEW: Get ALL on-chain metrics for comprehensive analysis"""
        print(f"📊 Fetching comprehensive on-chain data for {symbol}...")
        
        metrics = {}
        
        # Get all available metrics
        metrics['exchange_flow'] = self.get_exchange_flow(symbol)
        metrics['exchange_balance'] = self.get_exchange_balance(symbol)
        metrics['whale_ratio'] = self.get_whale_ratio(symbol)
        metrics['miner_flow'] = self.get_miner_flow(symbol)
        metrics['funding_rate'] = self.get_funding_rate(symbol)
        
        return metrics

    def analyze_comprehensive_signals(self, metrics):
        """NEW: Generate signals from ALL on-chain metrics"""
        symbol = metrics['exchange_flow']['symbol']
        signals = []
        
        # 1. Exchange Flow Analysis
        flow_data = metrics['exchange_flow']
        flow_signal = self.analyze_flow_signal(flow_data)
        signals.append(f"💰 Exchange Flow: {flow_signal}")
        
        # 2. Exchange Balance Analysis
        balance_data = metrics['exchange_balance']
        balance = balance_data['balance']
        if balance > self._get_balance_threshold(symbol):
            signals.append(f"🏦 Exchange Balance: 🔴 BEARISH (High: {balance:,.0f})")
        else:
            signals.append(f"🏦 Exchange Balance: 🟢 BULLISH (Low: {balance:,.0f})")
        
        # 3. Whale Ratio Analysis
        whale_data = metrics['whale_ratio']
        whale_ratio = whale_data['ratio']
        if whale_ratio > 0.8:  # High whale activity
            signals.append(f"🐋 Whale Ratio: ⚠️  CAUTION (High: {whale_ratio:.3f})")
        else:
            signals.append(f"🐋 Whale Ratio: ✅ NORMAL ({whale_ratio:.3f})")
        
        # 4. Miner Flow Analysis
        miner_data = metrics['miner_flow']
        miner_to_exchange = miner_data['miner_to_exchange']
        if miner_to_exchange > self._get_miner_threshold(symbol):
            signals.append(f"⛏️  Miner Flow: 🔴 BEARISH (Selling: {miner_to_exchange:,.0f})")
        else:
            signals.append(f"⛏️  Miner Flow: 🟢 BULLISH (Low selling: {miner_to_exchange:,.0f})")
        
        # 5. Funding Rate Analysis
        funding_data = metrics['funding_rate']
        funding_rate = funding_data['funding_rate']
        if funding_rate > 0.01:  # Extremely positive funding
            signals.append(f"📈 Funding Rate: 🔴 BEARISH (High: {funding_rate:.4%})")
        elif funding_rate < -0.01:  # Extremely negative funding
            signals.append(f"📈 Funding Rate: 🟢 BULLISH (Low: {funding_rate:.4%})")
        else:
            signals.append(f"📈 Funding Rate: ✅ NORMAL ({funding_rate:.4%})")
        
        return signals

    def _get_balance_threshold(self, symbol):
        """Threshold for exchange balance analysis"""
        balance_thresholds = {
            'BTC': 2000000,   # 2M BTC on exchanges
            'ETH': 15000000,  # 15M ETH on exchanges
            'SOL': 50000000,  # 50M SOL on exchanges
        }
        return balance_thresholds.get(symbol, 1000000)

    def _get_miner_threshold(self, symbol):
        """Threshold for miner flow analysis"""
        miner_thresholds = {
            'BTC': 1000,     # 1000 BTC miner selling
            'ETH': 50000,    # 50K ETH miner selling
        }
        return miner_thresholds.get(symbol, 1000)

    # COMPLETE DYNAMIC THRESHOLD SYSTEM - ALL METHODS IMPLEMENTED
    def add_flow_data(self, flow_data):
        """Store historical flow data for dynamic threshold calculation"""
        symbol = flow_data['symbol']
        self.historical_flows[symbol].append(flow_data)

        # Update volatility data
        self.volatility_data[symbol].append(abs(flow_data['net_flow']))

        # Keep only last 60 days of data to stay current
        if len(self.historical_flows[symbol]) > 60:
            self.historical_flows[symbol].pop(0)
        if len(self.volatility_data[symbol]) > 30:
            self.volatility_data[symbol].pop(0)

    def calculate_dynamic_threshold(self, symbol, timeframe='1d'):
        """
        COMPLETE: Calculate optimal threshold using ALL FOUR methods combined
        """
        # 1. Dynamic Calculation (Base)
        dynamic_thresh = self._calculate_dynamic_base(symbol)

        # 2. Historical Analysis (Constraints)
        historical_thresh = self._apply_historical_constraints(dynamic_thresh, symbol)

        # 3. Timeframe Adjustment
        timeframe_thresh = self._adjust_for_timeframe(historical_thresh, timeframe)

        # 4. Volatility Adjustment
        final_threshold = self._adjust_for_volatility(timeframe_thresh, symbol)

        return final_threshold

    def _calculate_dynamic_base(self, symbol):
        """1. Dynamic calculation based on recent flows"""
        if symbol not in self.historical_flows or len(self.historical_flows[symbol]) < self.min_data_points:
            return self._get_default_threshold(symbol)
        
        flows = self.historical_flows[symbol]
        net_flows = [f['net_flow'] for f in flows]

        # Use IQR method for robust threshold calculation
        Q1 = np.percentile(net_flows, 25)
        Q3 = np.percentile(net_flows, 75)
        IQR = Q3 - Q1

        dynamic_threshold = Q3 + 1.5 * IQR

        return max(dynamic_threshold, 500)  # Minimum threshold

    def _apply_historical_constraints(self, threshold, symbol):
        """2. Ensure thresholds stay within reasonable historical ranges"""
        historical_ranges = {
            'BTC': (1000, 5000),      # Based on BTC historical analysis
            'ETH': (20000, 150000),   # ETH typically has higher flows
            'SOL': (1000, 20000),     # SOL more volatile
            'ADA': (100000, 2000000), # ADA has large quantities
            'DOT': (50000, 500000),   # DOT typical ranges
            'AVAX': (5000, 50000),    # AVAX ranges
            'MATIC': (100000, 1000000), # MATIC quantities
            'LINK': (50000, 300000)   # LINK ranges
        }

        min_thresh, max_thresh = historical_ranges.get(symbol, (1000, 10000))
        return max(min_thresh, min(threshold, max_thresh))

    def _adjust_for_timeframe(self, threshold, timeframe):
        """3. Adjust threshold based on analysis timeframe"""
        timeframe_multipliers = {
            '1h': 0.3,   # Lower threshold for hourly (more sensitive)
            '4h': 0.6,   # Medium sensitivity
            '1d': 1.0,   # Base threshold for daily
            '1w': 2.0,   # Higher threshold for weekly (less noise)
        }

        multiplier = timeframe_multipliers.get(timeframe, 1.0)
        return threshold * multiplier

    def _adjust_for_volatility(self, threshold, symbol):
        """4. Adjust for current market volatility"""
        if symbol not in self.volatility_data or len(self.volatility_data[symbol]) < 10:
            return threshold
        
        # Calculate volatility from flow data
        recent_volatility = self.volatility_data[symbol][-10:]
        
        if len(recent_volatility) < 5:
            return threshold

        current_vol = np.std(recent_volatility)
        avg_vol = np.mean(recent_volatility)

        if avg_vol == 0:
            return threshold

        vol_ratio = current_vol / avg_vol

        # Adjust threshold based on volatility
        if vol_ratio > 2.0:    # Extreme volatility
            return threshold * 1.5
        elif vol_ratio > 1.5:  # High volatility
            return threshold * 1.3
        elif vol_ratio > 1.2:  # Elevated volatility
            return threshold * 1.1
        elif vol_ratio < 0.7:  # Low volatility
            return threshold * 0.8
        else:                  # Normal volatility
            return threshold

    def _get_default_threshold(self, symbol):
        """Fallback thresholds when not enough data"""
        defaults = {
            'BTC': 2000,
            'ETH': 50000,
            'SOL': 5000,
            'ADA': 500000,
            'DOT': 100000,
            'AVAX': 20000,
            'MATIC': 300000,
            'LINK': 100000
        }
        return defaults.get(symbol, 3000)

    def get_mock_flow_data(self, symbol='BTC'):
        """Mock data fallback when API fails"""
        import random
        flow_ranges = {
            'BTC': (-5000, 5000),
            'ETH': (-100000, 100000),
            'SOL': (-10000, 10000),
            'ADA': (-1000000, 1000000),
            'DOT': (-200000, 200000),
            'AVAX': (-50000, 50000),
            'MATIC': (-500000, 500000),
            'LINK': (-200000, 200000)
        }

        min_flow, max_flow = flow_ranges.get(symbol, (-5000, 5000))

        flow_data = {
            'symbol': symbol,
            'net_flow': random.randint(min_flow, max_flow),
            'inflow': random.randint(abs(min_flow)//2, abs(max_flow)//2),
            'outflow': random.randint(abs(min_flow)//2, abs(max_flow)//2),
            'timestamp': datetime.now(),
            'source': 'Mock Data'
        }

        # Store for dynamic thresholds
        self.add_flow_data(flow_data)

        return flow_data

    def analyze_flow_signal(self, flow_data, timeframe='1d'):
        """
        Generate trading signal with COMPLETE dynamic thresholds
        """
        symbol = flow_data['symbol']
        net_flow = flow_data['net_flow']
        source = flow_data.get('source', 'Unknown')

        # Get COMPLETE dynamic threshold (all 4 methods)
        threshold = self.calculate_dynamic_threshold(symbol, timeframe)

        # Determine threshold type and additional info
        data_points = len(self.historical_flows.get(symbol, []))
        if data_points >= self.min_data_points:
            threshold_type = f"Dynamic ({data_points} pts)"
            # Calculate which factors influenced the threshold
            factors = self._get_threshold_factors(symbol, timeframe, threshold)
            threshold_info = f"{threshold_type} - {factors}"
        else:
            threshold_type = f"Default (need {self.min_data_points - data_points} more)"
            threshold_info = threshold_type

        # Generate signal
        if net_flow < -threshold:
            return f"🟢 BULLISH: Strong outflow ({abs(net_flow):,.0f} > {threshold:,.0f}) [{threshold_info}]"
        elif net_flow > threshold:
            return f"🔴 BEARISH: Strong inflow ({net_flow:,.0f} > {threshold:,.0f}) [{threshold_info}]"
        else:
            return f"🟡 NEUTRAL: Balanced flows (±{threshold:,.0f}) [{threshold_info}]"

    def _get_threshold_factors(self, symbol, timeframe, final_threshold):
        """Explain which factors influenced the threshold calculation"""
        factors = []
        
        # Check timeframe factor
        base_threshold = self._calculate_dynamic_base(symbol)
        if timeframe != '1d':
            timeframe_effect = final_threshold / base_threshold
            factors.append(f"TF:{timeframe_effect:.1f}x")
        
        # Check volatility factor
        vol_adjusted = self._adjust_for_volatility(base_threshold, symbol)
        if vol_adjusted != base_threshold:
            vol_effect = vol_adjusted / base_threshold
            factors.append(f"Vol:{vol_effect:.1f}x")
        
        return " + ".join(factors) if factors else "Base dynamic"

    def get_threshold_info(self, symbol, timeframe='1d'):
        """Get detailed information about threshold calculation"""
        base_threshold = self._calculate_dynamic_base(symbol)
        historical_threshold = self._apply_historical_constraints(base_threshold, symbol)
        timeframe_threshold = self._adjust_for_timeframe(historical_threshold, timeframe)
        final_threshold = self._adjust_for_volatility(timeframe_threshold, symbol)
        
        data_points = len(self.historical_flows.get(symbol, []))

        return {
            'symbol': symbol,
            'base_threshold': base_threshold,
            'historical_constrained': historical_threshold,
            'timeframe_adjusted': timeframe_threshold,
            'final_threshold': final_threshold,
            'data_points': data_points,
            'min_data_points': self.min_data_points,
            'threshold_type': 'Dynamic' if data_points >= self.min_data_points else 'Default',
            'needed_points': max(0, self.min_data_points - data_points),
            'timeframe': timeframe
        }

# ENHANCED analysis function with COMPREHENSIVE on-chain data
def analyze_crypto_comprehensive(symbol='BTC/USDT', timeframe='1d'):
    """
    Analyze cryptocurrency with COMPREHENSIVE on-chain data from CryptoQuant
    """
    print(f"🔄 Analyzing {symbol} ({timeframe}) with COMPREHENSIVE on-chain data...")

    # Technical Analysis
    price_data = fetch_price_data(symbol, timeframe, 100)
    if price_data is None:
        print(f"❌ Could not fetch price data for {symbol}")
        return

    price_data = calculate_signals(price_data)
    ta_signals = analyze_current_signal(price_data)

    # COMPREHENSIVE On-Chain Analysis
    base_symbol = symbol.split('/')[0]
    onchain = OnChainAnalyzer()
    
    # Get ALL on-chain metrics
    all_metrics = onchain.get_comprehensive_onchain_analysis(base_symbol)
    comprehensive_signals = onchain.analyze_comprehensive_signals(all_metrics)
    
    # Get detailed threshold information for exchange flows
    threshold_info = onchain.get_threshold_info(base_symbol, timeframe)
    
    # Display Results
    print(f"\n{'='*60}")
    print(f"🎯 {symbol} COMPREHENSIVE TRADING SIGNALS - {timeframe}")
    print(f"{'='*60}")
    
    print(f"📊 Technical Analysis:")
    for signal in ta_signals:
        print(f"   • {signal}")

    print(f"\n⛓️  COMPREHENSIVE On-Chain Analysis:")
    for signal in comprehensive_signals:
        print(f"   • {signal}")

    print(f"\n🎯 Exchange Flow Threshold Details:")
    print(f"   • Base (IQR): {threshold_info['base_threshold']:,.0f}")
    print(f"   • Timeframe Adjusted: {threshold_info['timeframe_adjusted']:,.0f}")
    print(f"   • Final Threshold: {threshold_info['final_threshold']:,.0f}")
    print(f"   • Data Points: {threshold_info['data_points']}/15")

    # Generate enhanced final recommendation
    generate_comprehensive_recommendation(ta_signals, comprehensive_signals)

def generate_comprehensive_recommendation(ta_signals, onchain_signals):
    """Generate final recommendation considering ALL signals"""
    # Count bullish vs bearish signals
    bullish_count = 0
    bearish_count = 0
    total_signals = len(onchain_signals)
    
    for signal in onchain_signals:
        if "BULLISH" in signal:
            bullish_count += 1
        elif "BEARISH" in signal:
            bearish_count += 1
    
    # Technical Analysis sentiment
    bullish_ta = any("BUY" in str(s) or "bullish" in str(s).lower() for s in ta_signals)
    
    # On-chain sentiment (majority vote)
    onchain_bullish = bullish_count > bearish_count
    onchain_neutral = bullish_count == bearish_count
    
    print(f"\n📊 Signal Summary:")
    print(f"   • Bullish On-Chain Signals: {bullish_count}/{total_signals}")
    print(f"   • Bearish On-Chain Signals: {bearish_count}/{total_signals}")
    print(f"   • Technical Analysis: {'Bullish' if bullish_ta else 'Bearish'}")
    
    print(f"\n💡 FINAL RECOMMENDATION:")
    if bullish_ta and onchain_bullish:
        print("   🚀 STRONG BUY: Both TA and majority of On-Chain signals are bullish!")
    elif not bullish_ta and not onchain_bullish and not onchain_neutral:
        print("   🛑 STRONG SELL: Both TA and majority of On-Chain signals are bearish!")
    elif bullish_ta and onchain_neutral:
        print("   📗 CAUTIOUS BUY: TA bullish, On-Chain mixed")
    elif not bullish_ta and onchain_neutral:
        print("   📘 CAUTIOUS SELL: TA bearish, On-Chain mixed")
    elif bullish_ta:
        print("   📈 MODERATE BUY: TA bullish, On-Chain slightly bearish")
    else:
        print("   📉 MODERATE SELL: TA bearish, On-Chain slightly bullish")

# Test with comprehensive analysis
if __name__ == "__main__":
    print("🚀 CRYPTO TRADING SIGNAL BOT WITH COMPREHENSIVE ON-CHAIN ANALYSIS")
    print("=================================================================")
    
    cryptocurrencies = ['BTC/USDT', 'ETH/USDT']
    timeframes = ['1d']  # Start with daily to conserve API calls
    
    for crypto in cryptocurrencies:
        for tf in timeframes:
            analyze_crypto_comprehensive(crypto, tf)
            print("\n" + "="*60 + "\n")
            time.sleep(2)  # Be respectful to API limits