# ==================== BYBIT UNIVERSAL SCANNER ====================
# File: scanner.py
# Description: Universal scanner for Bybit coins with ML options
# Dependencies: trading_execution_systemsimple8.py, onchain_analyzer3.py, 
#               c_signal2.py, advanced_ml_predictorsimple5.py
# Usage: python scanner.py

import sys
import os
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import ccxt

# ==================== FIX ENCODING FOR WINDOWS ====================
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ==================== ADD CURRENT DIRECTORY TO PATH ====================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==================== IMPORT YOUR EXISTING MODULES ====================
print("📦 Loading dependencies...")

try:
    from c_signal2 import fetch_price_data, calculate_technical_indicators, generate_trading_signals
    print(f"✅ Successfully imported c_signal2 module")
except ImportError as e:
    print(f"❌ Could not import c_signal2: {e}")
    def fetch_price_data(*args, **kwargs): return None
    def calculate_technical_indicators(*args, **kwargs): return None
    def generate_trading_signals(*args, **kwargs): return []

try:
    from onchain_analyzer3 import OnChainAnalyzer
    print(f"✅ Successfully imported OnChainAnalyzer")
except ImportError as e:
    print(f"❌ Could not import OnChainAnalyzer: {e}")
    class OnChainAnalyzer:
        def __init__(self): self.historical_flows = {}
        def accelerate_data_collection(self, *args): pass
        def get_comprehensive_onchain_analysis(self, *args): return {}
        def analyze_comprehensive_signals(self, *args): return []

try:
    from trading_execution_systemsimple8 import TradingExecutionSystem, MLEnhancedTradingSystem
    print(f"✅ Successfully imported TradingExecutionSystem")
    ML_AVAILABLE = True
except ImportError as e:
    print(f"❌ Could not import TradingExecutionSystem: {e}")
    ML_AVAILABLE = False
    class TradingExecutionSystem:
        def __init__(self, *args, **kwargs): pass
        def generate_trading_signal(self, *args, **kwargs): return None
    class MLEnhancedTradingSystem:
        def __init__(self, *args, **kwargs): pass
        def train_ml_models(self, *args, **kwargs): return False
        def generate_enhanced_signal(self, *args, **kwargs): return "NO_TRADE", [], 0

try:
    from advanced_ml_predictorsimple5 import RealisticMLPredictor
    print(f"✅ Successfully imported RealisticMLPredictor")
    ADVANCED_ML_AVAILABLE = True
except ImportError as e:
    print(f"❌ Could not import RealisticMLPredictor: {e}")
    ADVANCED_ML_AVAILABLE = False

# ==================== COLOR CODES (WITHOUT EMOJIS FOR FILE SAVING) ====================
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
BLUE = '\033[94m'
WHITE = '\033[97m'
RESET = '\033[0m'
BOLD = '\033[1m'

# ==================== SCANNER ANIMATOR ====================
class ScannerAnimator:
    @staticmethod
    def clear_screen():
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def animate_title(title, delay=0.02):
        """Animate title text"""
        print(f"\n{CYAN}{'═'*80}{RESET}")
        for char in title:
            print(f"{CYAN}{char}{RESET}", end='', flush=True)
            time.sleep(delay)
        print(f"\n{CYAN}{'═'*80}{RESET}")
    
    @staticmethod
    def loading_bar(description, duration=1.0, width=40):
        """Display animated loading bar"""
        print(f"\n⏳ {description}")
        sys.stdout.write(f"[{BLUE}")
        for i in range(width):
            time.sleep(duration / width)
            sys.stdout.write("▓" if i % 2 == 0 else "▒")
            sys.stdout.flush()
        sys.stdout.write(f"{RESET}] 100%\n")
    
    @staticmethod
    def progress_percentage(description, current, total, delay=0.01):
        """Display percentage progress animation"""
        percentage = (current / total) * 100 if total > 0 else 0
        bar_length = 40
        filled_length = int(bar_length * current // total)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        color = GREEN if percentage > 66 else YELLOW if percentage > 33 else RED
        print(f"\r{description}: {color}[{bar}] {percentage:.1f}% ({current}/{total}){RESET}", end='', flush=True)
        
        if current == total:
            print()
    
    @staticmethod
    def print_section_header(title):
        """Print section header"""
        print(f"\n{MAGENTA}{'─'*60}{RESET}")
        print(f"{MAGENTA}{title}{RESET}")
        print(f"{MAGENTA}{'─'*60}{RESET}")

# ==================== PRICE DATA CACHE ====================
price_data_cache = {}
ONCHAIN_CACHE = {}
ONCHAIN_CACHE_DURATION = 300

def get_cached_or_fetch_price_data(symbol, timeframe, limit=100, cache_minutes=5):
    """Cache price data for consistency"""
    cache_key = f"{symbol}_{timeframe}_{limit}"
    current_time = datetime.now()
    
    if cache_key in price_data_cache:
        cache_time, cached_data = price_data_cache[cache_key]
        time_diff = (current_time - cache_time).total_seconds()
        
        if time_diff < cache_minutes * 60:
            return cached_data.copy()
    
    data = fetch_price_data(symbol, timeframe, limit)
    
    if data is not None:
        price_data_cache[cache_key] = (current_time, data.copy())
    return data

def get_cached_onchain_data(symbol):
    """Cache on-chain data for consistency"""
    cache_key = f"onchain_{symbol}"
    current_time = datetime.now()
    
    if cache_key in ONCHAIN_CACHE:
        cache_time, cached_data = ONCHAIN_CACHE[cache_key]
        time_diff = (current_time - cache_time).total_seconds()
        
        if time_diff < ONCHAIN_CACHE_DURATION:
            return cached_data.copy()
    
    return None

def cache_onchain_data(symbol, data):
    """Cache on-chain data"""
    cache_key = f"onchain_{symbol}"
    ONCHAIN_CACHE[cache_key] = (datetime.now(), data.copy())

# ==================== BYBIT UNIVERSAL SCANNER ====================
class BybitUniversalScanner:
    def __init__(self, account_balance=1000, risk_per_trade=0.02):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
        
        # Initialize Bybit exchange
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # Initialize trading system
        self.trading_system = TradingExecutionSystem(account_balance, risk_per_trade)
        
        # Cache for market data
        self.markets_cache = None
        self.markets_cache_time = None
        
        # Statistics
        self.scan_statistics = {
            'total_coins_scanned': 0,
            'signals_found': 0,
            'long_signals': 0,
            'short_signals': 0,
            'total_scan_time': 0,
            'coins_with_data': 0,
            'coins_failed': 0,
            'ml_used': False
        }
        
        ScannerAnimator.animate_title("🚀 BYBIT UNIVERSAL SCANNER INITIALIZED", 0.03)
        print(f"{GREEN}💰 Account Balance: ${account_balance:,.2f}{RESET}")
        print(f"{GREEN}🎯 Risk per Trade: {risk_per_trade*100}%{RESET}")
        print(f"{GREEN}🤖 ML Available: {ML_AVAILABLE}{RESET}")
    
    def get_all_bybit_coins(self, min_volume=100000, cache_minutes=5) -> List[Dict]:
        """Get all USDT trading pairs from Bybit with volume filtering"""
        try:
            # Check cache
            current_time = datetime.now()
            if (self.markets_cache is not None and 
                self.markets_cache_time is not None and
                (current_time - self.markets_cache_time).total_seconds() < cache_minutes * 60):
                print(f"{CYAN}📊 Using cached market data{RESET}")
                return self.markets_cache.copy()
            
            ScannerAnimator.loading_bar("Fetching Bybit market data", 1.0)
            
            # Load all markets
            markets = self.exchange.load_markets()
            
            usdt_pairs = []
            total_symbols = len(markets)
            
            print(f"{CYAN}🔍 Analyzing {total_symbols} symbols on Bybit...{RESET}")
            
            for i, (symbol, market) in enumerate(markets.items(), 1):
                if i % 100 == 0 or i == total_symbols:
                    ScannerAnimator.progress_percentage("Processing symbols", i, total_symbols)
                
                if not market.get('active', True):
                    continue
                
                # Check for USDT pairs
                is_usdt_pair = False
                clean_symbol = symbol
                
                if '/' in symbol:
                    if symbol.endswith('/USDT'):
                        is_usdt_pair = True
                        clean_symbol = symbol
                else:
                    if 'USDT' in symbol and not symbol.startswith('1000') and 'USDC' not in symbol:
                        if symbol.endswith('USDT') or 'USDT' in symbol.split('.')[0]:
                            is_usdt_pair = True
                            base = symbol.replace('USDT', '').replace('.P', '')
                            clean_symbol = f"{base}/USDT"
                
                if is_usdt_pair:
                    try:
                        ticker = self.exchange.fetch_ticker(symbol)
                        volume_24h = ticker.get('quoteVolume', 0)
                        last_price = ticker.get('last', 0)
                        
                        if volume_24h and volume_24h >= min_volume and last_price > 0:
                            usdt_pairs.append({
                                'symbol': clean_symbol,
                                'bybit_symbol': symbol,
                                'volume_24h': volume_24h,
                                'price': last_price,
                                'price_change_24h': ticker.get('percentage', 0),
                                'high_24h': ticker.get('high', 0),
                                'low_24h': ticker.get('low', 0)
                            })
                    except:
                        continue
            
            # Sort by volume
            usdt_pairs.sort(key=lambda x: x['volume_24h'], reverse=True)
            
            # Cache the results
            self.markets_cache = usdt_pairs.copy()
            self.markets_cache_time = datetime.now()
            
            print(f"\n{GREEN}✅ Found {len(usdt_pairs)} active USDT pairs with volume > ${min_volume:,.0f}{RESET}")
            
            if usdt_pairs:
                print(f"\n{CYAN}📈 Top 5 by volume:{RESET}")
                for i, coin in enumerate(usdt_pairs[:5], 1):
                    coin_name = coin['symbol'].split('/')[0]
                    print(f"{CYAN}   {i}. {coin_name}: ${coin['price']:,.2f} (Vol: ${coin['volume_24h']:,.0f}){RESET}")
            
            return usdt_pairs
            
        except Exception as e:
            print(f"{RED}❌ Error fetching Bybit data: {e}{RESET}")
            return []
    
    def analyze_single_coin(self, symbol: str, timeframe: str = '4h', 
                           enable_ml: bool = False) -> Optional[Dict]:
        """Analyze a single coin for trading signals"""
        try:
            coin_name = symbol.split('/')[0] if '/' in symbol else symbol.replace('USDT', '')
            
            # Step 1: Fetch price data
            price_data = get_cached_or_fetch_price_data(symbol, timeframe, 100)
            if price_data is None or len(price_data) < 20:
                return None
            
            self.scan_statistics['coins_with_data'] += 1
            
            # Step 2: Calculate indicators
            price_data = calculate_technical_indicators(price_data)
            ta_signals = generate_trading_signals(price_data)
            
            # Step 3: Get on-chain data
            onchain_signals = []
            base_symbol = symbol.split('/')[0] if '/' in symbol else symbol.replace('USDT', '')
            onchain_analyzer = OnChainAnalyzer()
            
            cached_onchain = get_cached_onchain_data(base_symbol)
            if cached_onchain is not None:
                all_metrics = cached_onchain
            else:
                try:
                    onchain_analyzer.accelerate_data_collection(base_symbol)
                    all_metrics = onchain_analyzer.get_comprehensive_onchain_analysis(base_symbol)
                    cache_onchain_data(base_symbol, all_metrics)
                except:
                    all_metrics = {}
            
            onchain_signals = onchain_analyzer.analyze_comprehensive_signals(all_metrics)
            
            # Step 4: Generate trading signal
            trade_signal = None
            
            if enable_ml and ML_AVAILABLE:
                try:
                    self.scan_statistics['ml_used'] = True
                    ml_system = MLEnhancedTradingSystem(self.account_balance)
                    ml_system.train_ml_models(price_data, symbol)
                    
                    trade_direction, ml_signals, combined_score = ml_system.generate_enhanced_signal(
                        ta_signals, onchain_signals, price_data
                    )
                    
                    if trade_direction != "NO_TRADE":
                        trade_signal = ml_system.generate_trading_signal(
                            ta_signals, onchain_signals, price_data, symbol
                        )
                        if trade_signal:
                            trade_signal['combined_score'] = combined_score
                            trade_signal['ml_signals'] = ml_signals
                except Exception as e:
                    print(f"{YELLOW}   ⚠️  ML analysis failed: {e}{RESET}")
                    trade_signal = self.trading_system.generate_trading_signal(
                        ta_signals, onchain_signals, price_data, symbol
                    )
            else:
                trade_signal = self.trading_system.generate_trading_signal(
                    ta_signals, onchain_signals, price_data, symbol
                )
            
            return trade_signal
            
        except Exception as e:
            print(f"{RED}   ❌ Error analyzing {symbol}: {e}{RESET}")
            self.scan_statistics['coins_failed'] += 1
            return None
    
    def scan_coins(self, coins_to_scan: List[str], timeframe: str = '4h', 
                  enable_ml: bool = False, min_signal_score: int = 3) -> Tuple[List[Dict], List[Dict]]:
        """
        Scan multiple coins for trading signals
        
        Returns:
            Tuple of (trading_signals, all_results)
        """
        ScannerAnimator.animate_title(f"🔍 SCANNING {len(coins_to_scan)} COINS", 0.02)
        print(f"{CYAN}🤖 ML Analysis: {'ENABLED' if enable_ml else 'DISABLED'}{RESET}")
        print(f"{CYAN}🎯 Minimum Signal Score: {min_signal_score}{RESET}")
        
        trading_signals = []
        all_results = []
        start_time = time.time()
        
        for i, symbol in enumerate(coins_to_scan, 1):
            self.scan_statistics['total_coins_scanned'] += 1
            coin_name = symbol.split('/')[0] if '/' in symbol else symbol.replace('USDT', '')
            
            ScannerAnimator.progress_percentage(f"Scanning coins", i, len(coins_to_scan))
            
            # Analyze the coin
            trade_signal = self.analyze_single_coin(symbol, timeframe, enable_ml)
            
            # Store result for ALL coins
            result = {
                'coin_name': coin_name,
                'symbol': symbol,
                'has_signal': False,
                'direction': None,
                'score': 0,
                'price': 0,
                'error': None
            }
            
            if trade_signal:
                signal_score = trade_signal.get('combined_score', trade_signal.get('signal_strength', 0))
                current_price = trade_signal.get('price_at_analysis', trade_signal.get('entry_price', 0))
                
                result.update({
                    'has_signal': True,
                    'direction': trade_signal['direction'],
                    'score': signal_score,
                    'price': current_price
                })
                
                if abs(signal_score) >= min_signal_score:
                    trade_signal['scan_timestamp'] = datetime.now()
                    trade_signal['coin_name'] = coin_name
                    
                    trading_signals.append(trade_signal)
                    self.scan_statistics['signals_found'] += 1
                    
                    if trade_signal['direction'] == "LONG":
                        self.scan_statistics['long_signals'] += 1
                    else:
                        self.scan_statistics['short_signals'] += 1
            
            all_results.append(result)
        
        # Calculate total scan time
        self.scan_statistics['total_scan_time'] = time.time() - start_time
        
        return trading_signals, all_results
    
    def display_all_coins_scanned(self, all_results: List[Dict], min_score: int = 3):
        """Display results for ALL coins scanned"""
        ScannerAnimator.animate_title("📊 ALL COINS SCANNED RESULTS", 0.02)
        
        print(f"{CYAN}🎯 Minimum Signal Score: {min_score}{RESET}")
        print(f"{CYAN}📊 Total Coins Scanned: {len(all_results)}{RESET}")
        print()
        
        # Create a table-like display
        print(f"{CYAN}{'Coin':<10} {'Signal':<12} {'Score':<8} {'Price':<15} {'Status':<15}{RESET}")
        print(f"{CYAN}{'-'*60}{RESET}")
        
        for result in all_results:
            coin_name = result['coin_name'][:9]
            
            if result['has_signal']:
                if result['score'] >= min_score:
                    if result['direction'] == "LONG":
                        signal_display = f"{GREEN}LONG{RESET}"
                        score_display = f"{GREEN}{result['score']:+d}{RESET}"
                        status = f"{GREEN}STRONG{RESET}"
                    else:
                        signal_display = f"{RED}SHORT{RESET}"
                        score_display = f"{RED}{result['score']:+d}{RESET}"
                        status = f"{RED}STRONG{RESET}"
                elif abs(result['score']) >= 1:
                    signal_display = f"{YELLOW}{result['direction']}{RESET}"
                    score_display = f"{YELLOW}{result['score']:+d}{RESET}"
                    status = f"{YELLOW}WEAK{RESET}"
                else:
                    signal_display = f"{WHITE}{result['direction']}{RESET}"
                    score_display = f"{WHITE}{result['score']:+d}{RESET}"
                    status = f"{WHITE}VERY WEAK{RESET}"
                
                price_display = f"${result['price']:,.2f}"
            else:
                signal_display = f"{WHITE}NONE{RESET}"
                score_display = f"{WHITE}0{RESET}"
                price_display = f"{WHITE}N/A{RESET}"
                status = f"{WHITE}NO SIGNAL{RESET}"
            
            print(f"{coin_name:<10} {signal_display:<12} {score_display:<8} {price_display:<15} {status:<15}")
        
        print(f"\n{CYAN}{'-'*60}{RESET}")
        
        # Count signals by strength
        strong_signals = sum(1 for r in all_results if r['has_signal'] and abs(r['score']) >= min_score)
        weak_signals = sum(1 for r in all_results if r['has_signal'] and 1 <= abs(r['score']) < min_score)
        very_weak_signals = sum(1 for r in all_results if r['has_signal'] and abs(r['score']) < 1)
        no_signals = sum(1 for r in all_results if not r['has_signal'])
        
        print(f"\n{CYAN}📈 SIGNAL DISTRIBUTION:{RESET}")
        print(f"{GREEN}   Strong Signals (≥{min_score}): {strong_signals} coins{RESET}")
        print(f"{YELLOW}   Weak Signals (1-{min_score-1}): {weak_signals} coins{RESET}")
        print(f"{WHITE}   Very Weak Signals (<1): {very_weak_signals} coins{RESET}")
        print(f"{WHITE}   No Signals: {no_signals} coins{RESET}")
    
    def display_detailed_signals(self, trading_signals: List[Dict], top_n: int = 20):
        """Display detailed trading signals"""
        if not trading_signals:
            print(f"\n{RED}❌ No strong trading signals found{RESET}")
            return
        
        ScannerAnimator.animate_title("🎯 DETAILED TRADING SIGNALS", 0.02)
        
        # Sort by signal strength
        trading_signals.sort(key=lambda x: abs(x.get('combined_score', x.get('signal_strength', 0))), reverse=True)
        
        for i, signal in enumerate(trading_signals[:top_n], 1):
            if signal['direction'] == "LONG":
                color = GREEN
                direction_emoji = "LONG"
            else:
                color = RED
                direction_emoji = "SHORT"
            
            score = signal.get('combined_score', signal.get('signal_strength', 0))
            coin_name = signal.get('coin_name', signal['symbol'].split('/')[0])
            
            print(f"\n{color}#{i} {coin_name} [{direction_emoji}]{RESET}")
            print(f"{color}   Signal Score: {score}{RESET}")
            print(f"{color}   Current Price: ${signal.get('price_at_analysis', signal.get('entry_price', 0)):,.2f}{RESET}")
            print(f"{color}   Entry: ${signal.get('entry_price', 0):,.2f}{RESET}")
            print(f"{color}   Stop Loss: ${signal.get('stop_loss', 0):,.2f}{RESET}")
            print(f"{color}   Take Profit: ${signal.get('take_profits', [0,0,0])[0]:,.2f}{RESET}")
            print(f"{color}   R:R Ratio: {signal.get('risk_reward_ratio', 0):.2f}:1{RESET}")
            
            if 'ml_signals' in signal and signal['ml_signals']:
                print(f"{MAGENTA}   ML Signals: {', '.join(signal['ml_signals'][:2])}{RESET}")
    
    def save_results_to_file(self, trading_signals: List[Dict], all_results: List[Dict], filename: str = None):
        """Save scan results to a file (ASCII-safe)"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bybit_scan_results_{timestamp}.txt"
        
        try:
            # Use UTF-8 encoding for Windows compatibility
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write(f"BYBIT SCAN RESULTS - COMPLETE REPORT\n")
                f.write(f"Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Coins Scanned: {self.scan_statistics['total_coins_scanned']}\n")
                f.write(f"ML Analysis Used: {'YES' if self.scan_statistics['ml_used'] else 'NO'}\n")
                f.write(f"Strong Signals Found: {len(trading_signals)}\n")
                f.write("="*80 + "\n\n")
                
                # Section 1: All Coins Summary
                f.write("ALL COINS SCANNED SUMMARY:\n")
                f.write("-"*80 + "\n")
                
                for result in all_results:
                    f.write(f"{result['coin_name']:<10} | ")
                    if result['has_signal']:
                        f.write(f"{result['direction']:<8} | ")
                        f.write(f"Score: {result['score']:+d} | ")
                        f.write(f"Price: ${result['price']:,.2f}\n")
                    else:
                        f.write(f"{'NO SIGNAL':<8} | ")
                        f.write(f"{'N/A':<12} | ")
                        f.write(f"{result.get('error', 'No data')}\n")
                
                f.write("\n" + "="*80 + "\n\n")
                
                # Section 2: Strong Signals
                if trading_signals:
                    f.write("STRONG TRADING SIGNALS:\n")
                    f.write("-"*80 + "\n\n")
                    
                    for i, signal in enumerate(trading_signals, 1):
                        coin_name = signal.get('coin_name', signal['symbol'].split('/')[0])
                        score = signal.get('combined_score', signal.get('signal_strength', 0))
                        
                        f.write(f"SIGNAL #{i}: {coin_name} [{signal['direction']}]\n")
                        f.write(f"  Score: {score}\n")
                        f.write(f"  Symbol: {signal['symbol']}\n")
                        f.write(f"  Current Price: ${signal.get('price_at_analysis', 0):,.2f}\n")
                        f.write(f"  Entry Price: ${signal.get('entry_price', 0):,.2f}\n")
                        f.write(f"  Stop Loss: ${signal.get('stop_loss', 0):,.2f}\n")
                        f.write(f"  Take Profits: {', '.join([f'${tp:,.2f}' for tp in signal.get('take_profits', [])])}\n")
                        f.write(f"  Risk/Reward: {signal.get('risk_reward_ratio', 0):.2f}:1\n")
                        f.write(f"  Leverage: {signal.get('leverage', 1)}x\n")
                        f.write(f"  Signal Time: {signal.get('scan_timestamp', datetime.now()).strftime('%H:%M:%S')}\n")
                        
                        if 'ml_signals' in signal:
                            f.write(f"  ML Signals: {', '.join(signal['ml_signals'])}\n")
                        
                        f.write("-"*60 + "\n")
                else:
                    f.write("NO STRONG SIGNALS FOUND\n")
                
                # Section 3: Statistics
                f.write("\n" + "="*80 + "\n")
                f.write("SCAN STATISTICS:\n")
                f.write("-"*80 + "\n")
                f.write(f"  Total coins scanned: {self.scan_statistics['total_coins_scanned']}\n")
                f.write(f"  Coins with data: {self.scan_statistics['coins_with_data']}\n")
                f.write(f"  Coins failed: {self.scan_statistics['coins_failed']}\n")
                f.write(f"  Strong LONG signals: {self.scan_statistics['long_signals']}\n")
                f.write(f"  Strong SHORT signals: {self.scan_statistics['short_signals']}\n")
                f.write(f"  Total signals found: {self.scan_statistics['signals_found']}\n")
                f.write(f"  ML used: {'YES' if self.scan_statistics['ml_used'] else 'NO'}\n")
                f.write(f"  Total scan time: {self.scan_statistics['total_scan_time']:.1f} seconds\n")
                f.write(f"  Average time per coin: {self.scan_statistics['total_scan_time']/max(1, self.scan_statistics['total_coins_scanned']):.1f} seconds\n")
            
            print(f"{GREEN}💾 Complete results saved to: {filename}{RESET}")
            
        except Exception as e:
            print(f"{RED}❌ Error saving results: {e}{RESET}")

# ==================== SCAN PRESETS WITH ML OPTIONS ====================
def quick_scan(scanner: BybitUniversalScanner, enable_ml: bool = False):
    """Quick scan with ML option"""
    ScannerAnimator.animate_title("⚡ QUICK SCAN", 0.02)
    
    print(f"{CYAN}🔍 Fetching top Bybit coins...{RESET}")
    all_coins = scanner.get_all_bybit_coins(min_volume=1000000)
    
    if not all_coins:
        print(f"{RED}❌ No coins found{RESET}")
        return [], []
    
    # Take top 20 coins
    top_coins = [coin['symbol'] for coin in all_coins[:20]]
    
    print(f"{GREEN}✅ Scanning top {len(top_coins)} coins{RESET}")
    print(f"{CYAN}🤖 ML Analysis: {'ENABLED' if enable_ml else 'DISABLED'}{RESET}")
    
    signals, all_results = scanner.scan_coins(
        coins_to_scan=top_coins,
        timeframe='4h',
        enable_ml=enable_ml,
        min_signal_score=3
    )
    
    return signals, all_results

def comprehensive_scan(scanner: BybitUniversalScanner, enable_ml: bool = True):
    """Comprehensive scan with ML option"""
    ScannerAnimator.animate_title("🔬 COMPREHENSIVE SCAN", 0.02)
    
    print(f"{CYAN}🔍 Fetching Bybit coins...{RESET}")
    all_coins = scanner.get_all_bybit_coins(min_volume=500000)
    
    if not all_coins:
        print(f"{RED}❌ No coins found{RESET}")
        return [], []
    
    # Take top 30 coins
    top_coins = [coin['symbol'] for coin in all_coins[:30]]
    
    print(f"{GREEN}✅ Scanning top {len(top_coins)} coins{RESET}")
    print(f"{CYAN}🤖 ML Analysis: {'ENABLED' if enable_ml else 'DISABLED'}{RESET}")
    
    signals, all_results = scanner.scan_coins(
        coins_to_scan=top_coins,
        timeframe='4h',
        enable_ml=enable_ml,
        min_signal_score=2
    )
    
    return signals, all_results

def custom_scan(scanner: BybitUniversalScanner):
    """Custom scan with full options"""
    ScannerAnimator.animate_title("🎯 CUSTOM SCAN", 0.02)
    
    # Get coin list
    print(f"{CYAN}Enter coins to scan (comma-separated, e.g., BTC/USDT,ETH/USDT){RESET}")
    print(f"{YELLOW}Leave empty for top 15 coins by volume{RESET}")
    
    user_input = input(f"\n{YELLOW}Coins: {RESET}").strip()
    
    if user_input:
        coins_list = [c.strip() for c in user_input.split(',')]
    else:
        all_coins = scanner.get_all_bybit_coins(min_volume=1000000)
        coins_list = [coin['symbol'] for coin in all_coins[:15]]
    
    # Get timeframe
    print(f"\n{CYAN}Select timeframe:{RESET}")
    print(f"1. 1 Hour")
    print(f"2. 4 Hours (recommended)")
    print(f"3. 1 Day")
    
    tf_choice = input(f"\n{YELLOW}Choice (1-3, default: 2): {RESET}").strip()
    timeframe = '4h'
    if tf_choice == '1':
        timeframe = '1h'
    elif tf_choice == '3':
        timeframe = '1d'
    
    # ML option
    print(f"\n{CYAN}ML Analysis:{RESET}")
    print(f"{GREEN}Y - Enable ML (more accurate, slower){RESET}")
    print(f"{YELLOW}N - Disable ML (faster, less accurate){RESET}")
    
    ml_choice = input(f"\n{YELLOW}Enable ML? (y/n, default: y): {RESET}").strip().lower()
    enable_ml = ml_choice != 'n'
    
    # Minimum score
    min_score = input(f"\n{YELLOW}Minimum signal score (default: 3): {RESET}").strip()
    min_score = int(min_score) if min_score.isdigit() else 3
    
    print(f"\n{CYAN}🎯 Starting custom scan...{RESET}")
    print(f"{CYAN}📊 Coins: {len(coins_list)}{RESET}")
    print(f"{CYAN}⏰ Timeframe: {timeframe}{RESET}")
    print(f"{CYAN}🤖 ML: {'ENABLED' if enable_ml else 'DISABLED'}{RESET}")
    print(f"{CYAN}🎯 Min Score: {min_score}{RESET}")
    
    signals, all_results = scanner.scan_coins(
        coins_to_scan=coins_list,
        timeframe=timeframe,
        enable_ml=enable_ml,
        min_signal_score=min_score
    )
    
    return signals, all_results

def volume_tier_scan(scanner: BybitUniversalScanner, enable_ml: bool = False):
    """Volume tier scan with ML option"""
    ScannerAnimator.animate_title("📊 VOLUME TIER SCAN", 0.02)
    
    all_coins = scanner.get_all_bybit_coins(min_volume=100000)
    
    if not all_coins:
        print(f"{RED}❌ No coins found{RESET}")
        return [], []
    
    print(f"\n{CYAN}📈 Volume Tiers:{RESET}")
    print(f"{GREEN}   1. High Volume (>$10M){RESET}")
    print(f"{YELLOW}   2. Medium Volume ($1M-$10M){RESET}")
    print(f"{RED}   3. Low Volume ($100k-$1M){RESET}")
    
    tier_choice = input(f"\n{YELLOW}Select tier (1-3): {RESET}").strip()
    
    if tier_choice == "1":
        tier_coins = [c for c in all_coins if c['volume_24h'] >= 10000000]
        tier_name = "High Volume"
    elif tier_choice == "2":
        tier_coins = [c for c in all_coins if 1000000 <= c['volume_24h'] < 10000000]
        tier_name = "Medium Volume"
    else:
        tier_coins = [c for c in all_coins if c['volume_24h'] < 1000000]
        tier_name = "Low Volume"
    
    # Take top 15 from the tier
    top_coins = [coin['symbol'] for coin in tier_coins[:15]]
    
    print(f"\n{GREEN}✅ Scanning {len(top_coins)} {tier_name} coins{RESET}")
    print(f"{CYAN}🤖 ML Analysis: {'ENABLED' if enable_ml else 'DISABLED'}{RESET}")
    
    signals, all_results = scanner.scan_coins(
        coins_to_scan=top_coins,
        timeframe='4h',
        enable_ml=enable_ml,
        min_signal_score=3
    )
    
    return signals, all_results

def multi_timeframe_scan(scanner: BybitUniversalScanner, enable_ml: bool = True):
    """Multi-timeframe scan with ML option"""
    ScannerAnimator.animate_title("⏰ MULTI-TIMEFRAME SCAN", 0.02)
    
    print(f"{CYAN}🔍 Fetching top Bybit coins...{RESET}")
    all_coins = scanner.get_all_bybit_coins(min_volume=1000000)
    
    if not all_coins:
        print(f"{RED}❌ No coins found{RESET}")
        return [], []
    
    # Take top 10 coins
    top_coins = [coin['symbol'] for coin in all_coins[:10]]
    
    print(f"\n{CYAN}🔄 Timeframes:{RESET}")
    print(f"1. 1 Hour")
    print(f"2. 4 Hours")
    print(f"3. 1 Day")
    print(f"4. All timeframes")
    
    tf_choice = input(f"\n{YELLOW}Select timeframe(s) (1-4, default: 4): {RESET}").strip()
    
    if tf_choice == "1":
        timeframes = ['1h']
    elif tf_choice == "2":
        timeframes = ['4h']
    elif tf_choice == "3":
        timeframes = ['1d']
    else:
        timeframes = ['1h', '4h', '1d']
    
    print(f"\n{GREEN}✅ Scanning {len(top_coins)} coins across {len(timeframes)} timeframe(s){RESET}")
    print(f"{CYAN}🤖 ML Analysis: {'ENABLED' if enable_ml else 'DISABLED'}{RESET}")
    
    # Scan for each timeframe
    all_signals = []
    all_timeframe_results = []
    
    for tf in timeframes:
        print(f"\n{CYAN}⏰ Timeframe: {tf}{RESET}")
        signals, results = scanner.scan_coins(
            coins_to_scan=top_coins,
            timeframe=tf,
            enable_ml=enable_ml,
            min_signal_score=2
        )
        
        # Add timeframe info to signals
        for signal in signals:
            signal['timeframe'] = tf
        
        all_signals.extend(signals)
        
        # Add timeframe info to results
        for result in results:
            result['timeframe'] = tf
        
        all_timeframe_results.extend(results)
    
    return all_signals, all_timeframe_results

# ==================== MAIN MENU ====================
def main():
    """Main menu with ML options"""
    ScannerAnimator.clear_screen()
    ScannerAnimator.animate_title("🤖 BYBIT UNIVERSAL SCANNER WITH ML", 0.03)
    
    print(f"{GREEN}🤖 ML System Available: {ML_AVAILABLE}{RESET}")
    print(f"{GREEN}📊 Advanced ML Available: {ADVANCED_ML_AVAILABLE}{RESET}")
    
    # Initialize scanner
    scanner = BybitUniversalScanner(account_balance=1000, risk_per_trade=1)
    
    while True:
        print(f"\n{CYAN}{'═'*60}{RESET}")
        print(f"{CYAN}📊 MAIN MENU - SCAN TYPE{RESET}")
        print(f"{CYAN}{'═'*60}{RESET}")
        print(f"{GREEN}1. ⚡ Quick Scan (Top 20, Fast){RESET}")
        print(f"{YELLOW}2. 🔬 Comprehensive Scan (Top 30, Detailed){RESET}")
        print(f"{MAGENTA}3. 🎯 Custom Scan (Your Selection){RESET}")
        print(f"{BLUE}4. 📊 Volume Tier Scan (By Volume){RESET}")
        print(f"{CYAN}5. ⏰ Multi-Timeframe Scan (Multiple TFs){RESET}")
        print(f"{RED}6. 🚪 Exit{RESET}")
        print(f"{CYAN}{'═'*60}{RESET}")
        
        choice = input(f"\n{YELLOW}Select option (1-6): {RESET}").strip()
        
        if choice == "1":
            # ML option for quick scan
            print(f"\n{CYAN}🤖 ML Analysis for Quick Scan:{RESET}")
            print(f"{GREEN}Y - Enable ML (more accurate, slower){RESET}")
            print(f"{YELLOW}N - Disable ML (faster, less accurate){RESET}")
            ml_choice = input(f"\n{YELLOW}Enable ML? (y/n, default: n): {RESET}").strip().lower()
            enable_ml = ml_choice == 'y'
            
            signals, all_results = quick_scan(scanner, enable_ml)
            
        elif choice == "2":
            # ML option for comprehensive scan
            print(f"\n{CYAN}🤖 ML Analysis for Comprehensive Scan:{RESET}")
            print(f"{GREEN}Y - Enable ML (recommended){RESET}")
            print(f"{YELLOW}N - Disable ML (faster){RESET}")
            ml_choice = input(f"\n{YELLOW}Enable ML? (y/n, default: y): {RESET}").strip().lower()
            enable_ml = ml_choice != 'n'
            
            signals, all_results = comprehensive_scan(scanner, enable_ml)
            
        elif choice == "3":
            signals, all_results = custom_scan(scanner)
            
        elif choice == "4":
            # ML option for volume tier scan
            print(f"\n{CYAN}🤖 ML Analysis for Volume Tier Scan:{RESET}")
            print(f"{GREEN}Y - Enable ML (more accurate){RESET}")
            print(f"{YELLOW}N - Disable ML (faster){RESET}")
            ml_choice = input(f"\n{YELLOW}Enable ML? (y/n, default: n): {RESET}").strip().lower()
            enable_ml = ml_choice == 'y'
            
            signals, all_results = volume_tier_scan(scanner, enable_ml)
            
        elif choice == "5":
            # ML option for multi-timeframe scan
            print(f"\n{CYAN}🤖 ML Analysis for Multi-Timeframe Scan:{RESET}")
            print(f"{GREEN}Y - Enable ML (recommended){RESET}")
            print(f"{YELLOW}N - Disable ML (faster){RESET}")
            ml_choice = input(f"\n{YELLOW}Enable ML? (y/n, default: y): {RESET}").strip().lower()
            enable_ml = ml_choice != 'n'
            
            signals, all_results = multi_timeframe_scan(scanner, enable_ml)
            
        elif choice == "6":
            print(f"\n{RED}👋 Exiting... Goodbye!{RESET}")
            break
        else:
            print(f"{RED}❌ Invalid choice. Please try again.{RESET}")
            continue
        
        # Display results
        if all_results:
            # Display ALL coins scanned
            scanner.display_all_coins_scanned(all_results, min_score=3)
            
            # Display detailed signals if any
            if signals:
                print(f"\n{GREEN}🎯 Found {len(signals)} strong trading signals{RESET}")
                scanner.display_detailed_signals(signals)
            else:
                print(f"\n{YELLOW}⚠️  No strong signals found (score ≥ 3){RESET}")
            
            # Save results
            save_choice = input(f"\n{YELLOW}💾 Save results to file? (y/n): {RESET}").strip().lower()
            if save_choice == 'y':
                scanner.save_results_to_file(signals, all_results)
        
        # Ask to continue
        continue_choice = input(f"\n{YELLOW}🔄 Perform another scan? (y/n): {RESET}").strip().lower()
        if continue_choice != 'y':
            print(f"\n{RED}👋 Exiting... Goodbye!{RESET}")
            break
        
        ScannerAnimator.clear_screen()

# ==================== RUN THE SCANNER ====================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{RED}👋 Scanner interrupted by user. Exiting...{RESET}")
    except Exception as e:
        print(f"\n\n{RED}❌ Critical error: {e}{RESET}")
        import traceback
        traceback.print_exc()