# test_integration.py - Script untuk test fitur baru

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path untuk import modules
sys.path.append(os.getcwd())

def test_fibonacci():
    """Test fibonacci implementation"""
    print("🧪 Testing Fibonacci Implementation...")
    
    try:
        # Import dari app.py 
        from app import AdvancedIndicators
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)  # Random walk starting at 50k
        
        # Add some volatility
        highs = prices * (1 + np.random.rand(100) * 0.02)
        lows = prices * (1 - np.random.rand(100) * 0.02)
        volumes = np.random.randint(1000000, 10000000, 100)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': highs,
            'low': lows, 
            'close': prices,
            'volume': volumes
        })
        
        # Test fibonacci calculation
        fib_result = AdvancedIndicators.calculate_fibonacci_levels(df)
        
        if fib_result:
            print("✅ Fibonacci calculation successful!")
            print(f"   - Swing High: ${fib_result.get('swing_high', 0):.2f}")
            print(f"   - Swing Low: ${fib_result.get('swing_low', 0):.2f}")
            print(f"   - Trend Direction: {fib_result.get('trend_direction', 'unknown')}")
            print(f"   - At Fib Level: {fib_result.get('at_fib_level', False)}")
            print(f"   - Confluence Count: {fib_result.get('confluence_count', 0)}")
            
            # Test confluence
            retracements = fib_result.get('retracements', {})
            if retracements:
                print(f"   - Retracement Levels: {len(retracements)}")
                for key, level_data in list(retracements.items())[:3]:
                    print(f"     • {level_data['label']}: ${level_data['price']:.2f}")
            
        else:
            print("❌ Fibonacci calculation returned empty result")
            
    except Exception as e:
        print(f"❌ Fibonacci test failed: {e}")
        import traceback
        traceback.print_exc()

def test_modules():
    """Test new module imports"""
    print("\n🧪 Testing Module Imports...")
    
    # Test WebSocket Manager
    try:
        from websocket_manager import BinanceWebSocketManager
        ws_manager = BinanceWebSocketManager()
        print("✅ WebSocket Manager imported successfully")
    except Exception as e:
        print(f"❌ WebSocket Manager error: {e}")
    
    # Test Database
    try:
        from database import TradingDatabase
        db = TradingDatabase()
        print("✅ Database imported successfully")
        print(f"   - Database file: {db.db_path}")
    except Exception as e:
        print(f"❌ Database error: {e}")
    
    # Test ML Enhancer
    try:
        from ml_enhancer import MLSignalEnhancer
        ml = MLSignalEnhancer()
        print("✅ ML Enhancer imported successfully")
    except Exception as e:
        print(f"❌ ML Enhancer error: {e}")

def test_trading_engine():
    """Test enhanced trading engine"""
    print("\n🧪 Testing Enhanced Trading Engine...")
    
    try:
        from app import TradingEngine, TradingConfig
        
        config = TradingConfig()
        engine = TradingEngine(config)
        
        print("✅ Enhanced Trading Engine initialized successfully")
        print(f"   - WebSocket Manager: {'✅' if engine.websocket_manager else '❌'}")
        print(f"   - Database: {'✅' if engine.database else '❌'}")
        print(f"   - ML Enhancer: {'✅' if engine.ml_enhancer else '❌'}")
        print(f"   - Portfolio Value: ${engine.portfolio_value:,}")
        
    except Exception as e:
        print(f"❌ Trading Engine test failed: {e}")
        import traceback
        traceback.print_exc()

def test_helper_functions():
    """Test helper functions"""
    print("\n🧪 Testing Helper Functions...")
    
    try:
        from app import get_fib_status, get_enhanced_signal_summary
        
        # Test fibonacci status
        test_indicators = {
            'fibonacci': {
                'at_fib_level': True,
                'confluence_count': 2,
                'distance_pct': 0.3,
                'nearest_level_info': {'label': '61.8%'}
            }
        }
        
        fib_status = get_fib_status(test_indicators)
        print(f"✅ Fibonacci status: {fib_status}")
        
        # Test signal summary
        test_result = {
            'signal_components': {
                'technical': 15,
                'momentum': 12,
                'volume': 8,
                'sentiment': 6,
                'structure': 5,
                'ml_enhanced': True
            }
        }
        
        signal_summary = get_enhanced_signal_summary(test_result)
        print(f"✅ Signal summary: {signal_summary}")
        
    except Exception as e:
        print(f"❌ Helper functions test failed: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting Integration Tests...")
    print("="*50)
    
    test_modules()
    test_fibonacci()
    test_trading_engine()
    test_helper_functions()
    
    print("\n" + "="*50)
    print("✨ Integration tests completed!")
    print("\nNext steps:")
    print("1. Run 'python test_integration.py' to verify everything works")
    print("2. If tests pass, run 'python app.py' to start the bot")
    print("3. Test fibonacci signals with /start command in Telegram")

if __name__ == "__main__":
    main()
