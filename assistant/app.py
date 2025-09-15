
#file app.py
import ccxt
import pandas as pd
import requests
import time
import asyncio
from datetime import datetime, timedelta
import talib
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import certifi
import traceback
import argparse
import sys, os
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pytz import timezone
import concurrent.futures
import threading
from collections import defaultdict, deque
from telegram.ext import MessageHandler, filters
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
from telegram.error import BadRequest
from .economic_data_provider import EconomicDataProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .websocket_manager import BinanceWebSocketManager
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logging.warning("WebSocket manager not available")

try:
    from .database import TradingDatabase
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logging.warning("Database manager not available")

try:
    from .ml_enhancer import MLSignalEnhancer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML enhancer not available")

try:
    from .economic_data_provider import EconomicDataProvider
except ImportError:
    logging.error("PENTING: Gagal mengimpor EconomicDataProvider.")

# Additional imports for new features
import sqlite3
from datetime import timedelta
from typing import Union, Any


# ===============================
# PROFESSIONAL CONFIGURATION
# ===============================

@dataclass
class TradingConfig:
    """Professional trading configuration"""
    # Risk Management
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk per trade
    max_daily_risk: float = 0.06     # 6% max daily risk
    max_correlation_exposure: float = 0.10  # 10% max correlated positions
    
    # Position Sizing
    min_position_size: float = 100    # Minimum $100 position
    max_position_size: float = 50000  # Maximum $50k position
    max_leverage: Dict[str, float] = None
    
    # Signal Thresholds
    min_signal_strength: int = 6      # Minimum signal strength (lowered from 8)
    min_mtf_confidence: float = 40    # Minimum multi-timeframe confidence (lowered from 65)
    min_volume_threshold: float = 1_000_000  # Minimum $1M volume (lowered from 5M)
    
    # Advanced Features
    use_portfolio_optimization: bool = True
    use_sector_rotation: bool = True
    use_market_regime_filter: bool = True
    use_risk_parity: bool = True
    
    def __post_init__(self):
        if self.max_leverage is None:
            self.max_leverage = {
                'SCALPING': 20.0,
                'DAY_TRADING': 10.0,
                'SWING': 5.0,
                'POSITION': 3.0
            }

class MarketRegime(Enum):
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending" 
    BULL_RANGING = "bull_ranging"
    BEAR_RANGING = "bear_ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

class AssetClass(Enum):
    LARGE_CAP = "large_cap"          # BTC, ETH
    MID_CAP = "mid_cap"              # BNB, ADA, SOL
    DEFI = "defi"                    # UNI, AAVE, COMP
    LAYER1 = "layer1"                # SOL, AVAX, DOT
    MEME = "meme"                    # DOGE, SHIB, PEPE
    AI = "ai"                        # FET, AGIX, RNDR
    GAMING = "gaming"                # AXS, SAND, GALA

# ===============================
# ADVANCED TECHNICAL ANALYSIS
# ===============================

class AdvancedIndicators:
    """Professional-grade technical indicators with null safety"""

    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame, bins: int = 24, value_area_pct: float = 0.70) -> Dict:
        """
        Calculate a professional-grade Volume Profile with POC, VAH, and VAL.
        """
        try:
            if df is None or df.empty or not all(c in df.columns for c in ['high', 'low', 'close', 'volume']):
                return {}

            # 1. Tentukan rentang harga dan ukuran setiap 'bin' (level harga)
            min_price = df['low'].min()
            max_price = df['high'].max()
            price_range = max_price - min_price
            if price_range == 0:
                return {}
            
            bin_size = price_range / bins
            
            # 2. Buat 'keranjang' untuk setiap level harga
            price_bins = np.arange(min_price, max_price + bin_size, bin_size)
            volume_per_bin = np.zeros(len(price_bins) - 1)

            # 3. Distribusikan volume dari setiap candle ke level harga yang sesuai
            #    Kita gunakan 'typical price' (H+L+C)/3 sebagai proxy harga transaksi
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # np.digitize menempatkan setiap harga ke dalam bin yang benar secara efisien
            bin_indices = np.digitize(typical_price, price_bins) - 1

            # Agregasi volume ke setiap bin
            for i, volume in enumerate(df['volume']):
                # Pastikan index berada dalam rentang yang valid
                if 0 <= bin_indices[i] < len(volume_per_bin):
                    volume_per_bin[bin_indices[i]] += volume
            
            # 4. Hitung Point of Control (POC)
            #    POC adalah level harga dengan volume tertinggi
            poc_index = np.argmax(volume_per_bin)
            poc_price = price_bins[poc_index] + (bin_size / 2)
            
            # 5. Hitung Value Area (VA), VAH, dan VAL
            total_volume = np.sum(volume_per_bin)
            va_volume_target = total_volume * value_area_pct
            
            # Mulai dari POC dan kembangkan ke atas dan bawah sampai 70% volume tercapai
            current_va_volume = volume_per_bin[poc_index]
            va_lower_index = va_upper_index = poc_index
            
            while current_va_volume < va_volume_target:
                # Cek bin di atas dan di bawah, pilih yang volumenya lebih besar untuk ditambahkan
                next_lower_index = va_lower_index - 1
                next_upper_index = va_upper_index + 1
                
                volume_lower = volume_per_bin[next_lower_index] if next_lower_index >= 0 else -1
                volume_upper = volume_per_bin[next_upper_index] if next_upper_index < len(volume_per_bin) else -1

                if volume_lower == -1 and volume_upper == -1:
                    break # Sudah mencapai batas atas dan bawah
                    
                if volume_upper > volume_lower:
                    current_va_volume += volume_upper
                    va_upper_index = next_upper_index
                else:
                    current_va_volume += volume_lower
                    va_lower_index = next_lower_index

            val_price = price_bins[va_lower_index]
            vah_price = price_bins[va_upper_index + 1]

            # 6. Susun hasil akhir
            profile_data = {
                price_bins[i] + (bin_size / 2): vol for i, vol in enumerate(volume_per_bin)
            }

            return {
                'poc_price': poc_price,
                'vah_price': vah_price,
                'val_price': val_price,
                'value_area_volume_pct': (current_va_volume / total_volume) if total_volume > 0 else 0,
                'profile': profile_data # Data mentah volume per harga
            }

        except Exception as e:
            # Ganti dengan 'logger.error' jika Anda sudah setup logger
            print(f"Professional volume profile calculation error: {e}")
            return {}
    
    @staticmethod
    def safe_array_operation(array1, array2, operation='add', default_value=0):
        """Safely perform array operations with null checks"""
        try:
            if array1 is None or array2 is None:
                return default_value
            
            # Convert to numpy arrays
            arr1 = np.array(array1) if not isinstance(array1, np.ndarray) else array1
            arr2 = np.array(array2) if not isinstance(array2, np.ndarray) else array2
            
            # Handle NaN values
            arr1 = np.nan_to_num(arr1, nan=default_value)
            arr2 = np.nan_to_num(arr2, nan=default_value)
            
            if operation == 'add':
                return arr1 + arr2
            elif operation == 'subtract':
                return arr1 - arr2
            elif operation == 'multiply':
                return arr1 * arr2
            elif operation == 'divide':
                return np.divide(arr1, arr2, out=np.zeros_like(arr1), where=arr2!=0)
            
            return default_value
            
        except Exception as e:
            logger.error(f"Safe array operation error: {e}")
            return default_value

    @staticmethod
    def calculate_bos_indicator(df: pd.DataFrame) -> np.ndarray:
        """Calculate Break of Structure (BOS) indicator with safety checks"""
        try:
            if df is None or len(df) < 20:
                return np.zeros(10)
            
            highs = df['high'].fillna(method='ffill').to_numpy()
            lows = df['low'].fillna(method='ffill').to_numpy()
            closes = df['close'].fillna(method='ffill').to_numpy()
            volumes = df['volume'].fillna(0).to_numpy()
            
            bos_signals = np.zeros(len(df))
            window = min(10, len(df) // 2)
            
            for i in range(window, len(df) - window):
                try:
                    # Look for higher highs (bullish BOS)
                    recent_highs = highs[max(0, i-window):i]
                    if len(recent_highs) > 0 and highs[i] > np.max(recent_highs) and closes[i] > closes[i-1]:
                        # Confirm with volume
                        avg_volume = np.mean(volumes[max(0, i-5):i]) if i >= 5 else volumes[i]
                        if avg_volume > 0 and volumes[i] > avg_volume * 1.2:
                            bos_signals[i] = 1  # Bullish BOS
                    
                    # Look for lower lows (bearish BOS)
                    recent_lows = lows[max(0, i-window):i]
                    if len(recent_lows) > 0 and lows[i] < np.min(recent_lows) and closes[i] < closes[i-1]:
                        # Confirm with volume
                        avg_volume = np.mean(volumes[max(0, i-5):i]) if i >= 5 else volumes[i]
                        if avg_volume > 0 and volumes[i] > avg_volume * 1.2:
                            bos_signals[i] = -1  # Bearish BOS
                except Exception:
                    continue
            
            return bos_signals
            
        except Exception as e:
            logger.error(f"Error calculating BOS: {e}")
            return np.zeros(len(df) if df is not None else 10)
                
    @staticmethod
    def calculate_market_structure(df: pd.DataFrame, lookback: int = 20) -> Dict:
        """Calculate market structure analysis with safety checks"""
        try:
            if df is None or len(df) < lookback:
                return {
                    'structure_score': 0,
                    'higher_highs': 0,
                    'lower_lows': 0,
                    'trend_strength': 0,
                    'structure_type': 'neutral'
                }
            
            highs = df['high'].fillna(method='ffill').to_numpy()
            lows = df['low'].fillna(method='ffill').to_numpy()
            
            # Higher Highs, Lower Lows detection
            hh_count = 0  # Higher Highs
            ll_count = 0  # Lower Lows
            hl_count = 0  # Higher Lows
            lh_count = 0  # Lower Highs
            
            for i in range(lookback, len(df)):
                try:
                    recent_highs = highs[max(0, i-lookback):i]
                    recent_lows = lows[max(0, i-lookback):i]
                    
                    if len(recent_highs) > 0 and highs[i] > np.max(recent_highs):
                        hh_count += 1
                    if len(recent_lows) > 0 and lows[i] < np.min(recent_lows):
                        ll_count += 1
                    if len(recent_lows) > 0 and lows[i] > np.min(recent_lows):
                        hl_count += 1
                    if len(recent_highs) > 0 and highs[i] < np.max(recent_highs):
                        lh_count += 1
                except Exception:
                    continue
            
            # Structure score: +1 for bullish structure, -1 for bearish
            total_signals = hh_count + hl_count + ll_count + lh_count
            if total_signals > 0:
                structure_score = (hh_count + hl_count - ll_count - lh_count) / total_signals
            else:
                structure_score = 0
            
            return {
                'structure_score': structure_score,
                'higher_highs': hh_count,
                'lower_lows': ll_count,
                'trend_strength': abs(structure_score),
                'structure_type': 'bullish' if structure_score > 0.1 else 'bearish' if structure_score < -0.1 else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"Error calculating market structure: {e}")
            return {
                'structure_score': 0,
                'higher_highs': 0,
                'lower_lows': 0,
                'trend_strength': 0,
                'structure_type': 'neutral'
            }
    
    @staticmethod
    def calculate_smart_money_index(df: pd.DataFrame) -> np.ndarray:
        """Calculate Smart Money Index (SMI) with safety checks"""
        try:
            if df is None or len(df) < 14:
                return np.zeros(len(df) if df is not None else 10)
            
            # Smart Money Index = Close - (High + Low + Close) / 3
            high_vals = df['high'].fillna(method='ffill').to_numpy()
            low_vals = df['low'].fillna(method='ffill').to_numpy()
            close_vals = df['close'].fillna(method='ffill').to_numpy()
            
            # Calculate typical price with safety checks
            typical_price = (high_vals + low_vals + close_vals) / 3
            smi = close_vals - typical_price
            
            # Replace NaN with 0
            smi = np.nan_to_num(smi, nan=0.0)
            
            return talib.EMA(smi, timeperiod=14)
            
        except Exception as e:
            logger.error(f"Error calculating SMI: {e}")
            return np.zeros(len(df) if df is not None else 10)
            
    @staticmethod
    def calculate_fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> Dict:
        """Calculate Fibonacci retracements and extensions with enhanced logic"""
        try:
            if df is None or len(df) < lookback:
                return {}
        
            # Get recent data
            recent_data = df.tail(lookback)
        
            # Find swing high and low
            swing_high_idx = recent_data['high'].idxmax()
            swing_low_idx = recent_data['low'].idxmin()
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
        
            # Determine trend direction
            trend_direction = 'bullish' if swing_low_idx < swing_high_idx else 'bearish'
        
            # Fibonacci ratios
            fib_ratios = {
                0.0: '0%',
                0.236: '23.6%', 
                0.382: '38.2%',
                0.5: '50%',
                0.618: '61.8%',
                0.786: '78.6%',
                1.0: '100%'
            }
        
            extension_ratios = {
                1.236: '123.6%',
                1.382: '138.2%', 
                1.618: '161.8%',
                2.0: '200%',
                2.618: '261.8%'
            }
        
            # Calculate levels
            diff = swing_high - swing_low
            retracements = {}
            extensions = {}
        
            if trend_direction == 'bullish':
                # For bullish trend, retracements from high
                for ratio, label in fib_ratios.items():
                    retracements[f'fib_{ratio}'] = {
                        'price': swing_high - (diff * ratio),
                        'label': label,
                        'type': 'retracement'
                    }
            
                # Extensions above high
                for ratio, label in extension_ratios.items():
                    extensions[f'fib_ext_{ratio}'] = {
                        'price': swing_low + (diff * ratio),
                        'label': label,
                        'type': 'extension'
                    }
            else:
                # For bearish trend, retracements from low  
                for ratio, label in fib_ratios.items():
                    retracements[f'fib_{ratio}'] = {
                        'price': swing_low + (diff * ratio),
                        'label': label,
                        'type': 'retracement'
                    }
            
                # Extensions below low
                for ratio, label in extension_ratios.items():
                    extensions[f'fib_ext_{ratio}'] = {
                        'price': swing_high - (diff * ratio),
                        'label': label,
                        'type': 'extension'
                    }
        
            current_price = float(df['close'].iloc[-1])
        
            # Find nearest fib level
            all_levels = []
            for level_data in list(retracements.values()) + list(extensions.values()):
                all_levels.append(level_data['price'])
        
            if all_levels:
                nearest_level = min(all_levels, key=lambda x: abs(x - current_price))
                distance_to_nearest = abs(current_price - nearest_level) / current_price
            
                # Find which fib level it is
                nearest_level_info = None
                for level_data in list(retracements.values()) + list(extensions.values()):
                    if abs(level_data['price'] - nearest_level) < 0.001:
                        nearest_level_info = level_data
                        break
            else:
                nearest_level = current_price
                distance_to_nearest = 1.0
                nearest_level_info = None
        
            # Determine if at significant fib level
            at_fib_level = distance_to_nearest < 0.005  # Within 0.5%
            near_fib_level = distance_to_nearest < 0.01  # Within 1%
        
            # Calculate fib confluence (multiple levels nearby)
            confluence_count = 0
            for level_data in list(retracements.values()) + list(extensions.values()):
                level_distance = abs(current_price - level_data['price']) / current_price
                if level_distance < 0.01:  # Within 1%
                    confluence_count += 1
        
            return {
                'retracements': retracements,
                'extensions': extensions,
                'swing_high': swing_high,
                'swing_low': swing_low,
                'swing_high_idx': swing_high_idx,
                'swing_low_idx': swing_low_idx,
                'trend_direction': trend_direction,
                'nearest_level': nearest_level,
                'nearest_level_info': nearest_level_info,
                'distance_pct': distance_to_nearest * 100,
                'at_fib_level': at_fib_level,
                'near_fib_level': near_fib_level,
                'confluence_count': confluence_count,
                'fib_strength': min(confluence_count * 2, 10)  # 0-10 scale
            }
        
        except Exception as e:
            logger.error(f"Fibonacci calculation error: {e}")
            return {}

    @staticmethod  
    def calculate_fibonacci_confluence(price_levels: List[float], current_price: float, tolerance: float = 0.01) -> Dict:
        """Calculate fibonacci confluence zones"""
        try:
           confluence_zones = []
        
           # Group nearby levels
           sorted_levels = sorted(price_levels)
        
           i = 0
           while i < len(sorted_levels):
               zone_levels = [sorted_levels[i]]
               zone_center = sorted_levels[i]
            
               # Find nearby levels within tolerance
               j = i + 1
               while j < len(sorted_levels):
                   if abs(sorted_levels[j] - zone_center) / zone_center <= tolerance:
                       zone_levels.append(sorted_levels[j])
                       j += 1
                   else:
                       break
            
               if len(zone_levels) > 1:  # Confluence exists
                   zone_avg = sum(zone_levels) / len(zone_levels)
                   confluence_zones.append({
                      'center': zone_avg,
                      'levels': zone_levels,
                      'strength': len(zone_levels),
                      'distance_from_price': abs(current_price - zone_avg) / current_price
                   })
            
               i = j if j > i else i + 1
        
           # Sort by strength
           confluence_zones.sort(key=lambda x: x['strength'], reverse=True)
        
           return {
               'zones': confluence_zones,
               'strongest_zone': confluence_zones[0] if confluence_zones else None,
               'total_zones': len(confluence_zones)
           }
        
        except Exception as e:
           logger.error(f"Confluence calculation error: {e}")
           return {'zones': [], 'strongest_zone': None, 'total_zones': 0}

# ===============================
# PROFESSIONAL EXECUTION ENGINE
# ===============================

class TradingEngine:
    """Professional trading execution engine with enhanced error handling"""
    
    def generate_sentiment_report(self) -> str:
        """Menghasilkan laporan intelijen pasar yang komprehensif dan anti-rapuh."""
        try:
            # --- PERIKSA DULU APAKAH "MATA-MATA" EKONOMI KITA AKTIF ---
            if not self.economic_provider:
                return "❌ Gagal menghasilkan laporan: Modul data ekonomi tidak aktif. Pastikan API Key di file .env sudah benar."

            # --- Jika mata-mata aktif, lanjutkan seperti biasa ---
            sentiment_score = 0
            
            # Ambil Data Kripto dari CoinGecko
            btc_d_str, total3_str = "Data tidak tersedia.", "Data tidak tersedia."
            crypto_data = self.economic_provider.get_global_crypto_data()
            if crypto_data:
                btc_d_str = f"{crypto_data['btc_dominance']:.2f}%"
                total3_str = f"${crypto_data['total3_market_cap']:,.0f}"
            
            # Ambil Data Ekonomi dari FRED
            cpi_data_str, fed_rate_str = "Data tidak tersedia.", "Data tidak tersedia."
            cpi_data = self.economic_provider.get_latest_cpi()
            if cpi_data: cpi_data_str = f"{cpi_data['value']} (MoM: {cpi_data['change_mom']})"
            fed_rate_data = self.economic_provider.get_interest_rate()
            if fed_rate_data: fed_rate_str = f"{fed_rate_data['value']} (Update: {fed_rate_data['date']})"
            
            # Ambil Data Kalender & Berita
            calendar_data = self.economic_provider.get_economic_calendar()
            events_today_str = "\n".join([f"• {e}" for e in calendar_data['today']])
            next_cpi = calendar_data['next_cpi']
            next_fomc = calendar_data['next_fomc']
            
            # 4. Data Berita dari NewsAPI
            news_list = self.economic_provider.get_market_news()
            news_str = "\n".join(news_list)

            # Mesin Kesimpulan Sederhana
            final_sentiment = "Netral 😐 - Pasar menunggu katalis berikutnya."
            if sentiment_score > 0:
                final_sentiment = "Condong ke Bullish 🐂 - Aliran dana positif ke Altcoin."
            elif sentiment_score < 0:
                final_sentiment = "Condong ke Bearish 🐻 - Aliran dana cenderung ke BTC/Safe Haven."

            # Susun Laporan Intelijen
            report_msg = (
                f"📊 **LAPORAN INTELIJEN PASAR** 📊\n"
                f"{datetime.now(timezone('Asia/Jakarta')).strftime('%A, %d %B %Y %H:%M WIB')}\n"
                f"{'='*30}\n\n"
                f"**Kesimpulan Sentimen:** **{final_sentiment}**\n\n"
                f"**1. Data Kripto (via CoinGecko):**\n"
                f"• BTC Dominance: **{btc_d_str}**\n"
                f"• Altcoin Market Cap (TOTAL3): **{total3_str}**\n\n"
                f"**2. Data Ekonomi Makro (AS):**\n"
                f"• Inflasi (CPI Terakhir): **{cpi_data_str}**\n"
                f"• Suku Bunga (Fed Rate): **{fed_rate_str}**\n\n"
                f"**3. Jadwal Ekonomi Penting Hari Ini (USD):**\n"
                f"**Acara Penting Hari Ini:**\n{events_today_str}\n\n"
                f"**Agenda Penting Mendatang:**\n"
                f"• Rilis CPI Berikutnya: **{next_cpi}**\n"
                f"• Rapat FOMC Berikutnya: **{next_fomc}**\n\n"
                f"**4. Berita Utama Pasar Global:**\n"
                f"{news_str}"
            )
            return report_msg

        except Exception as e:
            logger.error(f"Gagal membuat laporan intelijen: {e}", exc_info=True)
            return "❌ Gagal menghasilkan laporan intelijen."

    def update_config(self, key: str, value: str) -> str:
        """Mengubah parameter konfigurasi bot secara dinamis."""
        try:
            # Cek apakah parameter ada di dalam TradingConfig
            if hasattr(self.config, key):
                # Dapatkan tipe data asli dari parameter (misal: float, int, str)
                original_type = type(getattr(self.config, key))
                # Konversi nilai input ke tipe data yang benar
                new_value = original_type(value)
                
                # Update konfigurasi
                setattr(self.config, key, new_value)
                
                logger.info(f"Konfigurasi diubah: {key} -> {new_value}")
                return f"✅ `{key}` telah berhasil diubah menjadi `{new_value}`."
            else:
                return f"❌ Parameter `{key}` tidak ditemukan."
        except Exception as e:
            logger.error(f"Gagal mengubah konfigurasi {key}: {e}")
            return f"❌ Gagal mengubah `{key}`. Pastikan nilainya benar. Error: {e}"
                
    def generate_entry_zone(self, result: Dict) -> dict:
        """
        Menganalisis data untuk merekomendasikan zona entry.
        Mengembalikan sebuah dictionary berisi teks dan data angka mentah.
        """
        try:
            signal_type = result.get('signal_type')
            current_price = result.get('current_price')
            
            tf_analysis = result.get('tf_analysis', {})
            primary_tf = self.get_primary_timeframe_safe(tf_analysis)
            indicators = tf_analysis.get(primary_tf, {}).get('indicators', {})
            
            volume_profile = indicators.get('volume_profile', {})
            fib_data = indicators.get('fibonacci', {})
            
            if signal_type == 'BULLISH':
                support_levels = []
                if volume_profile.get('val_price'): support_levels.append(volume_profile['val_price'])
                if volume_profile.get('poc_price'): support_levels.append(volume_profile['poc_price'])
                if fib_data and fib_data.get('retracements'):
                    for level in fib_data['retracements'].values():
                        support_levels.append(level['price'])
                
                relevant_supports = [lvl for lvl in support_levels if lvl < current_price and lvl > current_price * 0.95]
                
                if not relevant_supports:
                    # Jika tidak ada support kuat, buat zona kecil di sekitar harga saat ini
                    zone_bawah = current_price * 0.998
                    zone_atas = current_price * 1.002
                    return {
                        "text": f"Dekat harga saat ini (${current_price:,.4f})",
                        "raw": [zone_bawah, zone_atas]
                    }
                
                strongest_support = max(relevant_supports)
                zone_bawah = strongest_support
                zone_atas = strongest_support * 1.003
                
                return {
                    "text": f"${zone_bawah:,.4f} - ${zone_atas:,.4f}",
                    "raw": [zone_bawah, zone_atas]
                }

            elif signal_type == 'BEARISH':
                resistance_levels = []
                if volume_profile.get('vah_price'): resistance_levels.append(volume_profile['vah_price'])
                if volume_profile.get('poc_price'): resistance_levels.append(volume_profile['poc_price'])
                if fib_data and fib_data.get('retracements'):
                    for level in fib_data['retracements'].values():
                        resistance_levels.append(level['price'])

                relevant_resistances = [lvl for lvl in resistance_levels if lvl > current_price and lvl < current_price * 1.05]
                
                if not relevant_resistances:
                    zone_bawah = current_price * 0.998
                    zone_atas = current_price * 1.002
                    return {
                        "text": f"Dekat harga saat ini (${current_price:,.4f})",
                        "raw": [zone_bawah, zone_atas]
                    }
                
                strongest_resistance = min(relevant_resistances)
                zone_atas = strongest_resistance
                zone_bawah = strongest_resistance * 0.997
                
                return {
                    "text": f"${zone_bawah:,.4f} - ${zone_atas:,.4f}",
                    "raw": [zone_bawah, zone_atas]
                }
            
            return {"text": "(N/A)", "raw": [0, 0]}

        except Exception as e:
            logger.error(f"Error saat generate entry zone: {e}")
            return {"text": "(Gagal menghitung zona)", "raw": [0, 0]}

    def __init__(self, config: TradingConfig):
        """Konstruktor untuk TradingEngine (Versi Final dengan 3 Kunci API)."""
        self.config = config
        self.watchlist = defaultdict(set)
        self.trade_history = deque(maxlen=1000)
        self.portfolio_value = 100000
        self.active_positions = {}
        self.sector_exposure = defaultdict(float)
        self.real_time_data = defaultdict(dict)
        self.signal_cache = {}
        self.auto_scanner_active = False
        self.active_profile = 'NORMAL'
        self.database = None
        self.ml_enhancer = None
        self.economic_provider = None

        fred_key = os.getenv("FRED_API_KEY")
        news_key = os.getenv("NEWS_API_KEY")

        try:
            from .economic_data_provider import EconomicDataProvider 
            self.economic_provider = EconomicDataProvider(
                fred_key=fred_key, 
                news_key=news_key
            )
        except Exception as e:
            self.economic_provider = None
            logger.warning(f"PENTING: Gagal menginisialisasi EconomicDataProvider: {e}")
            logger.warning("Fitur Sentimen/S&R tidak akan aktif. Pastikan SEMUA API Key di .env sudah benar.")

        if DATABASE_AVAILABLE:
            try:
                self.database = TradingDatabase()
                logger.info("Database initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
        
        if ML_AVAILABLE:
            try:
                self.ml_enhancer = MLSignalEnhancer()
                logger.info("ML enhancer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ML enhancer: {e}")
                
        logger.info("TradingEngine initialized with enhanced features")
        
    def start_scanner(self):
        """Mengaktifkan auto scanner."""
        if not self.auto_scanner_active:
            self.auto_scanner_active = True
            logger.info("Auto Scanner DIAKTIFKAN.")
            return "✅ Auto scanner telah DIAKTIFKAN. Bot akan mulai berburu sinyal untuk Anda."
        return "ℹ️ Auto scanner sudah aktif."

    def stop_scanner(self):
        """Menonaktifkan auto scanner."""
        if self.auto_scanner_active:
            self.auto_scanner_active = False
            logger.info("Auto Scanner DINONAKTIFKAN.")
            return "⏹️ Auto scanner telah DINONAKTIFKAN."
        return "ℹ️ Auto scanner sudah nonaktif."

    def safe_indicator_value(self, indicator_array, index=-1, default_value=0):
        """Safely get indicator value with fallback"""
        try:
            if indicator_array is None or len(indicator_array) == 0:
                return default_value
            
            # Handle negative indexing
            if index < 0:
                index = len(indicator_array) + index
                
            if 0 <= index < len(indicator_array):
                value = indicator_array[index]
                # Check for NaN or None
                if value is None or np.isnan(value):
                    return default_value
                return float(value)
            else:
                return default_value
                
        except Exception as e:
            logger.error(f"Error getting indicator value: {e}")
            return default_value

    def analyze_trend_professional(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Professional trend analysis with enhanced safety"""
        try:
            if df is None or len(df) < 10:
                return {'trend': 'NEUTRAL', 'strength': 0, 'market_structure': {}}
            
            # Safely get EMA values
            current_ema20 = self.safe_indicator_value(indicators.get('ema_20'), default_value=0)
            current_ema50 = self.safe_indicator_value(indicators.get('ema_50'), default_value=0)
            current_ema200 = self.safe_indicator_value(indicators.get('ema_200'), default_value=current_ema50)
            current_price = float(df['close'].iloc[-1]) if not df['close'].empty else 0
            
            if current_price == 0:
                return {'trend': 'NEUTRAL', 'strength': 0, 'market_structure': {}}
            
            # Trend direction scoring
            trend_score = 0
            
            # Price vs EMAs (safe comparisons)
            if current_ema20 > 0 and current_price > current_ema20:
                trend_score += 1
            if current_ema50 > 0 and current_price > current_ema50:
                trend_score += 1
            if current_ema200 > 0 and current_price > current_ema200:
                trend_score += 1
            
            # EMA alignment
            if current_ema20 > current_ema50 and current_ema50 > 0:
                trend_score += 1
            if current_ema50 > current_ema200 and current_ema200 > 0:
                trend_score += 1
            
            # EMA slope analysis with safety
            ema20_array = indicators.get('ema_20')
            ema50_array = indicators.get('ema_50')
            
            ema20_slope = 0
            ema50_slope = 0
            
            if ema20_array is not None and len(ema20_array) >= 5:
                current_ema20_val = self.safe_indicator_value(ema20_array, -1)
                prev_ema20_val = self.safe_indicator_value(ema20_array, -5)
                if prev_ema20_val > 0:
                    ema20_slope = (current_ema20_val - prev_ema20_val) / prev_ema20_val
            
            if ema50_array is not None and len(ema50_array) >= 5:
                current_ema50_val = self.safe_indicator_value(ema50_array, -1)
                prev_ema50_val = self.safe_indicator_value(ema50_array, -5)
                if prev_ema50_val > 0:
                    ema50_slope = (current_ema50_val - prev_ema50_val) / prev_ema50_val
            
            if ema20_slope > 0.001:  # Rising
                trend_score += 1
            elif ema20_slope < -0.001:  # Falling
                trend_score -= 1
                
            if ema50_slope > 0.001:  # Rising
                trend_score += 1
            elif ema50_slope < -0.001:  # Falling
                trend_score -= 1
            
            # Market structure analysis
            structure = AdvancedIndicators.calculate_market_structure(df)
            structure_score = structure.get('structure_score', 0)
            
            if structure_score > 0.1:
                trend_score += 1
            elif structure_score < -0.1:
                trend_score -= 1
            
            # Determine trend classification
            if trend_score >= 5:
                trend = 'STRONG_BULLISH'
                strength = min(trend_score / 8.0, 1.0)
            elif trend_score >= 3:
                trend = 'BULLISH'
                strength = trend_score / 8.0
            elif trend_score <= -5:
                trend = 'STRONG_BEARISH' 
                strength = abs(trend_score) / 8.0
            elif trend_score <= -3:
                trend = 'BEARISH'
                strength = abs(trend_score) / 8.0
            else:
                trend = 'NEUTRAL'
                strength = 0.3  # Lowered default strength
            
            return {
                'trend': trend,
                'strength': strength,
                'trend_score': trend_score,
                'market_structure': structure,
                'ema_alignment': {
                    'price_above_ema20': current_price > current_ema20 if current_ema20 > 0 else False,
                    'price_above_ema50': current_price > current_ema50 if current_ema50 > 0 else False,
                    'ema20_above_ema50': current_ema20 > current_ema50 if current_ema50 > 0 else False,
                    'ema20_slope': ema20_slope,
                    'ema50_slope': ema50_slope
                }
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {'trend': 'NEUTRAL', 'strength': 0, 'market_structure': {}}
            
    def calculate_signal_strength_advanced(self, indicators: Dict, current_price: float) -> int:
        """Advanced signal strength calculation with safety checks"""
        try:
            signal_strength = 0
            
            if current_price <= 0:
                return 0
            
            # RSI analysis with safety
            rsi_current = self.safe_indicator_value(indicators.get('rsi'), default_value=50)
            
            if rsi_current < 20:
                signal_strength += 3
            elif rsi_current < 30:
                signal_strength += 2
            elif rsi_current > 80:
                signal_strength += 3
            elif rsi_current > 70:
                signal_strength += 2
            elif 35 <= rsi_current <= 65:
                signal_strength += 1  # Neutral RSI gets some points
            
            # MACD analysis with safety
            macd_hist = indicators.get('macd_hist')
            if macd_hist is not None and len(macd_hist) >= 2:
                current_hist = self.safe_indicator_value(macd_hist, -1)
                prev_hist = self.safe_indicator_value(macd_hist, -2)
                
                # MACD crossover
                if current_hist > 0 and prev_hist <= 0:
                    signal_strength += 3
                elif current_hist < 0 and prev_hist >= 0:
                    signal_strength += 3
                elif abs(current_hist) > abs(prev_hist):
                    signal_strength += 1
            
            # Volume confirmation with safety
            volume = indicators.get('volume')
            volume_sma = indicators.get('volume_sma')
            
            if volume is not None and volume_sma is not None and len(volume) > 0 and len(volume_sma) > 0:
                current_volume = self.safe_indicator_value(volume, -1)
                avg_volume = self.safe_indicator_value(volume_sma, -1)
                
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    if volume_ratio > 2.0:
                        signal_strength += 2
                    elif volume_ratio > 1.5:
                        signal_strength += 1
            
            # Bollinger Bands analysis with safety
            bb_upper = self.safe_indicator_value(indicators.get('bb_upper'))
            bb_lower = self.safe_indicator_value(indicators.get('bb_lower'))
            
            if bb_upper > 0 and bb_lower > 0:
                if current_price <= bb_lower:
                    signal_strength += 2
                elif current_price >= bb_upper:
                    signal_strength += 2
            
            # Supertrend confirmation
            supertrend_dir = indicators.get('supertrend_direction')
            if supertrend_dir is not None and len(supertrend_dir) >= 2:
                current_dir = self.safe_indicator_value(supertrend_dir, -1)
                prev_dir = self.safe_indicator_value(supertrend_dir, -2)
                
                if current_dir != prev_dir:  # Direction change
                    signal_strength += 2
                elif abs(current_dir) > 0:  # Consistent direction
                    signal_strength += 1
            
            # BOS confirmation
            bos = indicators.get('bos')
            if bos is not None and len(bos) > 0:
                bos_signal = self.safe_indicator_value(bos, -1)
                if abs(bos_signal) > 0:
                    signal_strength += 1
            
            return min(signal_strength, 10)
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0
    
    def calculate_professional_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate professional indicators with comprehensive error handling"""
        try:
            if df is None or len(df) < 20:
                logger.warning("Insufficient data for indicator calculation")
                return {}
            
            indicators = {}
            
            # Ensure required columns exist and clean data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Missing column: {col}")
                    return {}
                
                # Fill NaN values
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            # Convert to numpy arrays with safety checks
            try:
                high_values = df['high'].to_numpy()
                low_values = df['low'].to_numpy()
                close_values = df['close'].to_numpy()
                volume_values = df['volume'].to_numpy()
                
                # Replace any remaining NaN values
                high_values = np.nan_to_num(high_values, nan=close_values)
                low_values = np.nan_to_num(low_values, nan=close_values)
                close_values = np.nan_to_num(close_values, nan=0)
                volume_values = np.nan_to_num(volume_values, nan=0)
                
            except Exception as e:
                logger.error(f"Error converting to numpy: {e}")
                return {}
            
            # Basic indicators with error handling
            try:
                indicators['rsi'] = talib.RSI(close_values, timeperiod=14)
            except Exception as e:
                logger.error(f"RSI calculation error: {e}")
                indicators['rsi'] = np.full(len(df), 50.0)  # Default neutral RSI
            
            # MACD
            try:
                macd, macd_signal, macd_hist = talib.MACD(close_values)
                indicators['macd'] = np.nan_to_num(macd, nan=0)
                indicators['macd_signal'] = np.nan_to_num(macd_signal, nan=0)
                indicators['macd_hist'] = np.nan_to_num(macd_hist, nan=0)
            except Exception as e:
                logger.error(f"MACD calculation error: {e}")
                indicators['macd'] = np.zeros(len(df))
                indicators['macd_signal'] = np.zeros(len(df))
                indicators['macd_hist'] = np.zeros(len(df))
            
            # Bollinger Bands
            try:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close_values, timeperiod=20)
                indicators['bb_upper'] = np.nan_to_num(bb_upper, nan=close_values)
                indicators['bb_middle'] = np.nan_to_num(bb_middle, nan=close_values)
                indicators['bb_lower'] = np.nan_to_num(bb_lower, nan=close_values)
            except Exception as e:
                logger.error(f"Bollinger Bands error: {e}")
                indicators['bb_upper'] = close_values * 1.02
                indicators['bb_middle'] = close_values
                indicators['bb_lower'] = close_values * 0.98
            
            # EMAs
            try:
                indicators['ema_20'] = talib.EMA(close_values, timeperiod=min(20, len(df)//2))
                indicators['ema_50'] = talib.EMA(close_values, timeperiod=min(50, len(df)//2))
                if len(df) > 100:
                    indicators['ema_200'] = talib.EMA(close_values, timeperiod=min(200, len(df)//2))
                else:
                    indicators['ema_200'] = indicators['ema_50']
                
                # Fill NaN values
                for key in ['ema_20', 'ema_50', 'ema_200']:
                    indicators[key] = np.nan_to_num(indicators[key], nan=close_values)
                    
            except Exception as e:
                logger.error(f"EMA calculation error: {e}")
                indicators['ema_20'] = close_values
                indicators['ema_50'] = close_values
                indicators['ema_200'] = close_values
            
            # ATR
            try:
                indicators['atr'] = talib.ATR(high_values, low_values, close_values, timeperiod=14)
                indicators['atr'] = np.nan_to_num(indicators['atr'], nan=close_values * 0.02)
            except Exception as e:
                logger.error(f"ATR calculation error: {e}")
                indicators['atr'] = close_values * 0.02  # Default 2% ATR
            
            # Volume indicators
            try:
                indicators['volume_sma'] = talib.SMA(volume_values, timeperiod=20)
                indicators['volume_sma'] = np.nan_to_num(indicators['volume_sma'], nan=volume_values)
                indicators['volume'] = volume_values
            except Exception as e:
                logger.error(f"Volume indicators error: {e}")
                indicators['volume_sma'] = volume_values
                indicators['volume'] = volume_values
            
            # Stochastic
            try:
                slowk, slowd = talib.STOCH(high_values, low_values, close_values)
                indicators['stoch_k'] = np.nan_to_num(slowk, nan=50.0)
                indicators['stoch_d'] = np.nan_to_num(slowd, nan=50.0)
            except Exception as e:
                logger.error(f"Stochastic error: {e}")
                indicators['stoch_k'] = np.full(len(df), 50.0)
                indicators['stoch_d'] = np.full(len(df), 50.0)
            
            # Advanced indicators with enhanced error handling
            try:
                indicators['williams_r'] = talib.WILLR(high_values, low_values, close_values)
                indicators['williams_r'] = np.nan_to_num(indicators['williams_r'], nan=-50.0)
            except Exception as e:
                logger.error(f"Williams %R error: {e}")
                indicators['williams_r'] = np.full(len(df), -50.0)
            
            try:
                indicators['mfi'] = talib.MFI(high_values, low_values, close_values, volume_values)
                indicators['mfi'] = np.nan_to_num(indicators['mfi'], nan=50.0)
            except Exception as e:
                logger.error(f"MFI error: {e}")
                indicators['mfi'] = np.full(len(df), 50.0)
            
            # Custom indicators
            try:
                indicators['bos'] = AdvancedIndicators.calculate_bos_indicator(df)
            except Exception as e:
                logger.error(f"BOS calculation error: {e}")
                indicators['bos'] = np.zeros(len(df))
                
            # Fibonacci levels
            try:
               indicators['fibonacci'] = AdvancedIndicators.calculate_fibonacci_levels(df)
    
               # Calculate fibonacci confluence if levels exist
               fib_data = indicators.get('fibonacci', {}) 
               all_fib_levels = []
                
               # Cek dulu apakah 'laci' retracements ada sebelum membukanya
               if fib_data.get('retracements'):
                   for level_data in fib_data['retracements'].values():
                       all_fib_levels.append(level_data['price'])
                
               # Lakukan hal yang sama untuk extensions
               if fib_data.get('extensions'):
                   for level_data in fib_data['extensions'].values():
                       all_fib_levels.append(level_data['price'])
        
               current_price = float(df['close'].iloc[-1])
               confluence_data = AdvancedIndicators.calculate_fibonacci_confluence(
                   all_fib_levels, current_price
               )
               indicators['fibonacci']['confluence'] = confluence_data
        
            except Exception as e:
               logger.error(f"Fibonacci calculation error: {e}")
               indicators['fibonacci'] = {}

            # Enhanced volume profile with fibonacci integration
            try:
              volume_profile = AdvancedIndicators.calculate_volume_profile(df, bins=20)
              indicators['volume_profile'] = volume_profile
    
              # Check if POC aligns with fibonacci levels
              fib_data = indicators.get('fibonacci', {})
              
              poc_price = 0 
              if fib_data and volume_profile:
                  poc_price = volume_profile.get('poc_price', 0)
              if poc_price > 0:
                  # Check alignment with fib levels
                  fib_alignment = False
                  if fib_data.get('retracements'):
                     for level_data in fib_data['retracements'].values():
                         if abs(poc_price - level_data['price']) / poc_price < 0.01:  # Within 1%
                            fib_alignment = True
                            break
            
                  indicators['fibonacci']['poc_alignment'] = fib_alignment
            
            except Exception as e:
                logger.error(f"Volume profile error: {e}")
                indicators['volume_profile'] = {}
            
            try:
                indicators['smart_money_index'] = AdvancedIndicators.calculate_smart_money_index(df)
            except Exception as e:
                logger.error(f"SMI calculation error: {e}")
                indicators['smart_money_index'] = np.zeros(len(df))
            
            try:
                indicators['market_structure'] = AdvancedIndicators.calculate_market_structure(df)
            except Exception as e:
                logger.error(f"Market structure error: {e}")
                indicators['market_structure'] = {'structure_score': 0, 'structure_type': 'neutral'}
            
            # Supertrend with error handling
            try:
                supertrend, direction = self.calculate_supertrend(df)
                indicators['supertrend'] = supertrend
                indicators['supertrend_direction'] = direction
            except Exception as e:
                logger.error(f"Supertrend calculation error: {e}")
                indicators['supertrend'] = close_values
                indicators['supertrend_direction'] = np.ones(len(df))
            
            return indicators
            
        except Exception as e:
            logger.error(f"Critical error calculating indicators: {e}")
            return {}
    
    def calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Supertrend with enhanced safety"""
        try:
            if df is None or len(df) < period:
                default_length = len(df) if df is not None else 10
                return np.zeros(default_length), np.ones(default_length)
            
            # Clean data
            high_vals = df['high'].fillna(method='ffill').to_numpy()
            low_vals = df['low'].fillna(method='ffill').to_numpy()
            close_vals = df['close'].fillna(method='ffill').to_numpy()
            
            hl2 = (high_vals + low_vals) / 2
            atr = talib.ATR(high_vals, low_vals, close_vals, timeperiod=period)
            atr = np.nan_to_num(atr, nan=np.mean(close_vals) * 0.02)  # Default 2% ATR
            
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            supertrend = np.zeros(len(df))
            direction = np.ones(len(df))
            
            # Initialize first value
            supertrend[0] = lower_band[0]
            
            for i in range(1, len(df)):
                try:
                    if close_vals[i] <= lower_band[i]:
                        supertrend[i] = upper_band[i]
                        direction[i] = -1
                    elif close_vals[i] >= upper_band[i]:
                        supertrend[i] = lower_band[i]
                        direction[i] = 1
                    else:
                        supertrend[i] = supertrend[i-1]
                        direction[i] = direction[i-1]
                except Exception:
                    supertrend[i] = supertrend[i-1] if i > 0 else close_vals[i]
                    direction[i] = direction[i-1] if i > 0 else 1
            
            return supertrend, direction
            
        except Exception as e:
            logger.error(f"Error calculating Supertrend: {e}")
            default_length = len(df) if df is not None else 10
            return np.zeros(default_length), np.ones(default_length)

    def get_dune_cex_flow_enhanced(self, symbol: str, headers: Dict, market_data: Dict) -> float:
        """Get enhanced CEX flow data with safety measures"""
        try:
            # For now, return mock netflow based on volume and price action
            # In production, this would connect to Dune Analytics API
            
            volume_24h = market_data.get('volume_24h', 0)
            price_change = market_data.get('price_change_24h', 0)
            
            # Mock netflow calculation
            if volume_24h > 0 and price_change != 0:
                # Positive price change with high volume suggests inflow
                netflow_ratio = (price_change / 100) * (volume_24h / 1000000)
                return min(max(netflow_ratio * 100000, -volume_24h * 0.1), volume_24h * 0.1)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting netflow for {symbol}: {e}")
            return 0.0

    async def scan_market_comprehensive(self, binance_client, headers, style_filter: Optional[str] = None) -> List[Dict]:
        """Comprehensive market scan with enhanced error handling"""
        
        try:
            logger.info("Starting comprehensive market scan...")
            
            # Get market data with retry
            tickers = {}
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    tickers = binance_client.fetch_tickers()
                    if tickers:
                        break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed to fetch tickers: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                    else:
                        logger.error("Failed to fetch tickers after all retries")
                        return []
            
            if not tickers:
                logger.error("No ticker data available")
                return []
            
            # Detect market regime (simplified)
            market_regime = MarketRegime.BULL_RANGING  # Default safe regime
            try:
                btc_data = binance_client.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=100)
                if len(btc_data) > 50:
                    btc_df = pd.DataFrame(btc_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    recent_change = (btc_df['close'].iloc[-1] - btc_df['close'].iloc[-24]) / btc_df['close'].iloc[-24] * 100
                    
                    if recent_change > 3:
                        market_regime = MarketRegime.BULL_TRENDING
                    elif recent_change < -3:
                        market_regime = MarketRegime.BEAR_TRENDING
                    else:
                        market_regime = MarketRegime.BULL_RANGING
                        
            except Exception as e:
                logger.warning(f"Failed to detect market regime: {e}")
            
            logger.info(f"Market regime detected: {market_regime.value}")
            
            # Get candidate symbols with safer filtering
            candidates = self.get_safe_candidates(tickers, style_filter)
            logger.info(f"Found {len(candidates)} candidate symbols")
            
            if not candidates:
                logger.warning("No candidates found")
                return []
            
            # Analyze symbols with controlled parallelism
            analysis_results = []
            max_concurrent = 3  # Reduced from 5 for stability
            
            for i in range(0, len(candidates[:15]), max_concurrent):  # Limit to 15 symbols
                batch = candidates[i:i + max_concurrent]
                batch_results = await self.analyze_symbol_batch(
                    batch, binance_client, headers, market_regime, style_filter
                )
                
                # Add valid results
                for result in batch_results:
                    if result and result.get('signal_type') != 'NEUTRAL':
                        analysis_results.append(result)
                
                # Rate limiting
                await asyncio.sleep(1)
            
            logger.info(f"Analysis complete. Found {len(analysis_results)} valid signals")
            
            if not analysis_results:
                return []
            
            # Final ranking and filtering
            final_results = self.rank_and_filter_results_safe(analysis_results, market_regime)
            
            logger.info(f"Final results: {len(final_results)} setups")
            return final_results
            
        except Exception as e:
            logger.error(f"Critical error in market scan: {e}")
            return []

    def get_safe_candidates(self, tickers: Dict, style_filter: Optional[str]) -> List[str]:
        """Get candidate symbols with safe filtering"""
        try:
            candidates = []
            
            # Adjusted thresholds for better signal detection
            min_volume = 500_000  # Reduced from 1M
            min_movement = 0.5 if style_filter != 'SCALPING' else 1.0  # Reduced thresholds
            
            for symbol, ticker in tickers.items():
                try:
                    if not symbol.endswith('/USDT'):
                        continue
                    
                    # Check required fields exist
                    if not all(k in ticker for k in ['quoteVolume', 'percentage']):
                        continue
                    
                    volume = float(ticker.get('quoteVolume', 0))
                    movement = abs(float(ticker.get('percentage', 0)))
                    
                    # Basic filters with safety
                    if volume >= min_volume and movement >= min_movement:
                        # Skip problematic symbols
                        base_symbol = symbol.split('/')[0]
                        if base_symbol not in ['BTCDOM', 'DEFI', 'USD']:  # Skip index tokens
                            candidates.append(symbol)
                            
                except Exception as e:
                    logger.debug(f"Error processing ticker {symbol}: {e}")
                    continue
            
            # Sort by composite score (volume + movement)
            def score_symbol(symbol):
                try:
                    ticker = tickers[symbol]
                    volume_score = min(float(ticker.get('quoteVolume', 0)) / 100000, 1000)
                    movement_score = min(abs(float(ticker.get('percentage', 0))) * 10, 200)
                    return volume_score + movement_score
                except:
                    return 0
            
            candidates.sort(key=score_symbol, reverse=True)
            return candidates[:30]  # Return top 30 candidates
            
        except Exception as e:
            logger.error(f"Error getting candidates: {e}")
            return []

    async def analyze_symbol_batch(self, symbols: List[str], binance_client, headers, 
                                 market_regime: MarketRegime, style_filter: Optional[str]) -> List[Dict]:
        """Analyze a batch of symbols concurrently"""
        results = []
        
        # Use ThreadPoolExecutor for I/O bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            for symbol in symbols:
                future = executor.submit(
                    self.analyze_symbol_safe, 
                    symbol, binance_client, headers, market_regime, style_filter
                )
                futures.append((symbol, future))
            
            # Collect results with timeout
            for symbol, future in futures:
                try:
                    result = future.result(timeout=20)  # 20 second timeout per symbol
                    if result:
                        results.append(result)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Analysis timeout for {symbol}")
                except Exception as e:
                    logger.warning(f"Analysis failed for {symbol}: {e}")
        
        return results

    def analyze_symbol_safe(self, symbol: str, binance_client, headers, 
                           market_regime: MarketRegime, style_filter: Optional[str]) -> Optional[Dict]:
        """Safe symbol analysis with comprehensive error handling"""
        try:
            # Get market data with retries
            market_data = self.get_enhanced_market_data_safe(symbol, binance_client)
            if not market_data or market_data.get('current_price', 0) <= 0:
                return None
            
            # Multi-timeframe analysis with fallback
            tf_analysis = self.analyze_multi_timeframe_safe(symbol, binance_client, style_filter)
            if not tf_analysis:
                return None
            
            # Calculate MTF consensus
            mtf_consensus = self.calculate_mtf_consensus_safe(tf_analysis)
            if mtf_consensus['confidence'] < self.config.min_mtf_confidence:
                logger.debug(f"{symbol}: Low MTF confidence {mtf_consensus['confidence']:.1f}%")
                return None
            
            # Get netflow (mock for now)
            netflow = self.get_dune_cex_flow_enhanced(symbol, headers, market_data)
            
            # Simple signal analysis
            signal_result = self.analyze_signal_safe(symbol, tf_analysis, market_data, netflow, style_filter)
            
            if signal_result['signal_type'] == 'NEUTRAL':
                return None
            
            # Risk management
            primary_tf = self.get_primary_timeframe_safe(tf_analysis)
            if not primary_tf or primary_tf not in tf_analysis:
                return None
                
            primary_data = tf_analysis[primary_tf]
            
            risk_mgmt = self.calculate_risk_management_safe(
                primary_data.get('indicators', {}),
                signal_result['signal_type'],
                primary_data.get('current_price', 0),
                style_filter
            )
            
            if not risk_mgmt:
                return None
            
            return {
                'symbol': symbol,
                'signal_type': signal_result['signal_type'],
                'confidence': signal_result['confidence'],
                'current_price': primary_data.get('current_price', 0),
                'market_data': market_data,
                'signal_components': signal_result.get('components', {}),
                'risk_mgmt': risk_mgmt,
                'tf_analysis': tf_analysis,
                'mtf_consensus': mtf_consensus,
                'netflow': netflow,
                'market_regime': market_regime
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

    def get_enhanced_market_data_safe(self, symbol: str, binance_client) -> Dict:
        """Get market data with comprehensive safety checks"""
        try:
            ticker = binance_client.fetch_ticker(symbol)
            
            if not ticker:
                return {}
            
            # Safely extract data with defaults
            current_price = float(ticker.get('last', 0))
            volume_24h = float(ticker.get('quoteVolume', 0))
            price_change_24h = float(ticker.get('percentage', 0))
            high_24h = float(ticker.get('high', current_price * 1.05))
            low_24h = float(ticker.get('low', current_price * 0.95))
            
            if current_price <= 0:
                return {}
            
            data = {
                'current_price': current_price,
                'volume_24h': volume_24h,
                'price_change_24h': price_change_24h,
                'high_24h': high_24h,
                'low_24h': low_24h
            }
            
            # Calculate range position
            daily_range = high_24h - low_24h
            if daily_range > 0:
                data['range_position'] = (current_price - low_24h) / daily_range
            else:
                data['range_position'] = 0.5
            
            # Volume ratio calculation
            try:
                ohlcv_data = binance_client.fetch_ohlcv(symbol, timeframe='1d', limit=7)
                if len(ohlcv_data) >= 3:
                    recent_volumes = [float(candle[5]) for candle in ohlcv_data[:-1]]  # Exclude current day
                    avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else volume_24h
                    data['volume_ratio'] = volume_24h / avg_volume if avg_volume > 0 else 1
                else:
                    data['volume_ratio'] = 1
            except:
                data['volume_ratio'] = 1
            
            # Volatility calculation
            if daily_range > 0 and current_price > 0:
                data['volatility_24h'] = daily_range / current_price
            else:
                data['volatility_24h'] = 0.02  # Default 2%
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {}

    def analyze_multi_timeframe_safe(self, symbol: str, binance_client, style_filter: Optional[str]) -> Dict:
        """Safe multi-timeframe analysis"""
        try:
            # Define timeframes based on style with fallbacks
            if style_filter == 'SCALPING':
                timeframes = {
                    '5m': {'weight': 3, 'periods': 80},
                    '15m': {'weight': 2, 'periods': 60}
                }
            elif style_filter == 'SWING':
                timeframes = {
                    '1h': {'weight': 3, 'periods': 80},
                    '4h': {'weight': 2, 'periods': 60}
                }
            else:
                # Default / DAY_TRADING
                timeframes = {
                    '15m': {'weight': 3, 'periods': 80},
                    '1h': {'weight': 2, 'periods': 60}
                }
            
            tf_analysis = {}
            
            for tf, config in timeframes.items():
                try:
                    ohlcv = binance_client.fetch_ohlcv(symbol, timeframe=tf, limit=config['periods'])
                    
                    if len(ohlcv) < 30:  # Minimum data requirement
                        continue
                        
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Calculate indicators
                    indicators = self.calculate_professional_indicators(df)
                    if not indicators:
                        continue
                    
                    current_price = float(df['close'].iloc[-1])
                    
                    # Trend analysis
                    trend_analysis = self.analyze_trend_professional(df, indicators)
                    
                    # Signal strength
                    signal_strength = self.calculate_signal_strength_advanced(indicators, current_price)
                    
                    tf_analysis[tf] = {
                        'trend': trend_analysis['trend'],
                        'trend_strength': trend_analysis['strength'],
                        'signal_strength': signal_strength,
                        'current_price': current_price,
                        'rsi': self.safe_indicator_value(indicators.get('rsi'), default_value=50),
                        'weight': config['weight'],
                        'indicators': indicators
                    }
                    
                except Exception as e:
                    logger.debug(f"Error analyzing {tf} for {symbol}: {e}")
                    continue
            
            return tf_analysis
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis for {symbol}: {e}")
            return {}

    def calculate_mtf_consensus_safe(self, tf_analysis: Dict) -> Dict:
        """Safe MTF consensus calculation"""
        try:
            if not tf_analysis:
                return {'consensus': 'NEUTRAL', 'confidence': 0}
            
            bullish_score = 0
            bearish_score = 0
            total_weight = 0
            
            for tf, data in tf_analysis.items():
                try:
                    weight = float(data.get('weight', 1))
                    trend = data.get('trend', 'NEUTRAL')
                    signal_strength = float(data.get('signal_strength', 0))
                    
                    # Convert trend to numeric score
                    if trend in ['STRONG_BULLISH', 'BULLISH']:
                        trend_score = 2 if 'STRONG' in trend else 1
                        bullish_score += trend_score * (signal_strength / 10) * weight
                    elif trend in ['STRONG_BEARISH', 'BEARISH']:
                        trend_score = 2 if 'STRONG' in trend else 1
                        bearish_score += trend_score * (signal_strength / 10) * weight
                    
                    total_weight += weight
                    
                except Exception as e:
                    logger.debug(f"Error processing timeframe {tf}: {e}")
                    continue
            
            if total_weight == 0:
                return {'consensus': 'NEUTRAL', 'confidence': 0}
            
            # Normalize scores
            bullish_score = bullish_score / total_weight
            bearish_score = bearish_score / total_weight
            
            # Determine consensus
            if bullish_score > bearish_score:
                consensus = 'BULLISH'
                confidence = min((bullish_score / (bullish_score + bearish_score + 0.01)) * 100, 100)
            elif bearish_score > bullish_score:
                consensus = 'BEARISH'
                confidence = min((bearish_score / (bullish_score + bearish_score + 0.01)) * 100, 100)
            else:
                consensus = 'NEUTRAL'
                confidence = 30  # Low confidence for neutral
            
            return {
                'consensus': consensus,
                'confidence': float(confidence),
                'bullish_score': float(bullish_score),
                'bearish_score': float(bearish_score)
            }
            
        except Exception as e:
            logger.error(f"Error calculating MTF consensus: {e}")
            return {'consensus': 'NEUTRAL', 'confidence': 0}

    def analyze_signal_safe(self, symbol: str, tf_analysis: Dict, market_data: Dict, netflow: float, style_filter: str) -> Dict:
        """Safe signal analysis dengan skor terpadu dan alur logika yang benar."""
        try:
            # LANGKAH 1: Validasi dan siapkan data utama
            if not tf_analysis:
                return {'signal_type': 'NEUTRAL', 'confidence': 0}

            primary_tf = self.get_primary_timeframe_safe(tf_analysis)
            if not primary_tf:
                return {'signal_type': 'NEUTRAL', 'confidence': 0}

            primary_data = tf_analysis[primary_tf]
            indicators = primary_data.get('indicators', {})
            current_price = primary_data.get('current_price', 0)
            if current_price == 0:
                return {'signal_type': 'NEUTRAL', 'confidence': 0}

            # LANGKAH 2: Hitung skor awal dari semua komponen analisis
            components = {'technical': 0, 'momentum': 0, 'volume': 0, 'sentiment': 0, 'structure': 0}

            # --- Perhitungan Skor Komponen (disalin dari kode Anda) ---
            # Technical score (RSI, MACD, BB)
            rsi_value = self.safe_indicator_value(indicators.get('rsi'), default_value=50)
            if rsi_value < 35: components['technical'] += 3
            elif rsi_value > 65: components['technical'] -= 3
            
            macd_hist = indicators.get('macd_hist')
            if macd_hist is not None and len(macd_hist) >= 2:
                current_hist = self.safe_indicator_value(macd_hist, -1)
                prev_hist = self.safe_indicator_value(macd_hist, -2)
                if current_hist > 0 and prev_hist <= 0: components['technical'] += 5 # Bullish crossover
                elif current_hist < 0 and prev_hist >= 0: components['technical'] -= 5 # Bearish crossover

            # Momentum score (trend alignment)
            bullish_tfs = sum(1 for data in tf_analysis.values() if 'BULLISH' in data.get('trend', ''))
            bearish_tfs = sum(1 for data in tf_analysis.values() if 'BEARISH' in data.get('trend', ''))
            if bullish_tfs > bearish_tfs: components['momentum'] = 5
            elif bearish_tfs > bullish_tfs: components['momentum'] = -5

            # Volume score
            volume_ratio = market_data.get('volume_ratio', 1)
            if volume_ratio > 1.5: components['volume'] = 5

            # Fibonacci & Volume Profile
            fibonacci_data = indicators.get('fibonacci', {})
            volume_profile = indicators.get('volume_profile', {})
            if fibonacci_data and volume_profile:
                at_fib = fibonacci_data.get('at_fib_level', False)
                near_fib = fibonacci_data.get('near_fib_level', False)
                confluence = fibonacci_data.get('confluence_count', 0)
                vah = volume_profile.get('vah_price')
                val = volume_profile.get('val_price')

                if at_fib: components['technical'] += 5
                elif near_fib: components['technical'] += 2
                if confluence > 1: components['technical'] += confluence * 2
                
                if val and current_price <= val * 1.005: components['technical'] += 7 # Support di VAL
                if vah and current_price >= vah * 0.995: components['technical'] -= 7 # Resistance di VAH

            # --- Akhir Perhitungan Skor Komponen ---
            
            # LANGKAH 3: Tentukan TIPE SINYAL awal
            total_pre_score = components['technical'] + components['momentum']
            preliminary_signal_type = 'NEUTRAL'
            if total_pre_score > 5: preliminary_signal_type = 'BULLISH'
            elif total_pre_score < -5: preliminary_signal_type = 'BEARISH'

            # Jika sinyal awal netral, tidak perlu lanjut
            if preliminary_signal_type == 'NEUTRAL':
                return {'signal_type': 'NEUTRAL', 'confidence': 0}

            # LANGKAH 4: Hitung Risk Management (SL/TP)
            risk_mgmt = self.calculate_risk_management_safe(
                indicators, preliminary_signal_type, current_price, style_filter
            )

            # LANGKAH 5: Hitung SKOR BONUS dari R:R dan Volume
            if risk_mgmt:
                best_rr = max(risk_mgmt.get('risk_reward_1', 0), risk_mgmt.get('risk_reward_2', 0))
                components['rr_bonus'] = min(best_rr * 4, 15)

            components['volume_bonus'] = min(volume_ratio * 2, 8)

            # LANGKAH 6: Hitung SKOR TOTAL dan CONFIDENCE FINAL
            total_score = sum(v for v in components.values() if isinstance(v, (int, float)))
            confidence = min(max(total_score * 1.5, 0), 100) # Pastikan tidak negatif dan tidak > 100

            # LANGKAH 7: Return hasil lengkap
            return {
                'signal_type': preliminary_signal_type,
                'confidence': confidence,
                'components': components,
                'risk_mgmt': risk_mgmt
            }

        except Exception as e:
            logger.error(f"Error in signal analysis for {symbol}: {e}", exc_info=True)
            return {'signal_type': 'NEUTRAL', 'confidence': 0}

    def get_primary_timeframe_safe(self, tf_analysis: Dict) -> Optional[str]:
        """Get primary timeframe safely"""
        try:
            if not tf_analysis:
                return None
            
            # Priority order
            preferred_order = ['15m', '1h', '5m', '4h', '30m']
            
            for tf in preferred_order:
                if tf in tf_analysis:
                    return tf
            
            # Return first available
            return list(tf_analysis.keys())[0] if tf_analysis else None
            
        except Exception as e:
            logger.error(f"Error getting primary timeframe: {e}")
            return None

    def calculate_risk_management_safe(self, indicators: Dict, signal_type: str, 
                                     current_price: float, style_filter: str) -> Optional[Dict]:
        """Safe risk management calculation"""
        try:
            if current_price <= 0:
                return None
            
            # Get ATR safely
            atr_array = indicators.get('atr')
            current_atr = self.safe_indicator_value(atr_array, default_value=current_price * 0.02)
            
            # Style-specific multipliers
            style_multipliers = {
                'SCALPING': {'stop': 1.0, 'tp1': 1.5, 'tp2': 2.5, 'tp3': 4.0},
                'DAY_TRADING': {'stop': 1.5, 'tp1': 2.0, 'tp2': 3.5, 'tp3': 6.0},
                'SWING': {'stop': 2.0, 'tp1': 3.0, 'tp2': 5.0, 'tp3': 8.0}
            }
            
            multipliers = style_multipliers.get(style_filter, style_multipliers['DAY_TRADING'])
            
            # Calculate levels
            if signal_type == 'BULLISH':
                stop_loss = current_price - (current_atr * multipliers['stop'])
                take_profit_1 = current_price + (current_atr * multipliers['tp1'])
                take_profit_2 = current_price + (current_atr * multipliers['tp2'])
                take_profit_3 = current_price + (current_atr * multipliers['tp3'])
            else:  # BEARISH
                stop_loss = current_price + (current_atr * multipliers['stop'])
                take_profit_1 = current_price - (current_atr * multipliers['tp1'])
                take_profit_2 = current_price - (current_atr * multipliers['tp2'])
                take_profit_3 = current_price - (current_atr * multipliers['tp3'])
            
            # Calculate risk metrics
            stop_distance = abs(current_price - stop_loss)
            stop_distance_percent = (stop_distance / current_price) * 100
            
            # Risk validation
            if stop_distance_percent > 10:  # Max 10% stop loss
                return None
            
            # Calculate R:R ratios safely
            rr1 = abs(take_profit_1 - current_price) / stop_distance if stop_distance > 0 else 0
            rr2 = abs(take_profit_2 - current_price) / stop_distance if stop_distance > 0 else 0
            rr3 = abs(take_profit_3 - current_price) / stop_distance if stop_distance > 0 else 0
            
            return {
                'stop_loss': stop_loss,
                'take_profit_1': take_profit_1,
                'take_profit_2': take_profit_2,
                'take_profit_3': take_profit_3,
                'risk_reward_1': rr1,
                'risk_reward_2': rr2,
                'risk_reward_3': rr3,
                'stop_distance': stop_distance,
                'stop_distance_percent': stop_distance_percent,
                'atr_value': current_atr,
                'volatility_ratio': current_atr / current_price if current_price > 0 else 0.02
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk management: {e}")
            return None

    def rank_and_filter_results_safe(self, results: List[Dict], market_regime: MarketRegime) -> List[Dict]:
        """Safe ranking and filtering dengan debug print yang detail."""
        try:
            if not results:
                return []
            
            print("\n" + "🔍" * 30)
            print("MEMULAI PROSES FILTER KUALITAS AKHIR...")
            print("🔍" * 30)

            filtered_results = []
            
            for result in results:
                try:
                    # Ambil semua data yang diperlukan dengan aman
                    symbol = result.get('symbol', 'UNKNOWN')
                    confidence = float(result.get('confidence', 0))
                    risk_mgmt = result.get('risk_mgmt', {})
                    
                    if not risk_mgmt:
                        print(f"\n===== DEBUGGING {symbol} =====")
                        print(" -> REJECTED: Tidak ada data Risk Management.")
                        continue

                    best_rr = max(risk_mgmt.get('risk_reward_1', 0), risk_mgmt.get('risk_reward_2', 0))
                    stop_dist_pct = risk_mgmt.get('stop_distance_percent', 100)

                    # --- INI BAGIAN PENTING UNTUK DEBUG ---
                    print(f"\n===== DEBUGGING {symbol} =====")
                    print(f"Conf: {confidence:.1f}% (Min: {self.config.min_mtf_confidence}%) | R:R: {best_rr:.1f} (Min: 1.2) | Risk: {stop_dist_pct:.1f}% (Max: 8%)")

                    # Filter 1: Confidence
                    if confidence < self.config.min_mtf_confidence:
                        print(f" -> REJECTED: Confidence terlalu rendah.")
                        continue
                    
                    # Filter 2: Risk Reward Ratio
                    if best_rr < 1.2:
                        print(f" -> REJECTED: Risk:Reward Ratio di bawah 1.2.")
                        continue
                        
                    # Filter 3: Stop Loss Distance
                    if stop_dist_pct > 8:
                        print(f" -> REJECTED: Jarak Stop Loss terlalu lebar (risiko > 8%).")
                        continue
                    
                    print(" -> ✅ LULUS SEMUA FILTER!")
                    filtered_results.append(result)
                
                except Exception as e:
                    logger.debug(f"Error filtering result for {symbol}: {e}")
                    continue
            
            print("\n" + "✅" * 30)
            print("PROSES FILTER SELESAI.")
            print("✅" * 30)
            
            # Sort by confidence
            filtered_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Return top 5 results
            return filtered_results[:5]
            
        except Exception as e:
            logger.error(f"Error ranking results: {e}")
            return []
            
    def format_single_signal_message(self, result: Dict) -> str:
        """Memformat satu hasil analisis menjadi pesan yang detail dan lengkap."""
        try:
            symbol = result.get('symbol', 'N/A')
            signal_type = result.get('signal_type', 'N/A')
            confidence = result.get('confidence', 0)
            risk_mgmt = result.get('risk_mgmt', {})
            current_price = result.get('current_price', 0)
            
            tf_analysis = result.get('tf_analysis', {})
            primary_tf = self.get_primary_timeframe_safe(tf_analysis)
            primary_indicators = tf_analysis.get(primary_tf, {}).get('indicators', {})

            # Buat pesan
            message = (
                f"🚀 **ANALISIS MANUAL: ${symbol}** 🚀\n"
                f"{'='*30}\n"
                f"**Sinyal:** {signal_type} ({confidence:.1f}% Confidence)\n"
                f"**Gaya Trading:** Day Trading (Default)\n\n"  # <-- BARIS BARU UNTUK KEJELASAN
                
                f"**Level Kunci & Rencana Trade:**\n"
                f"• **Harga Saat Ini:** ${current_price:,.4f}\n"
                f"• Entry Zone: {self.generate_entry_zone(result).get('text', '(Gagal menghitung)')}\n"
                f"• **Target 1:** ${risk_mgmt.get('take_profit_1', 0):,.4f} (R:R {risk_mgmt.get('risk_reward_1', 0):.1f})\n"
                f"• **Target 2:** ${risk_mgmt.get('take_profit_2', 0):,.4f} (R:R {risk_mgmt.get('risk_reward_2', 0):.1f})\n"
                f"• **Target 3:** ${risk_mgmt.get('take_profit_3', 0):,.4f} (R:R {risk_mgmt.get('risk_reward_3', 0):.1f})\n"  # <-- BARIS BARU UNTUK TP3
                f"• **Stop Loss:** ${risk_mgmt.get('stop_loss', 0):,.4f}\n\n"

                f"**Analisis Kunci:**\n"
                f"• **Status Fibonacci:** {get_fib_status(primary_indicators)}\n"
                f"• **Volume Profile:** POC @ ${primary_indicators.get('volume_profile', {}).get('poc_price', 0):,.2f}, "
                f"VAL @ ${primary_indicators.get('volume_profile', {}).get('val_price', 0):,.2f}, "
                f"VAH @ ${primary_indicators.get('volume_profile', {}).get('vah_price', 0):,.2f}\n\n"

                f"⚠️ **Risiko Trade:** {risk_mgmt.get('stop_distance_percent', 0):.1f}%"
            )
            return message
        except Exception as e:
            logger.error(f"Error memformat pesan untuk {symbol}: {e}")
            return f"❌ Terjadi kesalahan saat memformat hasil untuk {symbol}."
            
    def manual_analysis_logic(self, symbol: str) -> dict:
        """Menjalankan logika analisis & membuat chart, mengembalikan pesan dan path gambar."""
        from .chart_generator import generate_chart

        logger.info(f"Memulai analisis manual untuk {symbol}...")

        result = self.analyze_symbol_safe(
            symbol=symbol,
            binance_client=binance,
            headers=headers,
            market_regime=MarketRegime.BULL_RANGING,
            style_filter='DAY_TRADING'
        )

        if result and result.get('signal_type') != 'NEUTRAL':
            entry_zone_data = self.generate_entry_zone(result)
            result['entry_zone_text'] = entry_zone_data.get('text', '(N/A)')
            result['entry_zone_raw'] = entry_zone_data.get('raw', [0, 0])

            message = self.format_single_signal_message(result)
            chart_path = generate_chart(symbol, result)

            return {"message": message, "chart": chart_path}
        else:
            message = f"📊 **Analisis Selesai untuk {symbol}**\n\nSaat ini tidak ditemukan sinyal trading yang kuat."
            return {"message": message, "chart": None}
            
    def calculate_portfolio_pnl(self) -> dict:
        """Menghitung total P&L dari semua posisi yang sedang dilacak."""
        
        total_pnl = 0.0
        # Kita asumsikan total modal awal untuk perhitungan persentase
        total_initial_value = 100000 
        
        # Ambil semua sinyal yang statusnya 'TRACKED'
        tracked_signals = self.database.get_tracked_signals()
        
        if not tracked_signals:
            return {"total_pnl": 0, "pnl_percent": 0}
            
        for signal in tracked_signals:
            try:
                symbol = signal['symbol']
                signal_type = signal['signal_type']
                entry_price = float(signal['entry_price'])
                
                # Ambil harga pasar saat ini
                current_price = float(binance.fetch_ticker(symbol)['last'])
                
                # Hitung P&L berdasarkan tipe sinyal
                if signal_type == 'BULLISH':
                    pnl = (current_price - entry_price)
                elif signal_type == 'BEARISH':
                    pnl = (entry_price - current_price)
                else:
                    pnl = 0
                
                # Asumsi ukuran posisi $1000 per trade untuk simulasi
                # Nanti ini bisa kita buat lebih canggih
                position_size_usd = 1000 
                asset_amount = position_size_usd / entry_price
                
                total_pnl += pnl * asset_amount

            except Exception as e:
                logger.error(f"Gagal menghitung P&L untuk {signal.get('symbol')}: {e}")
                continue
                
        pnl_percent = (total_pnl / total_initial_value) * 100
        
        return {
            "total_pnl": total_pnl,
            "pnl_percent": pnl_percent
        }
        
    def start_scanner(self, profile='NORMAL'):
        """Mengaktifkan auto scanner dengan profil tertentu."""
        if self.auto_scanner_active:
            return f"ℹ️ Auto scanner sudah aktif dalam mode {getattr(self, 'active_profile', 'NORMAL')}."

        self.auto_scanner_active = True
        self.active_profile = profile
        
        # Terapkan setelan berdasarkan profil
        if profile == 'PROFESSIONAL':
            self.config.min_mtf_confidence = 60 # Standar lebih tinggi
            self.config.min_signal_strength = 7
            logger.info("Auto Scanner DIAKTIFKAN (Mode Profesional).")
            return "✅ Auto scanner mode **Profesional** telah DIAKTIFKAN. Standar sinyal lebih ketat."
        else: # NORMAL
            self.config.min_mtf_confidence = 40 # Standar normal
            self.config.min_signal_strength = 6
            logger.info("Auto Scanner DIAKTIFKAN (Mode Normal).")
            return "✅ Auto scanner mode **Normal** telah DIAKTIFKAN. Bot akan mulai berburu sinyal."
            
    def generate_briefing(self) -> str:
        """Menghasilkan rangkuman lengkap (briefing) kondisi saat ini."""
        try:
            # 1. Dapatkan Sentimen Pasar dari tren BTC
            btc_trend_data = self.analyze_trend_professional(
                pd.DataFrame(binance.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=100), 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']),
                self.calculate_professional_indicators(pd.DataFrame(binance.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=100), 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']))
            )
            market_sentiment = btc_trend_data.get('trend', 'TIDAK DIKETAHUI').replace('_', ' ')

            # 2. Dapatkan Ringkasan Portfolio
            pnl_data = self.calculate_portfolio_pnl()
            portfolio_stats = self.database.get_portfolio_stats()
            open_positions = portfolio_stats.get('open_positions', 0)
            total_pnl = pnl_data.get('total_pnl', 0)
            pnl_emoji = "✅" if total_pnl >= 0 else "😥"

            # 3. Dapatkan Performa Sinyal
            signal_stats = self.database.get_performance_stats()
            win_rate = signal_stats.get('win_rate', 0)

            # 4. Susun Pesan Briefing
            briefing_msg = (
                f" briefing **LAPORAN BRIEFING EKSEKUTIF** briefing\n"
                f"{datetime.now().strftime('%A, %d %B %Y %H:%M WIB')}\n"
                f"{'='*30}\n\n"
                f"** Ringkasan Pasar:**\n"
                f"• Sentimen Saat Ini: **{market_sentiment}** (berdasarkan tren BTC 4 Jam)\n\n"
                f"** Ringkasan Portfolio:**\n"
                f"• Posisi Aktif Dilacak: **{open_positions}**\n"
                f"• P&L Terbuka: **${total_pnl:,.2f}** {pnl_emoji}\n\n"
                f"** Performa Sinyal (Historis):**\n"
                f"• Tingkat Kemenangan: **{win_rate:.2f}%**\n\n"
                f"**Status Sistem:**\n"
                f"• Auto Scanner: {'🟢 AKTIF' if self.auto_scanner_active else '⏹️ NONAKTIF'}"
            )
            return briefing_msg

        except Exception as e:
            logger.error(f"Gagal membuat briefing: {e}")
            return "❌ Gagal menghasilkan briefing. Coba lagi nanti."

# ===============================
# TELEGRAM BOT HANDLERS
# ===============================

# Bot configuration
binance = ccxt.binance({
    'enableRateLimit': True,
    'timeout': 30000,
    'options': {
        'defaultType': 'spot'  # Ensure spot trading
    }
})

binance_futures = ccxt.binanceusdm({'options': {'defaultType': 'future'}})

# Professional configuration
trading_config = TradingConfig()
trading_engine = TradingEngine(trading_config)

DUNE_API_KEY = os.getenv("DUNE_API_KEY")

# Enhanced headers
headers = {
    "X-Dune-API-Key": DUNE_API_KEY,
    "Content-Type": "application/json",
    "User-Agent": "Professional-Trading-Bot/2.0"
}

bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

def create_main_menu():
    """Create the main menu keyboard"""
    keyboard = [
        [
            InlineKeyboardButton("⚡ Quick Scalping", callback_data='quick_scalping'),
            InlineKeyboardButton("📈 Day Trading", callback_data='day_trading'),
        ],
        [
            InlineKeyboardButton("📊 Swing Trading", callback_data='swing_trading'),
            InlineKeyboardButton("🎯 Manual Analysis", callback_data='manual_analysis'),
        ],
        [
            InlineKeyboardButton("🤖 Mode Otomatis", callback_data='auto_scanner_menu'),
            InlineKeyboardButton("📈 Sentimen Pasar", callback_data='market_sentiment'),
        ],
        [
            InlineKeyboardButton("📖 Panduan", callback_data='show_tutorial'),
            InlineKeyboardButton("📊 Portfolio", callback_data='portfolio'),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler (Versi Final dengan Spek Lengkap)."""
    
    # --- PESAN SELAMAT DATANG EDISI GOD TIER ---
    welcome_message = (
        "🚀 **SELAMAT DATANG DI ASISTEN TRADING OVERPOWERED v3.0** 🚀\n"
        f"*{datetime.now(timezone('Asia/Jakarta')).strftime('%A, %d %B %Y')}*\n"
        "==============================\n\n"
        
        "**KEMAMPUAN INTI:**\n"
        "• **Analisis Multi-Timeframe:** `Scalping`, `Day Trading`, `Swing`.\n"
        "• **Analisis On-Demand:** Cukup ketik `NAMA/USDT`.\n"
        "• **Mode Otomatis 24/7:** Pemindaian pasar proaktif.\n\n"
        
        "**FITUR INTELIJEN:**\n"
        "• **Indikator Lengkap:** EMA, RSI, MACD, dll.\n"
        "• **Analisis Institusional:** Fibonacci & Volume Profile (POC, VAH, VAL).\n"
        "• **Deteksi Break of Structure (BOS)** terintegrasi.\n"
        "• **Rekomendasi Entry Zone Cerdas**.\n\n"
        
        "**FITUR VISUAL:**\n"
        "• **Generator Chart Profesional** Otomatis.\n\n"
        
        "**FITUR ASISTEN:**\n"
        "• **Watchlist & Alerts** Proaktif (`/watch`).\n"
        "• **Jurnal Trading & Pelacakan Posisi** (`/portfolio`).\n"
        "• **Dashboard Performa Sinyal** (`/stats`).\n"
        "• **Panel Kontrol & Panduan**.\n\n"
        
        "**SUMBER DATA MULTI-DIMENSI:**\n"
        "• Binance (Spot & Futures), CoinGecko, FRED, NewsAPI, Forex Factory.\n\n"
        "==============================\n"
        "Pilih misi Anda di bawah ini."
    )
    # -----------------------------------
    
    menu = create_main_menu()
    
    if update.message:
        await update.message.reply_text(
            welcome_message, 
            reply_markup=menu,
            parse_mode='Markdown'
        )
    else:
        try:
            await update.callback_query.edit_message_text(
                welcome_message, 
                reply_markup=menu,
                parse_mode='Markdown'
            )
        except BadRequest as e:
            if "Message is not modified" in str(e):
                pass
            else:
                raise e

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks (versi master final yang sudah bersih)."""
    query = update.callback_query
    await query.answer()
    data = query.data
    
    try:
        # --- Navigasi Menu Utama ---
        if data == 'quick_scalping':
            await execute_scalping_scan(query, context)
        elif data == 'day_trading':
            await execute_daytrading_scan(query, context)
        elif data == 'swing_trading':
            await execute_swing_scan(query, context)
        elif data == 'manual_analysis':
            await show_manual_menu(query, context)
        elif data == 'market_sentiment':
            await show_sentiment_report(query, context)            
        elif data == 'back_main':
            await start(update, context)

        # --- Menu Portfolio ---
        elif data == 'portfolio':
            await show_portfolio_menu(query, context)
        elif data == 'show_stats':
            # 1. Langsung panggil database dari sini
            stats = trading_engine.database.get_performance_stats()
            
            # 2. Ambil semua datanya
            total = stats['total_trades']
            wins = stats['wins']
            losses = stats['losses']
            win_rate = stats['win_rate']
            
            # 3. Buat pesannya
            message = (
                f"📊 **LAPORAN PERFORMA SINYAL** 📊\n"
                f"{'='*30}\n"
                f"• Total Trade Selesai: {total}\n"
                f"• ✅ Menang (Win): {wins}\n"
                f"• 😥 Kalah (Loss): {losses}\n\n"
                f"• **🎯 Tingkat Kemenangan (Win Rate): {win_rate:.2f}%**"
            )
            
            # 4. Edit pesan yang sudah ada (Portfolio) menjadi pesan stats
            #    Ini lebih elegan daripada mengirim pesan baru
            keyboard = [[InlineKeyboardButton("↩️ Kembali ke Portfolio", callback_data='portfolio')]]
            await query.edit_message_text(
                message, 
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            await query.answer()

        # --- Menu Panduan & Bantuan (Settings) ---
        elif data == 'show_tutorial':
            await show_tutorial_menu(query, context)

        # --- Menu Mode Otomatis (Auto Scanner) ---
        elif data == 'auto_scanner_menu':
            await show_auto_scanner_menu(query, context)
        elif data == 'start_scanner_normal':
            response_text = trading_engine.start_scanner(profile='NORMAL')
            await query.answer(text=response_text, show_alert=True)
            await show_auto_scanner_menu(query, context)
        elif data == 'start_scanner_pro':
            response_text = trading_engine.start_scanner(profile='PROFESSIONAL')
            await query.answer(text=response_text, show_alert=True)
            await show_auto_scanner_menu(query, context)
        elif data == 'stop_scanner':
            response_text = trading_engine.stop_scanner()
            await query.answer(text=response_text)
            await show_auto_scanner_menu(query, context)

        # --- Logika Tombol Pelacak Sinyal ---
        elif data.startswith("track_"):
            try:
                signal_id = int(data.split('_')[1])
                trading_engine.database.track_signal(signal_id)
                keyboard = [[InlineKeyboardButton("✅ Sedang Dilacak", callback_data="do_nothing")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_reply_markup(reply_markup=reply_markup)
                await query.answer(text=f"Sinyal #{signal_id} sekarang sedang dilacak!")
            except (IndexError, ValueError) as e:
                logger.error(f"Error parsing callback_data for tracking: {data}, error: {e}")
                await query.answer(text="Error: Gagal melacak sinyal.")
        
        elif data == "do_nothing":
            await query.answer()
            
        else:
            await query.edit_message_text(
                "❌ Perintah tidak dikenal. Kembali ke menu utama.",
                reply_markup=create_main_menu()
            )

    except BadRequest as e:
        if "Message is not modified" in str(e):
            pass 
        elif "Query is too old" in str(e):
            await query.answer("Tombol ini sudah kedaluwarsa. Silakan gunakan /start lagi.", show_alert=True)
        else:
            logger.error(f"Button handler BadRequest error: {e}", exc_info=True)
            
    except Exception as e:
        # PASTIKAN BLOK INI TIDAK KOSONG
        logger.error(f"Button handler error: {e}", exc_info=True)
        try:
            # Coba kirim pesan balasan ke pengguna
            await query.message.reply_text(f"Terjadi error: {e}\nSilakan coba lagi.")
        except Exception as inner_e:
            # Jika gagal (misal karena query terlalu tua), catat saja
            logger.error(f"Gagal mengirim pesan error balasan: {inner_e}")

async def execute_scalping_scan(query, context):
    """Execute scalping analysis with enhanced error handling"""
    processing_msg = (
        "⚡ **Ultra-High Frequency ANALYSIS**\n\n"
        "🔬 Applying scalping filters...\n"
        "📊 Optimizing for scalping timeframes...\n"
        "⚖️ Calculating style-specific risk...\n"
        "🎯 Finding optimal setups...\n\n"
        "⏳ **Professional analysis in progress...**"
    )
    
    await query.edit_message_text(processing_msg, parse_mode='Markdown')
    
    try:
        chat_id = query.message.chat_id
        
        # Load markets with retry
        await load_markets_with_retry(binance, max_retries=3)
        
        # Execute scalping scan
        results = await trading_engine.scan_market_comprehensive(
            binance, headers, style_filter='SCALPING'
        )
        
        if results:
            await send_trading_results(context, chat_id, results, 'SCALPING')
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ No scalping setups found meeting our criteria"
            )
        
        # Return to menu
        completion_keyboard = [
            [
                InlineKeyboardButton("🔄 New Scan", callback_data='quick_scalping'),
                InlineKeyboardButton("↩️ Main Menu", callback_data='back_main')
            ]
        ]
        
        await context.bot.send_message(
            chat_id=chat_id,
            text="✅ Scalping analysis completed",
            reply_markup=InlineKeyboardMarkup(completion_keyboard)
        )
        
    except Exception as e:
        logger.error(f"Scalping scan error: {e}")
        error_msg = f"❌ **SCALPING ANALYSIS ERROR**: {str(e)}"
        
        keyboard = [[
            InlineKeyboardButton("🔄 Retry", callback_data='quick_scalping'),
            InlineKeyboardButton("↩️ Main Menu", callback_data='back_main')
        ]]
        
        await query.edit_message_text(
            error_msg, 
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

async def execute_daytrading_scan(query, context):
    """Execute day trading analysis"""
    processing_msg = (
        "📈 **Intraday Professional ANALYSIS**\n\n"
        "🔬 Applying daytrading filters...\n"
        "📊 Optimizing for daytrading timeframes...\n"
        "⚖️ Calculating style-specific risk...\n"
        "🎯 Finding optimal setups...\n\n"
        "⏳ **Professional analysis in progress...**"
    )
    
    await query.edit_message_text(processing_msg, parse_mode='Markdown')
    
    try:
        chat_id = query.message.chat_id
        
        await load_markets_with_retry(binance, max_retries=3)
        
        results = await trading_engine.scan_market_comprehensive(
            binance, headers, style_filter='DAY_TRADING'
        )
        
        if results:
            await send_trading_results(context, chat_id, results, 'DAY_TRADING')
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ No day trading setups found meeting our criteria"
            )
        
        completion_keyboard = [
            [
                InlineKeyboardButton("🔄 New Scan", callback_data='day_trading'),
                InlineKeyboardButton("↩️ Main Menu", callback_data='back_main')
            ]
        ]
        
        await context.bot.send_message(
            chat_id=chat_id,
            text="✅ Day trading analysis completed",
            reply_markup=InlineKeyboardMarkup(completion_keyboard)
        )
        
    except Exception as e:
        logger.error(f"Day trading scan error: {e}")
        error_msg = f"❌ **DAYTRADING ANALYSIS ERROR**: {str(e)}"
        
        keyboard = [[
            InlineKeyboardButton("🔄 Retry", callback_data='day_trading'),
            InlineKeyboardButton("↩️ Main Menu", callback_data='back_main')
        ]]
        
        await query.edit_message_text(
            error_msg, 
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

async def execute_swing_scan(query, context):
    """Execute swing trading analysis"""
    processing_msg = (
        "📊 **Multi-Day Position ANALYSIS**\n\n"
        "🔬 Applying swing filters...\n"
        "📊 Optimizing for swing timeframes...\n"
        "⚖️ Calculating style-specific risk...\n"
        "🎯 Finding optimal setups...\n\n"
        "⏳ **Professional analysis in progress...**"
    )
    
    await query.edit_message_text(processing_msg, parse_mode='Markdown')
    
    try:
        chat_id = query.message.chat_id
        
        await load_markets_with_retry(binance, max_retries=3)
        
        results = await trading_engine.scan_market_comprehensive(
            binance, headers, style_filter='SWING'
        )
        
        if results:
            await send_trading_results(context, chat_id, results, 'SWING')
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ No swing trading setups found meeting our criteria"
            )
        
        completion_keyboard = [
            [
                InlineKeyboardButton("🔄 New Scan", callback_data='swing_trading'),
                InlineKeyboardButton("↩️ Main Menu", callback_data='back_main')
            ]
        ]
        
        await context.bot.send_message(
            chat_id=chat_id,
            text="✅ Swing trading analysis completed",
            reply_markup=InlineKeyboardMarkup(completion_keyboard)
        )
        
    except Exception as e:
        logger.error(f"Swing scan error: {e}")
        error_msg = f"❌ **SWING ANALYSIS ERROR**: {str(e)}"
        
        keyboard = [[
            InlineKeyboardButton("🔄 Retry", callback_data='swing_trading'),
            InlineKeyboardButton("↩️ Main Menu", callback_data='back_main')
        ]]
        
        await query.edit_message_text(
            error_msg, 
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

async def show_auto_scanner_menu(query, context):
    """Menampilkan menu kontrol untuk Mode Otomatis."""
    if trading_engine.auto_scanner_active:
        status = f"🟢 AKTIF (Mode: **{getattr(trading_engine, 'active_profile', 'N/A')}**)"
    else:
        status = "⏹️ NONAKTIF"
    
    menu_msg = (
        f"🤖 **PUSAT KONTROL MODE OTOMATIS** 🤖\n\n"
        f"**Status Saat Ini:** {status}\n\n"
        "Pilih mode pemindaian otomatis di bawah ini. Bot akan bekerja 24/7 di latar belakang untuk mencari dan mengirimi Anda 'Proposal Trade' berkualitas."
    )
    
    keyboard = [
        [
            InlineKeyboardButton("▶️ Mulai (Normal)", callback_data='start_scanner_normal'),
            InlineKeyboardButton("▶️ Mulai (Profesional)", callback_data='start_scanner_pro')
        ],
        [
            InlineKeyboardButton("⏹️ Hentikan Pemindaian", callback_data='stop_scanner')
        ],
        [
            InlineKeyboardButton("↩️ Kembali ke Menu Utama", callback_data='back_main')
        ]
    ]
    
    try:
        await query.edit_message_text(
            menu_msg,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    except BadRequest as e:
        # Abaikan error "Message is not modified" dengan anggun
        if "Message is not modified" in str(e):
            pass
        else:
            # Jika error lain, tetap tampilkan
            logger.error(f"Error di show_auto_scanner_menu: {e}")

# Placeholder implementations for missing functions
async def show_manual_menu(query, context):
    """Show manual analysis menu"""
    menu_msg = (
        "🎯 **MANUAL ANALYSIS**\n\n"
        "Enter a symbol to analyze (e.g., BTC/USDT)\n"
        "Or select from popular options:"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("BTC/USDT", callback_data='analyze_BTC/USDT'),
            InlineKeyboardButton("ETH/USDT", callback_data='analyze_ETH/USDT')
        ],
        [
            InlineKeyboardButton("SOL/USDT", callback_data='analyze_SOL/USDT'),
            InlineKeyboardButton("ADA/USDT", callback_data='analyze_ADA/USDT')
        ],
        [
            InlineKeyboardButton("↩️ Back", callback_data='back_main')
        ]
    ]
    
    await query.edit_message_text(
        menu_msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_tutorial_menu(query, context):
    """Menampilkan teks tutorial lengkap dengan format yang sudah diperbaiki."""
    tutorial_text = (
        "📖 **PANDUAN LENGKAP BOT** 📖\n\n"
        "Berikut adalah penjelasan untuk setiap fitur utama:\n\n"
        "**1. Mode Pemindaian (Scalping/Day/Swing)**\n"
        "   - **Kegunaan:** Mencari sinyal di seluruh pasar secara otomatis.\n"
        "   - **Cara Pakai:** Tekan tombolnya di menu utama. Bot akan menampilkan 3 sinyal terbaik.\n\n"
        "**2. Mode Otomatis**\n"
        "   - **Kegunaan:** Mengaktifkan bot untuk memindai pasar 24/7 dan mengirim 'Proposal Trade' jika menemukan sinyal super kuat.\n"
        "   - **Cara Pakai:** Masuk ke menu `Mode Otomatis`, lalu tekan `Mulai (Normal/Profesional)`.\n\n"
        "**3. Analisis Manual (Ketik Simbol)**\n"
        "   - **Kegunaan:** Mendapatkan analisis mendalam untuk satu koin spesifik kapan saja.\n"
        "   - **Cara Pakai:** Cukup ketik nama koin di chat, contoh: `BTC/USDT`.\n\n"
        "**4. Watchlist (`/watch`)**\n"
        "   - **Kegunaan:** Memantau koin favorit Anda. Bot akan mengirim alert jika ada sinyal kuat.\n"
        "   - **Cara Pakai:** Kirim perintah `/watch NAMA_KOIN`, contoh: `/watch ETH`.\n\n"
        "**5. Portfolio & Jurnal**\n"
        "   - **Kegunaan:** Melihat ringkasan posisi yang sedang Anda lacak dan P&L real-time (simulasi).\n"
        "   - **Cara Pakai:** Tekan tombol 'Lacak Trade Ini' pada sebuah sinyal, lalu cek hasilnya di menu `Portfolio`.\n\n"
        "**6. Statistik (`/stats`)**\n"
        "   - **Kegunaan:** Melihat 'rapor' atau tingkat keberhasilan dari sinyal-sinyal yang sudah selesai dilacak.\n"
        "   - **Cara Pakai:** Kirim perintah `/stats`.\n\n"
        "**7. Pengaturan (`/set` & `/settings`)**\n"
        "   - **Kegunaan:** Mengubah perilaku bot secara real-time. `/settings` untuk melihat, `/set parameter nilai` untuk mengubah.\n"
        "   - **Cara Pakai:** Contoh: `/set min_mtf_confidence 38`."
    )
    
    keyboard = [[InlineKeyboardButton("↩️ Kembali ke Menu Utama", callback_data='back_main')]]
    
    try:
        await query.edit_message_text(
            tutorial_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    except BadRequest as e:
        if "Message is not modified" in str(e):
            pass
        else:
            logger.error(f"Error di show_tutorial_menu: {e}")

async def show_portfolio_menu(query, context):
    """Menampilkan ringkasan portfolio dinamis dengan P&L real-time."""
    
    stats = trading_engine.database.get_portfolio_stats()
    open_positions_count = stats.get('open_positions', 0)
    
    # Ambil detail posisi aktif
    detail_posisi_aktif = ""
    if open_positions_count > 0:
        tracked_signals = trading_engine.database.get_tracked_signals()
        detail_posisi_aktif += "\n**DETAIL POSISI AKTIF:**\n"
        for signal in tracked_signals:
            symbol = signal.get('symbol', 'N/A')
            signal_type = signal.get('signal_type', 'N/A')
            timestamp = datetime.strptime(signal.get('timestamp'), '%Y-%m-%d %H:%M:%S.%f').strftime('%d %b, %H:%M')
            detail_posisi_aktif += f"• **{symbol}** ({signal_type}) - Dilacak sejak {timestamp}\n"

    # --- PANGGIL KALKULATOR P&L KITA ---
    pnl_data = trading_engine.calculate_portfolio_pnl()
    total_pnl = pnl_data.get('total_pnl', 0)
    pnl_percent = pnl_data.get('pnl_percent', 0)
    # Tentukan emoji berdasarkan P&L
    pnl_emoji = "✅" if total_pnl >= 0 else "😥"

    portfolio_msg = (
        "📊 **RINGKASAN PORTFOLIO** 📊\n"
        f"{'='*30}\n\n"
        "**Status Portfolio (Simulasi):**\n"
        "• Total Nilai: $100,000\n"
        f"• Posisi Aktif Dilacak: **{open_positions_count}**\n"
        f"{detail_posisi_aktif}\n"
        f"**Performa Saat Ini (Real-Time):**\n"
        f"• P&L Terbuka: **${total_pnl:,.2f} ({pnl_percent:+.2f}%)** {pnl_emoji}\n\n" # <-- DATA P&L NYATA
        "Gunakan `/stats` untuk melihat performa sinyal yang sudah selesai."
    )
    
    keyboard = [[
        InlineKeyboardButton("📊 Tampilkan /stats", callback_data='show_stats'),
        InlineKeyboardButton("↩️ Kembali", callback_data='back_main')
    ]]
    
    await query.edit_message_text(
        portfolio_msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )
    
async def show_sentiment_report(query, context):
    """Menampilkan laporan sentimen pasar."""
    await query.edit_message_text("Menganalisis sentimen pasar global, harap tunggu...")
    report = trading_engine.generate_sentiment_report()
    
    keyboard = [[InlineKeyboardButton("↩️ Kembali ke Menu Utama", callback_data='back_main')]]
    
    await query.edit_message_text(
        report,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown',
        disable_web_page_preview=True # Agar link tidak menampilkan preview besar
    )

async def send_trading_results(context, chat_id: int, results: List[Dict], style: str):
    """Send trading results to user"""
    
    if not results:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ No qualified setups found"
        )
        return
    
    # Style configuration
    style_config = {
        'SCALPING': {
            'emoji': '⚡',
            'name': 'SCALPING',
            'hold_time': '1-15 minutes'
        },
        'DAY_TRADING': {
            'emoji': '📈',
            'name': 'DAY TRADING',
            'hold_time': '30min-6hours'
        },
        'SWING': {
            'emoji': '📊',
            'name': 'SWING TRADING',
            'hold_time': '1-10 days'
        }
    }
    
    config = style_config.get(style, style_config['DAY_TRADING'])
    
    # Header message
    header_msg = (
        f"{config['emoji']} **{config['name']} ANALYSIS**\n"
        f"{'='*30}\n"
        f"🎯 Qualified Setups: {len(results)}\n"
        f"⏰ Hold Time: {config['hold_time']}\n"
        f"📊 Professional Filtering Applied\n"
        f"🔬 Analysis Time: {datetime.now().strftime('%H:%M:%S UTC')}"
    )
    
    await context.bot.send_message(chat_id=chat_id, text=header_msg, parse_mode='Markdown')
    await asyncio.sleep(1)
    
    # Send each setup
    for i, result in enumerate(results[:3], 1):  # Limit to top 3
        try:
            signal_id = trading_engine.database.save_signal(result)
            risk_mgmt = result.get('risk_mgmt', {})
            market_data = result.get('market_data', {})
            
            setup_msg = (
                f"{config['emoji']} **{config['name']} #{i}: {result['symbol']}**\n"
                f"{'='*25}\n"
                f"📊 **Signal:** {result['signal_type']} ({result['confidence']:.1f}%)\n"
                f"💰 **Entry:** ${result['current_price']:.4f}\n"
                f"🎯 **Target 1:** ${risk_mgmt.get('take_profit_1', 0):.4f} (R:R {risk_mgmt.get('risk_reward_1', 0):.1f})\n"
                f"🎯 **Target 2:** ${risk_mgmt.get('take_profit_2', 0):.4f} (R:R {risk_mgmt.get('risk_reward_2', 0):.1f})\n"
                f"• **Target 3:** ${risk_mgmt.get('take_profit_3', 0):,.4f} (R:R {risk_mgmt.get('risk_reward_3', 0):.1f})\n"
                f"🛑 **Stop Loss:** ${risk_mgmt.get('stop_loss', 0):.4f}\n\n"
                f"📈 **Market Data:**\n"
                f"• 24h Volume: ${market_data.get('volume_24h', 0):,.0f}\n"
                f"• Volume Ratio: {market_data.get('volume_ratio', 1):.1f}x\n"
                f"• Price Change 24h: {market_data.get('price_change_24h', 0):+.1f}%\n"
                f"• Volatility: {risk_mgmt.get('volatility_ratio', 0):.1%}\n\n"
                f"⚠️ **Risk:** {risk_mgmt.get('stop_distance_percent', 0):.1f}% | Confidence: {result.get('confidence', 0):.1f}%"
            )
            
            keyboard = [[
                InlineKeyboardButton("👍 Lacak Trade Ini", callback_data=f"track_{signal_id}")
            ]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            # LANGKAH D: Kirim pesan dengan tombol
            await context.bot.send_message(
                chat_id=chat_id, 
                text=setup_msg, 
                parse_mode='Markdown',
                reply_markup=reply_markup # Tambahkan tombolnya di sini
            )
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error sending setup {i}: {e}")
            continue

async def send_professional_results(context, chat_id: int, results: List[Dict]):
    """Send professional analysis results"""
    
    if not results:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ No professional setups found"
        )
        return
    
    # Header
    header_msg = (
        f"🏛️ **PROFESSIONAL MARKET ANALYSIS**\n"
        f"{'='*35}\n"
        f"🎯 Qualified Setups: {len(results)}\n"
        f"⏰ Analysis Time: {datetime.now().strftime('%H:%M:%S UTC')}\n"
        f"🔬 Institutional-Grade Filtering\n"
        f"⚖️ Risk-Optimized Portfolio Allocation"
    )
    
    await context.bot.send_message(chat_id=chat_id, text=header_msg, parse_mode='Markdown')
    await asyncio.sleep(2)
    
    # Send each setup
    for i, result in enumerate(results, 1):
        try:
            risk_mgmt = result.get('risk_mgmt', {})
            market_data = result.get('market_data', {})
            components = result.get('signal_components', {})
            
            setup_msg = (
                f"🎯 **PROFESSIONAL SETUP #{i}: {result['symbol']}**\n"
                f"{'='*30}\n"
                f"🔥 **Signal:** {result['signal_type']} | Confidence: {result['confidence']:.1f}%\n"
                f"💰 **Entry:** ${result['current_price']:.4f}\n\n"
                f"🎯 **TARGETS & RISK:**\n"
                f"• TP1: ${risk_mgmt.get('take_profit_1', 0):.4f} (R:R {risk_mgmt.get('risk_reward_1', 0):.1f})\n"
                f"• TP2: ${risk_mgmt.get('take_profit_2', 0):.4f} (R:R {risk_mgmt.get('risk_reward_2', 0):.1f})\n"
                f"• TP3: ${risk_mgmt.get('take_profit_3', 0):.4f} (R:R {risk_mgmt.get('risk_reward_3', 0):.1f})\n"
                f"• SL: ${risk_mgmt.get('stop_loss', 0):.4f} ({risk_mgmt.get('stop_distance_percent', 0):.1f}% risk)\n\n"
                f"📊 **SIGNAL COMPONENTS:**\n"
                f"• Technical: {components.get('technical', 0):.0f}/25\n"
                f"• Momentum: {components.get('momentum', 0):.0f}/20\n"
                f"• Volume: {components.get('volume', 0):.0f}/15\n"
                f"• Sentiment: {components.get('sentiment', 0):.0f}/10\n\n"
                f"💹 **Market Metrics:**\n"
                f"• Final Score: {result.get('final_score', 0):.1f}/100\n"
                f"• Analisis Kunci: {get_fib_status(result.get('tf_analysis', {}).get(self.get_primary_timeframe_safe(result.get('tf_analysis')), {}).get('indicators', {}))}\n" # BARIS BARU
                f"• Signal Summary: {get_enhanced_signal_summary(result)}\n"
                f"• 24h Volume: ${market_data.get('volume_24h', 0):,.0f}\n"
                f"• Volume Quality: {market_data.get('volume_ratio', 1):.1f}x average\n"
                f"• Volatility: {risk_mgmt.get('volatility_ratio', 0):.1%}\n\n"
                f"🎯 **Fibonacci Analysis:**\n"
                f"• Status: {get_fib_status(primary_data.get('indicators', {}))}\n"
                f"⚠️ **Risk:** {risk_mgmt.get('stop_distance_percent', 0):.1f}% | Confidence: {result.get('confidence', 0):.1f}%"
            )
            
            await context.bot.send_message(chat_id=chat_id, text=setup_msg, parse_mode='Markdown')
            await asyncio.sleep(3)
            
        except Exception as e:
            logger.error(f"Error sending professional setup {i}: {e}")
            continue
    
    # Risk disclaimer
    disclaimer_msg = (
        f"⚠️ **PROFESSIONAL DISCLAIMER**\n"
        f"{'='*25}\n"
        f"• This is algorithmic analysis, not financial advice\n"
        f"• Professional risk management is mandatory\n"
        f"• Never risk more than 1-2% per trade\n"
        f"• Market conditions can change rapidly\n\n"
        f"🎯 **Trade Responsibly & Use Proper Position Sizing**"
    )
    
    await context.bot.send_message(chat_id=chat_id, text=disclaimer_msg, parse_mode='Markdown')

async def load_markets_with_retry(exchange, max_retries: int = 3):
    """Load markets with retry logic"""
    for attempt in range(max_retries):
        try:
            exchange.load_markets()
            logger.info(f"Markets loaded successfully on attempt {attempt + 1}")
            return True
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed to load markets: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
            else:
                raise Exception(f"Failed to load markets after {max_retries} attempts")
    return False
    
def get_fib_status(indicators: Dict) -> str:
    """Get fibonacci status for display"""
    try:
        fib_data = indicators.get('fibonacci', {})
        if not fib_data:
            return "📊 No Fib Data"
        
        at_fib_level = fib_data.get('at_fib_level', False)
        near_fib_level = fib_data.get('near_fib_level', False)
        confluence_count = fib_data.get('confluence_count', 0)
        distance_pct = fib_data.get('distance_pct', 100)
        
        if at_fib_level:
            if confluence_count > 1:
                return f"🎯 At Fib Confluence x{confluence_count} ({distance_pct:.2f}%)"
            else:
                nearest_info = fib_data.get('nearest_level_info', {})
                level_label = nearest_info.get('label', 'Unknown')
                return f"🎯 At Fib {level_label} ({distance_pct:.2f}%)"
                
        elif near_fib_level:
            if confluence_count > 1:
                return f"📍 Near Fib Confluence x{confluence_count} ({distance_pct:.1f}%)"
            else:
                nearest_info = fib_data.get('nearest_level_info', {})
                level_label = nearest_info.get('label', 'Unknown') 
                return f"📍 Near Fib {level_label} ({distance_pct:.1f}%)"
        else:
            return f"📊 Fib Normal ({distance_pct:.1f}%)"
            
    except Exception as e:
        logger.debug(f"Fib status error: {e}")
        return "📊 Fib Error"

def get_enhanced_signal_summary(result: Dict) -> str:
    """Get enhanced signal summary with all new features"""
    try:
        components = result.get('signal_components', {})
        
        # Core metrics
        summary_parts = []
        summary_parts.append(f"T:{components.get('technical', 0):.0f}")
        summary_parts.append(f"M:{components.get('momentum', 0):.0f}")
        summary_parts.append(f"V:{components.get('volume', 0):.0f}")
        summary_parts.append(f"S:{components.get('sentiment', 0):.0f}")
        
        # Enhanced features
        if components.get('structure', 0) > 0:
            summary_parts.append(f"St:{components.get('structure', 0):.0f}")
        
        if components.get('ml_enhanced', False):
            summary_parts.append("ML✓")
        
        return " | ".join(summary_parts)
        
    except Exception as e:
        return "Summary Error"

def format_fibonacci_levels(fibonacci_data: Dict, current_price: float) -> str:
    """Format fibonacci levels for display"""
    try:
        if not fibonacci_data:
            return "No Fibonacci data available"
        
        lines = []
        
        # Key levels near current price
        retracements = fibonacci_data.get('retracements', {})
        extensions = fibonacci_data.get('extensions', {})
        
        # Find closest levels
        close_levels = []
        
        for level_key, level_data in retracements.items():
            price = level_data['price']
            distance = abs(price - current_price) / current_price
            if distance < 0.05:  # Within 5%
                close_levels.append({
                    'price': price,
                    'label': level_data['label'],
                    'type': 'Retracement',
                    'distance': distance
                })
        
        for level_key, level_data in extensions.items():
            price = level_data['price']
            distance = abs(price - current_price) / current_price
            if distance < 0.05:  # Within 5%
                close_levels.append({
                    'price': price,
                    'label': level_data['label'], 
                    'type': 'Extension',
                    'distance': distance
                })
        
        # Sort by distance
        close_levels.sort(key=lambda x: x['distance'])
        
        if close_levels:
            lines.append("🎯 **Key Fibonacci Levels:**")
            for level in close_levels[:3]:  # Show top 3
                direction = "↑" if level['price'] > current_price else "↓"
                lines.append(f"• {direction} {level['label']} {level['type']}: ${level['price']:.4f} ({level['distance']:.1%})")
        
        # Confluence info
        confluence = fibonacci_data.get('confluence', {})
        if confluence.get('total_zones', 0) > 0:
            lines.append(f"\n🔥 **Confluence Zones:** {confluence['total_zones']}")
        
        return "\n".join(lines) if lines else "No significant levels nearby"
        
    except Exception as e:
        return f"Fibonacci format error: {e}"

# ===============================
# MAIN APPLICATION
# ===============================

def main():
    """Main application entry point with enhanced features"""
    try:
        # Display startup banner
        print("🚀" + "="*50 + "🚀")
        print("    CRYPTO TRADING BOT v2.1 - ENHANCED EDITION")
        print("🚀" + "="*50 + "🚀")
        
        # Check available features
        features_status = []
        features_status.append(f"✅ Core Analysis Engine")
        features_status.append(f"✅ Fibonacci Retracements & Extensions")
        features_status.append(f"{'✅' if WEBSOCKET_AVAILABLE else '❌'} Real-time WebSocket Data")
        features_status.append(f"{'✅' if DATABASE_AVAILABLE else '❌'} Database Integration") 
        features_status.append(f"{'✅' if ML_AVAILABLE else '❌'} ML Signal Enhancement")
        
        print("\n📊 FEATURE STATUS:")
        for status in features_status:
            print(f"   {status}")
        
        # Initialize enhanced trading engine
        print(f"\n⚙️ Initializing Enhanced Trading Engine...")
        global trading_engine
        trading_config = TradingConfig()
        trading_engine = TradingEngine(trading_config)
        
        print(f"   • Portfolio Value: ${trading_engine.portfolio_value:,}")
        print(f"   • Min Confidence: {trading_config.min_mtf_confidence}%")
        print(f"   • Min Volume: ${trading_config.min_volume_threshold:,}")
        
        # Test fibonacci on startup
        try:
            print(f"\n🧪 Testing Fibonacci Implementation...")
            test_data = pd.DataFrame({
                'high': [100, 102, 101, 103, 105, 104, 106, 108],
                'low': [98, 99, 98, 100, 102, 101, 103, 105],
                'close': [99, 101, 100, 102, 104, 103, 105, 107],
                'volume': [1000] * 8
            })
            
            fib_result = AdvancedIndicators.calculate_fibonacci_levels(test_data)
            if fib_result:
                print(f"   ✅ Fibonacci calculation working")
                print(f"   • Trend: {fib_result.get('trend_direction', 'unknown')}")
                print(f"   • Levels: {len(fib_result.get('retracements', {}))}")
            else:
                print(f"   ⚠️ Fibonacci calculation returned empty (normal for small dataset)")
                
        except Exception as e:
            print(f"   ⚠️ Fibonacci test failed: {e}")
        
        # Create application
        application = Application.builder().token(bot_token).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CallbackQueryHandler(button_handler))
        
        # Enhanced error handler
        async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            logger.error(f"Update {update} caused error {context.error}")
            
            # Try to send error message to user
            try:
                if update.effective_chat:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=f"❌ An error occurred. Please try again.\nError: {str(context.error)[:100]}...",
                        reply_markup=create_main_menu()
                    )
            except:
                pass
        
        application.add_error_handler(error_handler)
        
        logger.info("Bot started successfully with enhanced features")
        
        print(f"\n🎯 TRADING STYLES AVAILABLE:")
        print(f"   ⚡ Scalping (1-15min) - Ultra-fast profits")
        print(f"   📈 Day Trading (30min-6h) - Intraday momentum") 
        print(f"   📊 Swing Trading (1-10 days) - Multi-day trends")
        print(f"   🏛️ Professional Mode - Advanced analysis")
        
        print(f"\n🎯 NEW FEATURES IN v2.1:")
        print(f"   🎯 Fibonacci Retracements & Extensions")
        print(f"   🔥 Fibonacci Confluence Detection")
        print(f"   📊 Enhanced Signal Scoring")
        print(f"   ⚖️ Improved Risk Management")
        print(f"   🧠 ML-Ready Architecture")
        print(f"   💾 Database Integration")
        
        print(f"\n✨ Bot is running! Send /start in Telegram")
        print("Press Ctrl+C to stop")
        print("="*60)
        
        # Run the bot
        application.run_polling()
        
    except KeyboardInterrupt:
        print(f"\n👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"Critical error starting bot: {e}")
        print(f"❌ Failed to start bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    os.system('clear')
    main()
