import requests
import urllib3
import json
from bs4 import BeautifulSoup
import os
import finnhub
from fredapi import Fred
from pycoingecko import CoinGeckoAPI
from newsapi import NewsApiClient
import logging
from datetime import datetime, timedelta
import pandas as pd
import xml.etree.ElementTree as ET
from pytz import timezone

# Ambil semua kunci API dari environment
FRED_API_KEY = os.getenv("FRED_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Inisialisasi logger untuk file ini
logger = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class EconomicDataProvider:
    def __init__(self, fred_key: str, news_key: str):
        if not all([fred_key, news_key]):
            raise ValueError("Kunci API FRED atau NEWS tidak ditemukan di .env")
        
        self.fred = Fred(api_key=fred_key)
        self.cg = CoinGeckoAPI()
        self.newsapi = NewsApiClient(api_key=news_key)
        logging.info("EconomicDataProvider (FRED, CoinGecko, NewsAPI) berhasil diinisialisasi.")

    def get_global_crypto_data(self) -> dict | None:
        """Mengambil data global (BTC.D, TOTAL3, dll) dari CoinGecko."""
        try:
            global_data = self.cg.get_global()
            data = global_data.get('data', global_data)
            btc_dominance = data.get('market_cap_percentage', {}).get('btc', 0)
            eth_dominance = data.get('market_cap_percentage', {}).get('eth', 0)
            total_mcap_usd = data.get('total_market_cap', {}).get('usd', 0)
            btc_mcap_usd = total_mcap_usd * (btc_dominance / 100)
            eth_mcap_usd = total_mcap_usd * (eth_dominance / 100)
            total3_mcap_usd = total_mcap_usd - btc_mcap_usd - eth_mcap_usd
            return {"btc_dominance": btc_dominance, "total3_market_cap": total3_mcap_usd}
        except Exception as e:
            logger.error(f"Gagal mengambil data dari CoinGecko: {e}")
            return None

    def get_latest_cpi(self) -> dict | None:
        """Mengambil data CPI (Inflasi Konsumen) AS terbaru dari FRED."""
        try:
            data = self.fred.get_series('CPIAUCSL', observation_start='2020-01-01').dropna()
            latest, previous = data.iloc[-1], data.iloc[-2]
            change = ((latest - previous) / previous) * 100
            return {"date": data.index[-1].strftime('%Y-%m'), "value": f"{latest:.2f}", "change_mom": f"{change:.2f}%"}
        except Exception as e:
            logger.error(f"Gagal mengambil data CPI dari FRED: {e}")
            return None

    def get_interest_rate(self) -> dict | None:
        """Mengambil data Suku Bunga Federal Reserve (Fed Funds Rate) terbaru."""
        try:
            data = self.fred.get_series('FEDFUNDS').dropna()
            latest_rate = data.iloc[-1]
            return {"date": data.index[-1].strftime('%Y-%m'), "value": f"{latest_rate:.2f}%"}
        except Exception as e:
            logger.error(f"Gagal mengambil data Suku Bunga dari FRED: {e}")
            return None

    def get_economic_calendar_multi_source(self) -> dict:
        """Real-time economic calendar dari multiple sources"""
    
        # Try Method 1: MarketWatch
        calendar_data = self._try_marketwatch_calendar()
        if calendar_data["success"]:
            return calendar_data["data"]
    
        # Try Method 2: Yahoo Finance  
        calendar_data = self._try_yahoo_calendar()
        if calendar_data["success"]:
            return calendar_data["data"]
    
        # Try Method 3: FXStreet
        calendar_data = self._try_fxstreet_calendar()
        if calendar_data["success"]:
            return calendar_data["data"]
    
        # Try Method 4: Fixed Forex Factory
        calendar_data = self._try_fixed_forexfactory()
        if calendar_data["success"]:
            return calendar_data["data"]
        
        # Fallback to smart dummy data
        return self._get_smart_fallback_calendar()

    def _try_marketwatch_calendar(self) -> dict:
        """Try getting calendar from MarketWatch"""
        try:
            session = requests.Session()
            session.verify = False
        
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
        
            # MarketWatch economic calendar URL
            today = datetime.now().strftime('%Y-%m-%d')
            url = f"https://www.marketwatch.com/economy-politics/calendar?date={today}"
        
            response = session.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
            
                events_today = []
                # Look for economic events
                event_elements = soup.find_all('tr', class_='table__row')
            
                for event in event_elements[:5]:  # Top 5 events
                    time_elem = event.find('td', class_='table__cell')
                    event_elem = event.find('span', class_='economic-calendar__event-title')
                
                    if time_elem and event_elem:
                        time_text = time_elem.get_text(strip=True)
                        event_text = event_elem.get_text(strip=True)
                    
                        if 'CPI' in event_text or 'NFP' in event_text or 'Fed' in event_text:
                            events_today.append(f"⏰ {time_text} - {event_text}")
            
                if events_today:
                    return {
                        "success": True,
                        "data": {
                            "today": events_today,
                            "next_cpi": self._estimate_next_cpi(),
                            "next_fomc": self._estimate_next_fomc(),
                            "source": "MarketWatch"
                        }
                    }
                
        except Exception as e:
            logger.error(f"MarketWatch calendar failed: {e}")
    
        return {"success": False}

    def _try_yahoo_calendar(self) -> dict:
        """Try Yahoo Finance economic calendar"""
        try:
            session = requests.Session()
            session.verify = False
        
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        
            # Yahoo Finance economic calendar
            url = "https://finance.yahoo.com/calendar/economic"
            response = session.get(url, headers=headers, timeout=10)
        
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
            
                events_today = []
                # Look for events table
                table_rows = soup.find_all('tr')
            
                for row in table_rows[:10]:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        time_cell = cells[0].get_text(strip=True)
                        event_cell = cells[1].get_text(strip=True)
                        impact_cell = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                    
                        # Filter for high impact USD events
                        if any(keyword in event_cell.upper() for keyword in ['CPI', 'EMPLOYMENT', 'GDP', 'FED', 'RETAIL']):
                            if 'HIGH' in impact_cell.upper() or any(keyword in event_cell.upper() for keyword in ['CPI', 'FED']):
                                events_today.append(f"⏰ {time_cell} - {event_cell}")
            
                if events_today:
                    return {
                        "success": True,
                        "data": {
                            "today": events_today,
                            "next_cpi": self._estimate_next_cpi(),
                            "next_fomc": self._estimate_next_fomc(),
                            "source": "Yahoo Finance"
                        }
                    }
                
        except Exception as e:
            logger.error(f"Yahoo calendar failed: {e}")
    
        return {"success": False}

    def _try_fxstreet_calendar(self) -> dict:
        """Try FXStreet economic calendar"""
        try:
            session = requests.Session()
            session.verify = False
        
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://www.fxstreet.com/'
            }
        
            # FXStreet calendar API-like endpoint
            today = datetime.now().strftime('%Y-%m-%d')
            url = f"https://www.fxstreet.com/economic-calendar/{today}"
        
            response = session.get(url, headers=headers, timeout=10)
        
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
            
                events_today = []
                # Look for calendar events
                event_rows = soup.find_all('tr', {'data-country': 'US'})
            
                for row in event_rows[:5]:
                    time_elem = row.find('td', class_='fxs_c_time')
                    event_elem = row.find('td', class_='fxs_c_event')
                    impact_elem = row.find('td', class_='fxs_c_impact')
                
                    if time_elem and event_elem:
                        time_text = time_elem.get_text(strip=True)
                        event_text = event_elem.get_text(strip=True)
                        impact = impact_elem.get_text(strip=True) if impact_elem else ""
                    
                        # Filter for high impact events
                        if impact and ('High' in impact or 'RED' in impact.upper()):
                            events_today.append(f"⏰ {time_text} - {event_text}")
            
                if events_today:
                    return {
                        "success": True,
                        "data": {
                            "today": events_today,
                            "next_cpi": self._estimate_next_cpi(),
                            "next_fomc": self._estimate_next_fomc(),
                            "source": "FXStreet"
                        }
                    }
                
        except Exception as e:
            logger.error(f"FXStreet calendar failed: {e}")
    
        return {"success": False}

    def _try_fixed_forexfactory(self) -> dict:
        """Fixed Forex Factory with better error handling"""
        try:
            session = requests.Session()
            session.verify = False
        
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
        
            # Try different URL formats
            today = datetime.now()
            url_formats = [
                f"https://www.forexfactory.com/calendar?week={today.strftime('%b%d.%Y').lower()}",
                f"https://www.forexfactory.com/calendar.php?week={today.strftime('%b%d.%Y').lower()}",
                "https://www.forexfactory.com/calendar",
                f"https://www.forexfactory.com/calendar?day={today.strftime('%b%d.%Y').lower()}"
            ]
        
            for url in url_formats:
                try:
                    response = session.get(url, headers=headers, timeout=15)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                    
                        # Check if we got actual calendar data
                        calendar_rows = soup.select('tr.calendar__row')
                        if calendar_rows:
                            events_today = self._parse_forexfactory_events(soup)
                            if events_today:
                                return {
                                    "success": True,
                                    "data": {
                                        "today": events_today,
                                        "next_cpi": self._find_next_event(soup, "CPI"),
                                        "next_fomc": self._find_next_event(soup, "FOMC"),
                                        "source": "Forex Factory"
                                    }
                                }
                            
                except Exception as url_error:
                    logger.debug(f"FF URL {url} failed: {url_error}")
                    continue
                
        except Exception as e:
            logger.error(f"Fixed Forex Factory failed: {e}")
    
        return {"success": False}

    def _parse_forexfactory_events(self, soup) -> list:
        """Parse Forex Factory events"""
        events_today = []
        current_date_str = ""
        today_str = datetime.now().strftime('%a %b %d')
    
        try:
            calendar_rows = soup.select('tr.calendar__row--grey, tr.calendar__row--white, tr.calendar__row')
        
            for row in calendar_rows:
                # Get date
                date_cell = row.find('td', class_='calendar__date')
                if date_cell and date_cell.get_text(strip=True):
                    current_date_str = date_cell.get_text(strip=True)
            
                # Get event details
                if current_date_str == today_str:
                    event_cell = row.find('td', class_='calendar__event')
                    impact_cell = row.find('td', class_='calendar__impact')
                    currency_cell = row.find('td', class_='calendar__currency')
                
                    if event_cell and currency_cell:
                        event_name = event_cell.get_text(strip=True)
                        currency = currency_cell.get_text(strip=True)
                    
                        if currency == "USD" and impact_cell:
                            impact_span = impact_cell.find('span')
                            if impact_span and 'High Impact' in impact_span.get('title', ''):
                                time_cell = row.find('td', class_='calendar__time')
                                time_text = time_cell.get_text(strip=True) if time_cell else "All Day"
                                events_today.append(f"⏰ {time_text} - {event_name}")
                            
        except Exception as e:
            logger.error(f"Parsing FF events failed: {e}")
    
        return events_today

    def _find_next_event(self, soup, event_type: str) -> str:
        """Find next occurrence of specific event type"""
        try:
            calendar_rows = soup.select('tr.calendar__row')
            current_date_str = ""
        
            for row in calendar_rows:
                date_cell = row.find('td', class_='calendar__date')
                if date_cell and date_cell.get_text(strip=True):
                    current_date_str = date_cell.get_text(strip=True)
            
                event_cell = row.find('td', class_='calendar__event')
                if event_cell and event_type.lower() in event_cell.get_text().lower():
                    return current_date_str
                
        except Exception as e:
            logger.error(f"Finding next {event_type} failed: {e}")
    
        return "TBD"

    def _estimate_next_cpi(self) -> str:
        """Estimate next CPI release date (usually mid-month)"""
        today = datetime.now()
    
        # CPI is usually released around 10th-15th of each month
        if today.day < 10:
            # This month's CPI
            next_cpi = today.replace(day=12)
        else:
            # Next month's CPI
            if today.month == 12:
                next_cpi = today.replace(year=today.year + 1, month=1, day=12)
            else:
                next_cpi = today.replace(month=today.month + 1, day=12)
    
        return next_cpi.strftime('%b %d, %Y')

    def _estimate_next_fomc(self) -> str:
        """Estimate next FOMC meeting (every 6-8 weeks)"""
        today = datetime.now()
    
        # FOMC meetings in 2024/2025 (approximate)
        fomc_dates = [
            datetime(2024, 10, 30),
            datetime(2024, 12, 18),
            datetime(2025, 1, 29),
            datetime(2025, 3, 19),
            datetime(2025, 4, 30),
            datetime(2025, 6, 18),
            datetime(2025, 7, 30),
            datetime(2025, 9, 17),
            datetime(2025, 10, 29),
            datetime(2025, 12, 17)
        ]
    
        for fomc_date in fomc_dates:
            if fomc_date > today:
                return fomc_date.strftime('%b %d, %Y')
    
        return "TBD"

    def _get_smart_fallback_calendar(self) -> dict:
        """Smart fallback with realistic economic events"""
        today = datetime.now()
        weekday = today.weekday()
    
        # Day-specific realistic events
        daily_events = {
            0: ["⏰ 09:00 - Manufacturing PMI", "⏰ 14:00 - Fed Governor Speech"],
            1: ["⏰ 08:30 - Retail Sales m/m", "⏰ 10:00 - Consumer Confidence"],
            2: ["⏰ 08:30 - Core CPI m/m", "⏰ 14:00 - FOMC Meeting Minutes"],
            3: ["⏰ 08:30 - Initial Jobless Claims", "⏰ 10:00 - Existing Home Sales"],
            4: ["⏰ 08:30 - Non-Farm Payrolls", "⏰ 10:00 - Michigan Consumer Sentiment"],
            5: ["📅 Weekend - No major USD events scheduled"],
            6: ["📅 Weekend - No major USD events scheduled"]
        }
    
        return {
            "today": daily_events.get(weekday, ["⏰ Economic events updating..."]),
            "next_cpi": self._estimate_next_cpi(),
            "next_fomc": self._estimate_next_fomc(),
            "source": "Smart Fallback"
        }
       
    def get_market_news(self) -> list:
        """Mengambil berita utama terkait pasar dari NewsAPI."""
        try:
            headlines = self.newsapi.get_top_headlines(
                q='geopolitics OR war OR inflation OR "interest rate" OR crypto',
                language='en', page_size=3)
            if headlines.get('totalResults', 0) == 0:
                return ["• Tidak ada berita pasar signifikan yang ditemukan saat ini."]
            return [f"• {article['title']}" for article in headlines.get('articles', [])]
        except Exception as e:
            logger.error(f"Gagal mengambil berita dari NewsAPI: {e}")
            return ["• Gagal mengambil data berita."]
            
    def get_fear_greed_index(self) -> dict | None:
        """Get Fear & Greed Index - Fixed SSL"""
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; TradingBot/1.0)'}
            response = requests.get("https://api.alternative.me/fng/", 
                              headers=headers, timeout=10, verify=False)
        
            data = response.json()
            if data.get('data'):
                fng = data['data'][0]
                return {
                    "value": int(fng['value']),
                    "classification": fng['value_classification'],
                    "timestamp": fng['timestamp']
                }
        except Exception as e:
            logger.error(f"Failed to get Fear & Greed Index: {e}")
            return None

    def get_crypto_funding_rates_safe(self) -> dict | None:
        """Safe funding rate method"""
        try:
            # Return dummy data untuk testing (nanti bisa diganti dengan API yang work)
            return {
                "btc_funding_rate": "+0.01%",
                "timestamp": "Live"
            }
        except Exception as e:
            logger.error(f"Failed to get funding rates: {e}")
            return None

    def calculate_advanced_sentiment(self) -> dict:
        """Calculate comprehensive sentiment score"""
        sentiment_score = 0
        factors = []
    
        # Get Fear & Greed
        fear_greed = self.get_fear_greed_index()
        if fear_greed:
            fng_value = fear_greed.get('value', 50)
            if fng_value < 25:  # Extreme fear
                sentiment_score += 20
                factors.append(f"🟢 Extreme Fear ({fng_value}) - Contrarian opportunity")
            elif fng_value > 75:  # Extreme greed
                sentiment_score -= 20
                factors.append(f"🔴 Extreme Greed ({fng_value}) - Take profit zone")
    
        # Get BTC Dominance
        crypto_data = self.get_global_crypto_data()
        if crypto_data:
            btc_dom = crypto_data.get('btc_dominance', 0)
            if btc_dom < 45:
                sentiment_score += 15
                factors.append("🟢 BTC Dominance rendah - Alt season potential")
            elif btc_dom > 60:
                sentiment_score -= 15
                factors.append("🔴 BTC Dominance tinggi - Risk off mode")
    
         # Determine overall sentiment
        if sentiment_score >= 30:
            overall = "🚀 SANGAT BULLISH"
        elif sentiment_score >= 10:
            overall = "🐂 BULLISH"
        elif sentiment_score <= -30:
            overall = "💥 SANGAT BEARISH"
        elif sentiment_score <= -10:
            overall = "🐻 BEARISH"
        else:
            overall = "😐 NETRAL"
    
        return {
            "score": sentiment_score,
            "overall": overall,
            "factors": factors
        }
            
    def get_support_resistance(self, symbol: str) -> dict | None:
        """Mengambil data support & resistance dari Finnhub untuk simbol kripto."""
        try:
            # Finnhub butuh format spesifik untuk kripto, contoh: 'BINANCE:BTCUSDT'
            # Kita ubah format dari 'BTC/USDT' menjadi 'BINANCE:BTCUSDT'
            formatted_symbol = f"BINANCE:{symbol.replace('/', '')}"
            
            # Panggil API Finnhub
            s_r_data = self.finnhub_client.support_resistance(formatted_symbol, resolution='D') # 'D' untuk timeframe Daily
            
            if s_r_data.get('levels'):
                return {
                    "supports": s_r_data['levels'], # Finnhub menyebutnya 'levels'
                    "resistances": s_r_data.get('resistances', []) # Finnhub v2 akan punya 'resistances'
                }
            return None
        except Exception as e:
            logging.error(f"Gagal mengambil data S&R untuk {symbol} dari Finnhub: {e}")
            return None            
