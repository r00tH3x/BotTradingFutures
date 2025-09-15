import requests
from bs4 import BeautifulSoup
import os
import finnhub
from fredapi import Fred
from pycoingecko import CoinGeckoAPI
from newsapi import NewsApiClient
import logging
from datetime import datetime, timedelta
import pandas as pd

# Ambil semua kunci API dari environment
FRED_API_KEY = os.getenv("FRED_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Inisialisasi logger untuk file ini
logger = logging.getLogger(__name__)

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

    def get_economic_calendar(self) -> dict:
        """
        Scraping Forex Factory untuk minggu ini, mencari event High Impact hari ini
        DAN mencari tanggal event CPI & FOMC berikutnya. Mengembalikan dictionary.
        """
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            url = f"https://www.forexfactory.com/calendar?week={datetime.now().strftime('%b%d.%Y').lower()}"
            response = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            calendar_rows = soup.select('tr.calendar__row--grey, tr.calendar__row--white')
            
            events_today = []
            next_cpi_date = "Belum ditemukan"
            next_fomc_date = "Belum ditemukan"
            
            current_date_str = ""
            today_str = datetime.now().strftime('%a %b %d')

            for row in calendar_rows:
                date_cell = row.find('td', class_='calendar__date')
                if date_cell and date_cell.text.strip():
                    current_date_str = date_cell.text.strip()

                event_cell = row.find('td', class_='calendar__event')
                if event_cell:
                    event_name = event_cell.text.strip()
                    
                    if "CPI m/m" in event_name and next_cpi_date == "Belum ditemukan":
                        next_cpi_date = current_date_str
                    if "FOMC Statement" in event_name and next_fomc_date == "Belum ditemukan":
                        next_fomc_date = current_date_str

                    if current_date_str == today_str:
                        impact_cell = row.find('td', class_='calendar__impact')
                        currency_cell = row.find('td', class_='calendar__currency')
                        if impact_cell and currency_cell and "USD" in currency_cell.text:
                            impact_span = impact_cell.find('span')
                            impact_title = impact_span.get('title', '') if impact_span else ''
                            if 'High Impact Expected' in impact_title:
                                time_cell = row.find('td', class_='calendar__time')
                                time = time_cell.text.strip() if time_cell else "All Day"
                                events_today.append(f"⏰ {time} - {event_name}")

            return {
                "today": events_today if events_today else ["Tidak ada acara berdampak tinggi untuk USD hari ini."],
                "next_cpi": next_cpi_date,
                "next_fomc": next_fomc_date
            }

        except Exception as e:
            logger.error(f"Gagal scraping kalender ekonomi: {e}", exc_info=True)
            return {"today": ["Gagal mengambil data kalender."], "next_cpi": "Error", "next_fomc": "Error"}

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
