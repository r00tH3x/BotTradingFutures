# chart_generator.py

import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
from .app import binance
import logging

def generate_chart(symbol: str, analysis_result: dict) -> str | None:
    """
    Menghasilkan gambar chart lengkap dengan analisis dan menyimpannya sebagai file.
    """
    try:
        logging.info(f"Membuat chart untuk {symbol}...")
        # 1. Ambil data historis (100 bar untuk perhitungan indikator yang akurat)
        ohlcv = binance.fetch_ohlcv(symbol, timeframe='1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # 2. Hitung semua indikator pada data penuh (100 bar)
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)

        # --- PERUBAHAN UTAMA DIMULAI DI SINI ---
        
        # 3. Potong DataFrame menjadi 60 bar TERAKHIR untuk digambar
        df_to_plot = df.tail(60)

        # 4. Ambil data analisis dari hasil yang sudah kita hitung
        risk_mgmt = analysis_result.get('risk_mgmt', {})
        entry_zone = analysis_result.get('entry_zone_raw', [0, 0])
        stop_loss = risk_mgmt.get('stop_loss')
        tp1 = risk_mgmt.get('take_profit_1')
        tp2 = risk_mgmt.get('take_profit_2')
        tp3 = risk_mgmt.get('take_profit_3')
        
        volume_profile = analysis_result.get('tf_analysis', {}).get('1h', {}).get('indicators', {}).get('volume_profile', {})
        poc = volume_profile.get('poc_price')

        # 5. Siapkan plot tambahan DARI DATA YANG SUDAH DIPOTONG
        add_plots = [
            mpf.make_addplot(df_to_plot['EMA_20'], color='blue', width=0.7),
            mpf.make_addplot(df_to_plot['EMA_50'], color='orange', width=0.7),
        ]

        hlines_data = {'hlines': [], 'colors': [], 'linestyle': []}
        if poc:
            hlines_data['hlines'].append(poc)
            hlines_data['colors'].append('cyan')
            hlines_data['linestyle'].append('--')
        
        # 6. Gambar chart-nya!
        chart_style = 'yahoo'
        market_colors = mpf.make_marketcolors(up='green', down='red', inherit=True)
        style = mpf.make_mpf_style(base_mpf_style=chart_style, marketcolors=market_colors)
        
        fill_between_plot = None
        if entry_zone and entry_zone[0] > 0 and entry_zone[1] > 0:
            fill_between_plot = dict(y1=entry_zone[0], y2=entry_zone[1], alpha=0.2, color='green')

        fig, axes = mpf.plot(
            df_to_plot, # Gunakan data yang sudah dipotong
            type='candle', style=style,
            title=f"\nChart Analysis: {symbol} (1H)",
            ylabel='Price ($)', volume=True,
            addplot=add_plots, hlines=hlines_data,
            fill_between=fill_between_plot,
            figsize=(15, 8), returnfig=True
        )

        # 7. Tambahkan label teks
        ax = axes[0]
        # Gunakan 'len(df_to_plot)' sebagai referensi posisi x agar presisi
        text_x_position = len(df_to_plot) * 1.01 
        if all([tp1, tp2, tp3, stop_loss]):
            ax.text(text_x_position, tp1, f' TP1', color='green', va='center', fontsize=9)
            ax.text(text_x_position, tp2, f' TP2', color='green', va='center', fontsize=9)
            ax.text(text_x_position, tp3, f' TP3', color='green', va='center', fontsize=9)
            ax.text(text_x_position, stop_loss, f' SL', color='red', va='center', fontsize=9)
        if poc:
             ax.text(text_x_position, poc, f' POC', color='cyan', va='center', fontsize=9)

        # 8. Simpan chart
        chart_filename = f"chart_{symbol.replace('/', '_')}.png"
        fig.savefig(chart_filename, bbox_inches='tight')
        logging.info(f"Chart berhasil disimpan sebagai {chart_filename}")
        
        return chart_filename

    except Exception as e:
        logging.error(f"Gagal membuat chart untuk {symbol}: {e}", exc_info=True)
        return None
