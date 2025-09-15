# BUAT FILE BARU: database.py
import sqlite3
import json
from datetime import datetime
import pandas as pd
import logging
logger = logging.getLogger(__name__)

class TradingDatabase:
    def __init__(self, db_path="trading_bot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables dengan struktur baru."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Hapus tabel lama jika ada untuk pembaruan (hanya untuk pengembangan)
        # cursor.execute('DROP TABLE IF EXISTS signals')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                signal_type TEXT,
                confidence REAL,
                entry_price REAL,
                targets TEXT,
                stop_loss REAL,
                status TEXT DEFAULT 'ACTIVE' -- Kolom baru: ACTIVE, TRACKED, CLOSED
            )
        ''')
        
        # ... tabel performance tetap sama ...
        conn.commit()
        conn.close()

    def save_signal(self, signal_data: dict) -> int:
        """Save trading signal to database dan KEMBALIKAN ID-nya."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals (timestamp, symbol, signal_type, confidence, 
                                 entry_price, targets, stop_loss)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            signal_data['symbol'],
            signal_data['signal_type'],
            signal_data['confidence'],
            signal_data['current_price'],
            json.dumps(signal_data.get('risk_mgmt', {})),
            signal_data.get('risk_mgmt', {}).get('stop_loss', 0)
        ))
        
        signal_id = cursor.lastrowid # Ambil ID dari baris yang baru saja dimasukkan
        conn.commit()
        conn.close()
        return signal_id # Kembalikan ID

    def track_signal(self, signal_id: int):
        """Update status sinyal menjadi 'TRACKED'."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE signals SET status = 'TRACKED' WHERE id = ?", (signal_id,))
        conn.commit()
        conn.close()
        
    def get_tracked_signals(self) -> list:
        """Mengambil semua sinyal dengan status 'TRACKED'."""
        conn = sqlite3.connect(self.db_path)
        # Menggunakan row_factory agar hasilnya seperti dictionary
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM signals WHERE status = 'TRACKED'")
        signals = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return signals

    def update_signal_status(self, signal_id: int, new_status: str):
        """Mengubah status sinyal (misal: menjadi CLOSED_WIN atau CLOSED_LOSS)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE signals SET status = ? WHERE id = ?", (new_status, signal_id))
        conn.commit()
        conn.close()
        logger.info(f"Status sinyal #{signal_id} diubah menjadi {new_status}")
        
    def get_performance_stats(self) -> dict:
        """Menghitung dan mengembalikan statistik performa dari sinyal yang sudah selesai."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Hitung total trade yang sudah selesai (statusnya diawali dengan 'CLOSED_')
        cursor.execute("SELECT COUNT(*) FROM signals WHERE status LIKE 'CLOSED_%'")
        total_closed = cursor.fetchone()[0]
        
        # Hitung total trade yang menang
        cursor.execute("SELECT COUNT(*) FROM signals WHERE status LIKE 'CLOSED_WIN_%'")
        total_wins = cursor.fetchone()[0]
        
        # Hitung total trade yang kalah
        cursor.execute("SELECT COUNT(*) FROM signals WHERE status = 'CLOSED_LOSS'")
        total_losses = cursor.fetchone()[0]
        
        conn.close()
        
        # Hitung win rate, hindari pembagian dengan nol
        win_rate = (total_wins / total_closed) * 100 if total_closed > 0 else 0
        
        return {
            "total_trades": total_closed,
            "wins": total_wins,
            "losses": total_losses,
            "win_rate": win_rate
        }
        
    def get_portfolio_stats(self) -> dict:
        """Menghitung statistik untuk ringkasan portfolio."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Hitung total posisi yang sedang aktif dilacak
        cursor.execute("SELECT COUNT(*) FROM signals WHERE status = 'TRACKED'")
        open_positions = cursor.fetchone()[0]
        
        # Hitung total posisi yang sudah selesai
        cursor.execute("SELECT COUNT(*) FROM signals WHERE status LIKE 'CLOSED_%'")
        closed_trades = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "open_positions": open_positions,
            "closed_trades": closed_trades
        }
        
    def get_all_signals_for_training(self) -> list:
        """Mengambil semua data sinyal yang sudah selesai untuk pelatihan ML."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Kita ambil sinyal yang statusnya sudah final (WIN atau LOSS)
        cursor.execute("SELECT * FROM signals WHERE status LIKE 'CLOSED_%'")
        signals = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return signals        
