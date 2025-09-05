# BUAT FILE BARU: database.py
import sqlite3
import json
from datetime import datetime
import pandas as pd

class TradingDatabase:
    def __init__(self, db_path="trading_bot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                symbol TEXT,
                signal_type TEXT,
                confidence REAL,
                entry_price REAL,
                targets TEXT,
                stop_loss REAL,
                status TEXT DEFAULT 'ACTIVE'
            )
        ''')
        
        # Performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY,
                signal_id INTEGER,
                timestamp DATETIME,
                pnl REAL,
                pnl_pct REAL,
                status TEXT,
                FOREIGN KEY (signal_id) REFERENCES signals (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_signal(self, signal_data: dict):
        """Save trading signal to database"""
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
        
        conn.commit()
        conn.close()
