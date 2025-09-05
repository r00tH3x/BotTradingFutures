# BUAT FILE BARU: websocket_manager.py
import websocket
import json
import threading
from collections import defaultdict
import logging

class BinanceWebSocketManager:
    def __init__(self):
        self.connections = {}
        self.data_buffer = defaultdict(list)
        self.callbacks = defaultdict(list)
        
    def subscribe_ticker(self, symbol: str, callback):
        """Subscribe to real-time ticker updates"""
        stream_name = f"{symbol.lower().replace('/', '')}@ticker"
        
        def on_message(ws, message):
            data = json.loads(message)
            self.data_buffer[symbol].append(data)
            for cb in self.callbacks[symbol]:
                cb(data)
        
        def on_error(ws, error):
            logging.error(f"WebSocket error for {symbol}: {error}")
            
        ws_url = f"wss://stream.binance.com:9443/ws/{stream_name}"
        ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error)
        
        self.connections[symbol] = ws
        self.callbacks[symbol].append(callback)
        
        # Start in separate thread
        threading.Thread(target=ws.run_forever, daemon=True).start()
