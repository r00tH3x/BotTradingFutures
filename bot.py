#file bot.py
import os
import json
import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta

# --- Pastikan ini ada di paling atas ---
from dotenv import load_dotenv
load_dotenv()
# ----------------------------------------

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters

# Impor semua komponen dari package 'assistant'
from assistant.app import start, button_handler, trading_engine, binance

# Konfigurasi logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Membaca semua konfigurasi dari .env ---
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")
# -------------------------------------------

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log Errors caused by Updates."""
    logger.error(f"Update {update} caused error {context.error}", exc_info=context.error)

# --- SEMUA FUNGSI HANDLER ADA DI SINI ---

async def watch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Menambahkan simbol ke watchlist untuk user tertentu (versi pintar)."""
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Gunakan format: /watch SIMBOL (contoh: /watch BTC atau /watch BTC/USDT)")
        return
        
    symbol_input = " ".join(context.args).upper()
    
    if '/' not in symbol_input:
        symbol = f"{symbol_input}/USDT"
        await update.message.reply_text(f"📝 Format disesuaikan. Memantau {symbol}...")
    else:
        symbol = symbol_input
    
    trading_engine.watchlist[symbol].add(chat_id)
    await update.message.reply_text(f"✅ {symbol} telah ditambahkan ke watchlist Anda. Saya akan memberitahu Anda jika ada sinyal kuat!")

async def background_tasks(context: ContextTypes.DEFAULT_TYPE):
    """Menjalankan semua tugas latar belakang: Jurnal, Watchlist, dan Auto Scanner."""
    logger.info("Running background tasks...")
    
    try:
        admin_id = int(ADMIN_CHAT_ID)
    except (ValueError, TypeError):
        logger.error("ADMIN_CHAT_ID tidak valid. Notifikasi latar belakang tidak akan terkirim.")
        return
    
    # --- BAGIAN 1: Cek Watchlist untuk Sinyal Baru (Kode Lama) ---
    if trading_engine.watchlist:
        for symbol, chat_ids in list(trading_engine.watchlist.items()):
            if not chat_ids:
                continue
            try:
                logger.info(f"Checking watchlist for {symbol}...")
                # Kita gunakan manual_analysis_logic karena ia mengembalikan pesan yang sudah diformat
                result_message = trading_engine.manual_analysis_logic(symbol)
            
                if "Sinyal:" in result_message and "Confidence" in result_message:
                    try:
                        confidence_str = result_message.split('(')[1].split('%')[0]
                        confidence = float(confidence_str)
                    
                        if confidence > 70:
                            alert_message = f"🔥 **ALERT WATCHLIST** 🔥\n\n" + result_message
                            for chat_id in chat_ids:
                                await context.bot.send_message(chat_id=chat_id, text=alert_message, parse_mode='Markdown')
                        
                            # Hapus dari watchlist setelah alert terkirim agar tidak spam
                            trading_engine.watchlist[symbol].clear()

                    except Exception as e:
                        logger.error(f"Could not parse confidence or send alert for {symbol}: {e}")
            except Exception as e:
                logger.error(f"Failed to analyze watchlist symbol {symbol}: {e}")

    # --- BAGIAN 2: Jurnal Trade Otomatis (Logika Baru) ---
    tracked_signals = trading_engine.database.get_tracked_signals()
    if tracked_signals:
        logger.info(f"Memantau {len(tracked_signals)} trade yang sedang dilacak...")
        for signal in tracked_signals:
            try:
                symbol, signal_id, signal_type = signal['symbol'], signal['id'], signal['signal_type']
                ticker = binance.fetch_ticker(symbol)
                current_price = ticker['last']
                risk_mgmt = json.loads(signal['targets'])
                stop_loss = risk_mgmt.get('stop_loss', 0)
                take_profit_1 = risk_mgmt.get('take_profit_1', 0)

                if signal_type == 'BULLISH':
                    if current_price <= stop_loss:
                        trading_engine.database.update_signal_status(signal_id, 'CLOSED_LOSS')
                        await context.bot.send_message(chat_id=admin_id, text=f"😥 JURNAL TRADE 😥\nSinyal BULLISH untuk {symbol} telah menyentuh Stop Loss.")
                    elif current_price >= take_profit_1:
                        trading_engine.database.update_signal_status(signal_id, 'CLOSED_WIN_TP1')
                        await context.bot.send_message(chat_id=admin_id, text=f"🎉 JURNAL TRADE 🎉\nSinyal BULLISH untuk {symbol} telah mencapai Target Profit 1!")
            except Exception as e:
                logger.error(f"Error saat memproses jurnal untuk sinyal #{signal.get('id')}: {e}")
            
    # --- TUGAS 3: AUTO SCANNER ---
    if not trading_engine.auto_scanner_active:
        return # Jika scanner tidak aktif, hentikan tugas di sini

    logger.info("Auto Scanner is ACTIVE. Starting full market scan...")
    try:
        # Menjalankan pemindaian pasar penuh dengan setelan default (Day Trading)
        results = await trading_engine.scan_market_comprehensive(
            binance, headers, style_filter='DAY_TRADING'
        )
        
        if results:
            # Kirim sinyal teratas sebagai "Proposal"
            top_signal = results[0]
            
            # Cek apakah kita sudah pernah mengirim proposal ini
            symbol = top_signal['symbol']
            last_proposal_time = trading_engine.signal_cache.get(symbol)
            
            # Kirim proposal hanya jika belum pernah, atau sudah lebih dari 4 jam yang lalu
            if not last_proposal_time or (datetime.now() - last_proposal_time) > timedelta(hours=4):
                
                logger.info(f"Auto Scanner found a new high-quality signal for {symbol}. Sending proposal...")
                
                # Simpan waktu proposal ini dikirim
                trading_engine.signal_cache[symbol] = datetime.now()

                proposal_message = "🔥 **PROPOSAL TRADE BARU** 🔥\n" + trading_engine.format_single_signal_message(top_signal)
                
                # Siapkan tombol Lacak
                signal_id = trading_engine.database.save_signal(top_signal)
                keyboard = [[ InlineKeyboardButton("👍 Lacak Trade Ini", callback_data=f"track_{signal_id}") ]]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await context.bot.send_message(
                    chat_id=chat_id_to_notify,
                    text=proposal_message,
                    parse_mode='Markdown',
                    reply_markup=reply_markup
                )

    except Exception as e:
        logger.error(f"Error during Auto Scan task: {e}")

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Menangani pesan teks untuk analisis manual dan mengirim chart."""
    message_text = update.message.text.upper()

    if '/' in message_text and message_text.endswith("USDT"):
        symbol = message_text

        processing_message = await update.message.reply_text(f"🔬 Menganalisis ${symbol} dan membuat chart, harap tunggu...")

        analysis_data = trading_engine.manual_analysis_logic(symbol)

        response_message = analysis_data['message']
        chart_file = analysis_data['chart']

        # Hapus pesan "sedang diproses"
        await processing_message.delete()

        if chart_file:
            # Kirim foto dengan teks sebagai caption
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=open(chart_file, 'rb'),
                caption=response_message,
                parse_mode='Markdown'
            )
            # Hapus file chart setelah dikirim agar tidak menumpuk
            os.remove(chart_file)
        else:
            # Jika tidak ada chart, kirim teks saja
            await update.message.reply_text(response_message, parse_mode='Markdown')

async def set_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Mengatur parameter konfigurasi bot."""
    if len(context.args) != 2:
        await update.message.reply_text("Gunakan format: /set [parameter] [nilai]\nContoh: /set min_mtf_confidence 35")
        return
    key, value = context.args
    response_message = trading_engine.update_config(key, value)
    await update.message.reply_text(response_message, parse_mode='Markdown')

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Menampilkan pengaturan bot saat ini."""
    config = trading_engine.config
    config_vars = [attr for attr in dir(config) if not callable(getattr(config, attr)) and not attr.startswith("__")]
    message = "⚙️ **PENGATURAN BOT SAAT INI** ⚙️\n" + "="*30 + "\n"
    for var in config_vars:
        message += f"• `{var}`: {getattr(config, var)}\n"
    await update.message.reply_text(message, parse_mode='Markdown')
    
async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Menampilkan dashboard statistik performa sinyal."""
    stats = trading_engine.database.get_performance_stats()
    
    total = stats['total_trades']
    wins = stats['wins']
    losses = stats['losses']
    win_rate = stats['win_rate']
    
    message = (
        f"📊 **LAPORAN PERFORMA SINYAL** 📊\n"
        f"{'='*30}\n"
        f"• Total Trade Selesai: {total}\n"
        f"• ✅ Menang (Win): {wins}\n"
        f"• 😥 Kalah (Loss): {losses}\n\n"
        f"• **🎯 Tingkat Kemenangan (Win Rate): {win_rate:.2f}%**"
    )
    
    await update.message.reply_text(message, parse_mode='Markdown')
    
async def briefing_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Menghasilkan dan mengirim laporan briefing."""
    await update.message.reply_text("Membuat laporan briefing, harap tunggu...")
    briefing_message = trading_engine.generate_briefing()
    await update.message.reply_text(briefing_message, parse_mode='Markdown')
    
async def snr_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Menampilkan level Support & Resistance untuk sebuah simbol."""
    if not context.args:
        await update.message.reply_text("Gunakan format: /snr SIMBOL (contoh: /snr BTC/USDT)")
        return
        
    symbol = " ".join(context.args).upper()
    await update.message.reply_text(f"Mencari level Support & Resistance untuk {symbol}...")
    
    # Panggil logika dari provider kita
    s_r_data = trading_engine.economic_provider.get_support_resistance(symbol)
    
    if s_r_data and s_r_data.get('supports'):
        supports = s_r_data['supports']
        
        # Format pesannya
        message = f"📈 **Analisis Support & Resistance untuk {symbol}** 📈\n"
        message += "Resolusi: Harian (Daily)\n\n"
        message += "**Level Support Terdeteksi:**\n"
        for level in supports:
            message += f"• ${level:,.2f}\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')
    else:
        await update.message.reply_text(f"❌ Gagal menemukan data S&R untuk {symbol}. Pastikan simbol valid.")

def main() -> None:
    """Start the bot."""
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN tidak diatur! Bot tidak bisa dimulai.")
        return

    os.system('clear')
    print("🚀 Memulai Bot... 🚀")
    
    # --- KONFIGURASI TIMEOUT VERSI FINAL & BENAR ---
    application = (
        Application.builder()
        .token(bot_token)
        .connect_timeout(10.0)
        .read_timeout(30.0)
        .write_timeout(20.0)
        .build()
    )
    # ---------------------------------------------

    # Daftarkan semua handler
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(CommandHandler("watch", watch_command))
    application.add_handler(CommandHandler("set", set_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("briefing", briefing_command))
    application.add_handler(CommandHandler("snr", snr_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    
    # Setup JobQueue untuk tugas latar belakang
    job_queue = application.job_queue
    job_queue.run_repeating(background_tasks, interval=300, first=15)
    
    application.add_error_handler(error_handler)

    print("✨ Bot is running! Kirim /start di Telegram. ✨")
    print("Tekan Ctrl+C untuk berhenti.")
    
    application.run_polling()

if __name__ == '__main__':
    main()
