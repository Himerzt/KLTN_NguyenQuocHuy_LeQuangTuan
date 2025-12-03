# telegram_utils.py
import aiohttp

TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = 123456789  # ID Telegram của bạn

async def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with aiohttp.ClientSession() as session:
        await session.post(url, json={
            "chat_id": CHAT_ID,
            "text": text
        })

async def send_telegram_photo(image_bytes: bytes, caption: str = ""):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    data = aiohttp.FormData()
    data.add_field('chat_id', str(CHAT_ID))
    data.add_field('caption', caption)
    data.add_field('photo', image_bytes, filename="alert.jpg", content_type="image/jpeg")

    async with aiohttp.ClientSession() as session:
        await session.post(url, data=data)
