# live_loop.py
import time, requests
from predict_live import predict_from_tick

def fetch_prices():
    # Replace with your preferred API (this is just an example)
    j = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,litecoin&vs_currencies=usd").json()
    return {
        "bitcoin_price":  j["bitcoin"]["usd"],
        "ethereum_price": j["ethereum"]["usd"],
        "litecoin_price": j["litecoin"]["usd"],
    }

def fetch_volumes():
    # Replace or remove if not easily available from your API; 0 is acceptable placeholder
    return {"bitcoin_volume": 0.0, "ethereum_volume": 0.0, "litecoin_volume": 0.0}

while True:
    tick = {**fetch_prices(), **fetch_volumes()}
    res = predict_from_tick(tick)
    if res["ready"]:
        print(res)
    time.sleep(15)
