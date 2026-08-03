"""One-off probe for NSE feed alternatives. Run: python scripts/probe_feeds.py"""
import re
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0 Safari/537.36"
    )
}

URLS = [
    "https://live.mystocks.co.ke/m/pricelist",
    "https://live.mystocks.co.ke/pricelist.php",
    "https://live.mystocks.co.ke/stock=SCOM",
    "https://live.mystocks.co.ke/stock=EQTY",
    "https://afx.kwayisi.org/nse/",
]

for url in URLS:
    print("\n===", url)
    try:
        r = requests.get(url, timeout=15, headers=HEADERS)
        print("status", r.status_code, "bytes", len(r.content))
        blobs = re.findall(r'\{"reload".*?\}', r.text)
        if blobs:
            print("json blob sample:", blobs[0][:250])
        print("tr count:", len(re.findall(r"<tr", r.text, re.I)))
    except Exception as exc:
        print("FAIL", type(exc).__name__, exc)
