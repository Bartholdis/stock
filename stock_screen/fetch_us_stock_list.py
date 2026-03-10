# -*- coding: utf-8 -*-
"""
从网络获取美股列表（NASDAQ / NYSE / AMEX），保存到 stock_screen/data/us_stocks_list.csv。
代理与 get_stock.ipynb 一致。
"""

import os
proxy = 'http://127.0.0.1:7890'  # 代理设置，此处修改
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy

import json
import urllib.request
import ssl
import pandas as pd
from datetime import datetime

# 保存路径：stock_screen/data/us_stocks_list.csv
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "us_stocks_list.csv")

# NASDAQ 官方 Screener API（需带 User-Agent）
BASE_URL = "https://api.nasdaq.com/api/screener/stocks"
EXCHANGES = ["nasdaq", "nyse", "amex"]
LIMIT = 25000


def _request(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        },
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        return json.loads(resp.read().decode())


def fetch_exchange(exchange: str) -> list:
    """拉取单个交易所的股票列表，返回 [{"symbol","name","exchange","assetType"}, ...]"""
    url = f"{BASE_URL}?tableonly=true&limit={LIMIT}&exchange={exchange}"
    try:
        data = _request(url)
    except Exception as e:
        print(f"  {exchange} 请求失败: {e}")
        return []
    # 结构一般为 data.rows 或 data.table.rows
    rows = []
    if "data" in data:
        d = data["data"]
        if "rows" in d:
            rows = d["rows"]
        elif "table" in d and "rows" in d["table"]:
            rows = d["table"]["rows"]
    out = []
    for r in rows:
        if isinstance(r, dict):
            sym = (r.get("symbol") or r.get("Symbol") or "").strip()
            name = (r.get("name") or r.get("Name") or "").strip()
            asset = (r.get("assetType") or r.get("type") or "Common Stock").strip()
        elif isinstance(r, list):
            # 部分 API 返回 [symbol, name, ...]
            sym = str(r[0]).strip() if len(r) > 0 else ""
            name = str(r[1]).strip() if len(r) > 1 else ""
            asset = "Common Stock"
        else:
            continue
        if not sym or len(sym) > 10:
            continue
        out.append({
            "symbol": sym,
            "name": name,
            "exchange": exchange.upper(),
            "assetType": asset,
        })
    return out


def fetch_and_save(save_path: str = None) -> pd.DataFrame:
    """
    从网络获取全部美股列表，保存为 CSV，返回 DataFrame。
    列：symbol, name, exchange, assetType, fetch_date
    """
    save_path = save_path or OUTPUT_CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_rows = []
    for ex in EXCHANGES:
        print(f"正在获取 {ex.upper()} 列表...")
        rows = fetch_exchange(ex)
        print(f"  -> {len(rows)} 条")
        all_rows.extend(rows)
    if not all_rows:
        raise RuntimeError("未获取到任何股票数据，请检查网络或代理。")
    df = pd.DataFrame(all_rows)
    # 去重（同一 symbol 可能多交易所）
    df = df.drop_duplicates(subset=["symbol"], keep="first")
    df["fetch_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"已保存 {len(df)} 只标的到: {save_path}")
    return df


if __name__ == "__main__":
    try:
        fetch_and_save()
    except Exception as e:
        import traceback
        print("报错:", e)
        traceback.print_exc()
        exit(1)
