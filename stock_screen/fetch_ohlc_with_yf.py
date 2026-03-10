# -*- coding: utf-8 -*-
"""
用 yfinance 从网络获取美股及指数 OHLC，保存到项目 stock_screen/data/ 下。
先拉取标的列表（若无则自动拉取），再批量下载 OHLC 并落盘。代理与 get_stock.ipynb 一致。
"""

import os
proxy = 'http://127.0.0.1:7890'  # 代理设置，此处修改
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
LIST_CSV = os.path.join(DATA_DIR, "us_stocks_list.csv")
INDEX_CSV = os.path.join(DATA_DIR, "index_QQQ.csv")
OHLC_CSV = os.path.join(DATA_DIR, "ohlc_us.csv")

BACKTEST_DAYS = 400
BATCH_SIZE = 80   # 每批下载数量，避免 URL 过长


def _ensure_df(df):
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def ensure_symbol_list():
    """若无 us_stocks_list.csv 则先拉取美股列表并保存。"""
    if os.path.isfile(LIST_CSV):
        return pd.read_csv(LIST_CSV, encoding="utf-8-sig")
    try:
        from fetch_us_stock_list import fetch_and_save
        fetch_and_save(LIST_CSV)
        return pd.read_csv(LIST_CSV, encoding="utf-8-sig")
    except Exception as e:
        raise FileNotFoundError(
            f"未找到 {LIST_CSV}，且自动拉取失败: {e}。请先运行: python fetch_us_stock_list.py"
        )


def fetch_index_and_save(index_symbol: str = "QQQ", start: str = None, end: str = None):
    """用 yf 拉取指数 OHLC 并保存到 data/index_QQQ.csv。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    if end is None:
        end_d = datetime.now()
        start_d = end_d - timedelta(days=BACKTEST_DAYS)
        end = end_d.strftime("%Y-%m-%d")
        start = start_d.strftime("%Y-%m-%d")
    print(f"正在拉取指数 {index_symbol} OHLC: {start} ~ {end}")
    df = yf.download(index_symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"指数 {index_symbol} 未取到数据")
    df = _ensure_df(df)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[["Open", "High", "Low", "Close"]].copy()
    df.to_csv(INDEX_CSV, encoding="utf-8-sig")
    print(f"已保存: {INDEX_CSV}")
    return df


def fetch_stocks_ohlc_and_save(
    symbols: list,
    start: str,
    end: str,
    save_path: str = None,
):
    """用 yf 分批拉取个股 OHLC，合并为长表保存到 data/ohlc_us.csv。"""
    save_path = save_path or OHLC_CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out_rows = []
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i : i + BATCH_SIZE]
        try:
            data = yf.download(
                batch,
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as e:
            print(f"  批次 {i//BATCH_SIZE + 1} 下载失败: {e}")
            continue
        if data is None or data.empty:
            continue
        # 多只时为 MultiIndex (symbol, Open)...；单只时可能无 MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            tickers = data.columns.get_level_values(0).unique()
            for sym in tickers:
                try:
                    sub = data[sym].copy()
                except Exception:
                    continue
                if sub is None or sub.empty:
                    continue
                sub = _ensure_df(sub)
                for c in ("Open", "High", "Low", "Close"):
                    if c not in sub.columns:
                        break
                else:
                    sub.index = pd.to_datetime(sub.index)
                    if sub.index.tz is not None:
                        sub.index = sub.index.tz_localize(None)
                    for ts, row in sub.iterrows():
                        out_rows.append({
                            "date": ts.strftime("%Y-%m-%d"),
                            "symbol": str(sym),
                            "open": float(row["Open"]) if pd.notna(row["Open"]) else np.nan,
                            "high": float(row["High"]) if pd.notna(row["High"]) else np.nan,
                            "low": float(row["Low"]) if pd.notna(row["Low"]) else np.nan,
                            "close": float(row["Close"]) if pd.notna(row["Close"]) else np.nan,
                        })
        else:
            sub = _ensure_df(data)
            sym = batch[0]
            for c in ("Open", "High", "Low", "Close"):
                if c not in sub.columns:
                    break
            else:
                sub.index = pd.to_datetime(sub.index)
                if sub.index.tz is not None:
                    sub.index = sub.index.tz_localize(None)
                for ts, row in sub.iterrows():
                    out_rows.append({
                        "date": ts.strftime("%Y-%m-%d"),
                        "symbol": str(sym),
                        "open": float(row["Open"]) if pd.notna(row["Open"]) else np.nan,
                        "high": float(row["High"]) if pd.notna(row["High"]) else np.nan,
                        "low": float(row["Low"]) if pd.notna(row["Low"]) else np.nan,
                        "close": float(row["Close"]) if pd.notna(row["Close"]) else np.nan,
                    })
        print(f"  已处理 {min(i + BATCH_SIZE, len(symbols))}/{len(symbols)} 只")
    if not out_rows:
        raise RuntimeError("未成功下载任何个股 OHLC")
    ohlc_df = pd.DataFrame(out_rows)
    ohlc_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"已保存: {save_path}，共 {len(ohlc_df)} 行")
    return ohlc_df


def main():
    end_d = datetime.now()
    start_d = end_d - timedelta(days=BACKTEST_DAYS)
    start = start_d.strftime("%Y-%m-%d")
    end = end_d.strftime("%Y-%m-%d")
    print("回测区间:", start, "~", end)

    list_df = ensure_symbol_list()
    col_sym = "symbol" if "symbol" in list_df.columns else list_df.columns[0]
    symbols = list_df[col_sym].astype(str).str.strip().dropna().unique().tolist()
    symbols = [s for s in symbols if s and len(s) <= 10 and s != "nan"]
    print(f"标的数量: {len(symbols)}")

    fetch_index_and_save("QQQ", start, end)
    fetch_stocks_ohlc_and_save(symbols, start, end)
    print("全部保存完成，可运行筛选脚本。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("报错:", e)
        traceback.print_exc()
        exit(1)
