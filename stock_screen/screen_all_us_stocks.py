# -*- coding: utf-8 -*-
"""
对美股做 PCR 四层筛选：从项目下已保存的 OHLC 数据读取，不再从项目内其他 CSV 或实时网络拉取。
请先运行 fetch_ohlc_with_yf.py 用 yf 获取数据并保存到 stock_screen/data/，再运行本脚本。
"""

import os
proxy = 'http://127.0.0.1:7890'  # 代理设置，此处修改
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy

import pandas as pd
import sys

from screen_pcr import run_pcr_on_ohlc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OHLC_CSV = os.path.join(DATA_DIR, "ohlc_us.csv")
INDEX_CSV = os.path.join(DATA_DIR, "index_QQQ.csv")
# 结果文件统一保存到 data 目录
OUTPUT_DIR = DATA_DIR
OUTPUT_FULL_CSV = "pcr_all_us_results.csv"
OUTPUT_PASSED_CSV = "pcr_passed_us_stocks.csv"
PRINT_EVERY = 100
MAX_SYMBOLS = None   # None=不限制，可设 500 等先试跑


def load_saved_ohlc():
    """从项目 data/ 下读取已保存的 OHLC 与指数。"""
    if not os.path.isfile(OHLC_CSV):
        raise FileNotFoundError(
            f"未找到 {OHLC_CSV}。请先运行: python fetch_ohlc_with_yf.py"
        )
    if not os.path.isfile(INDEX_CSV):
        raise FileNotFoundError(
            f"未找到 {INDEX_CSV}。请先运行: python fetch_ohlc_with_yf.py"
        )
    ohlc = pd.read_csv(OHLC_CSV, encoding="utf-8-sig")
    ohlc["date"] = pd.to_datetime(ohlc["date"])
    idx = pd.read_csv(INDEX_CSV, index_col=0, parse_dates=True)
    idx.index = pd.to_datetime(idx.index)
    if idx.index.tz is not None:
        idx.index = idx.index.tz_localize(None)
    index_close = idx["Close"].copy()
    return ohlc, index_close


def run_screen_from_saved(
    ohlc_df: pd.DataFrame,
    index_close_series: pd.Series,
    max_symbols: int = None,
) -> pd.DataFrame:
    """基于已加载的 OHLC 与指数，对每只股票跑 PCR，返回汇总表。"""
    symbols = ohlc_df["symbol"].astype(str).unique().tolist()
    symbols = [s for s in symbols if s and s != "nan"]
    if max_symbols is not None:
        symbols = symbols[: int(max_symbols)]
    print(f"共 {len(symbols)} 只标的待筛选（数据来自本地 {OHLC_CSV}）")

    rows = []
    for i, sym in enumerate(symbols):
        try:
            sub = ohlc_df[ohlc_df["symbol"] == sym].copy()
            if sub.empty or len(sub) < 250:
                continue
            sub = sub.set_index("date").sort_index()
            sub = sub.rename(columns={
                "open": "Open", "high": "High", "low": "Low", "close": "Close",
            })
            for c in ("Open", "High", "Low", "Close"):
                if c not in sub.columns:
                    break
            else:
                out = run_pcr_on_ohlc(sub, index_close_series)
                if out.empty:
                    continue
                allow = out["allow_entry"]
                n_days = int(allow.sum())
                last_date = out.index[allow].max() if n_days > 0 else pd.NaT
                rows.append({
                    "symbol": sym,
                    "allow_entry_days": n_days,
                    "last_allow_entry_date": last_date if pd.notna(last_date) else "",
                })
        except Exception as e:
            if (i + 1) % 500 == 0 or i < 5:
                print(f"  Skip {sym}: {e}")
        if (i + 1) % PRINT_EVERY == 0:
            print(f"  已处理 {i + 1}/{len(symbols)}")

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values("allow_entry_days", ascending=False).reset_index(drop=True)
    return result


def main():
    print("从项目下已保存数据读取并筛选（不访问网络）")
    try:
        ohlc_df, index_close = load_saved_ohlc()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    result = run_screen_from_saved(
        ohlc_df,
        index_close,
        max_symbols=MAX_SYMBOLS,
    )

    if result.empty:
        print("没有产出任何有效结果。")
        sys.exit(0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_path = os.path.join(OUTPUT_DIR, OUTPUT_FULL_CSV)
    result.to_csv(full_path, index=False, encoding="utf-8-sig")
    print(f"全量结果已写入: {full_path}")

    passed = result[result["allow_entry_days"] >= 1]
    passed_path = os.path.join(OUTPUT_DIR, OUTPUT_PASSED_CSV)
    passed.to_csv(passed_path, index=False, encoding="utf-8-sig")
    print(f"通过列表（allow_entry_days>=1）已写入: {passed_path}，共 {len(passed)} 只")
    print("\n通过数量前 20 只:")
    print(passed.head(20).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("报错:", e)
        traceback.print_exc()
        sys.exit(1)
