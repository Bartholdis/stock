# -*- coding: utf-8 -*-
"""
从 pcr_passed_us_stocks.csv 中筛选「最后一次入场时间」在指定年月的股票，结果保存到 data/ 下。
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
PASSED_CSV = os.path.join(DATA_DIR, "pcr_passed_us_stocks.csv")

# 目标年月：最后一次入场时间在该月的股票
TARGET_YEAR = 2026
TARGET_MONTH = 2
OUTPUT_CSV = os.path.join(DATA_DIR, "pcr_passed_last_entry_2026_02.csv")


def filter_by_last_entry_month(
    csv_path: str,
    year: int,
    month: int,
    output_path: str = None,
) -> pd.DataFrame:
    """
    读取 pcr_passed_us_stocks.csv，筛选 last_allow_entry_date 在指定年月的股票。
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"未找到文件: {csv_path}，请先运行 screen_all_us_stocks.py")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "last_allow_entry_date" not in df.columns:
        raise ValueError("CSV 中缺少列 last_allow_entry_date")
    df["last_allow_entry_date"] = pd.to_datetime(df["last_allow_entry_date"], errors="coerce")
    df = df.dropna(subset=["last_allow_entry_date"])
    mask = (df["last_allow_entry_date"].dt.year == year) & (
        df["last_allow_entry_date"].dt.month == month
    )
    out = df.loc[mask].copy()
    out = out.sort_values("allow_entry_days", ascending=False).reset_index(drop=True)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        out.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"已保存: {output_path}，共 {len(out)} 只（最后一次入场在 {year}-{month:02d}）")
    return out


if __name__ == "__main__":
    try:
        result = filter_by_last_entry_month(
            PASSED_CSV,
            TARGET_YEAR,
            TARGET_MONTH,
            output_path=OUTPUT_CSV,
        )
        if result.empty:
            print(f"没有最后一次入场时间在 {TARGET_YEAR}-{TARGET_MONTH:02d} 的股票。")
        else:
            print("\n前 20 只:")
            print(result.head(20).to_string(index=False))
    except Exception as e:
        import traceback
        print("报错:", e)
        traceback.print_exc()
        exit(1)
