# -*- coding: utf-8 -*-
"""
PCR 四层门禁选股：环境层 / 模板层 / RS层 / 选择压
数据源：yfinance，代理与 get_stock.ipynb 一致。
"""

import os
proxy = 'http://127.0.0.1:7890'  # 代理设置，此处修改
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


# ========== 参数（调优只动这里） ==========
SHORT_MA = 50
LONG_MA = 200
RS_FAST = 20
RS_SLOW = 50
ATR_LEN = 14
ATR_SMOOTH = 50
LOWEST_LOW_LOOKBACK = 250
PRIOR_HIGH_LOOKBACK = 120
INDEX_SYMBOL = "QQQ"   # 市场指数，不用指数可设为 None
SECTOR_SYMBOL = None   # 行业 ETF，不用可设为 None
USE_VIX = False        # 是否用 VIX 做环境风险


def _ensure_df(df):
    """yf.download 单标的时列可能是 MultiIndex，统一成单层列。"""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_ohlc(symbol, start, end):
    """用 yf 拉取 OHLC，返回列 Open/High/Low/Close 的 DataFrame，索引为日期。"""
    data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()
    data = _ensure_df(data)
    for c in ("Open", "High", "Low", "Close"):
        if c not in data.columns:
            return pd.DataFrame()
    data = data[["Open", "High", "Low", "Close"]].copy()
    # 统一为日期索引、无时区，避免与其它标的 reindex 对不齐
    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    return data


def calc_env_layer(idx_close: pd.Series) -> pd.Series:
    """环境层：指数趋势 (50>200 或 200 线斜率>0)。"""
    idx_ma50 = idx_close.rolling(SHORT_MA).mean()
    idx_ma200 = idx_close.rolling(LONG_MA).mean()
    slope_200 = idx_ma200.diff(20)
    env_trend = ((idx_ma50 > idx_ma200) | (slope_200 > 0)).fillna(False)
    return env_trend


def calc_env_with_vix(vix_close: pd.Series) -> pd.Series:
    """环境风险：VIX < VIX_MA50。"""
    vix_ma = vix_close.rolling(50).mean()
    return vix_close < vix_ma


def calc_template_layer(df: pd.DataFrame) -> pd.Series:
    """模板层：波动收缩 + 无 250 周期新低。"""
    atr = (df["High"] - df["Low"]).rolling(ATR_LEN).mean()
    atr_smooth = atr.rolling(ATR_SMOOTH).mean()
    volatility_compress = (atr < atr_smooth).fillna(False)
    no_lower_low = (df["Low"] > df["Low"].rolling(LOWEST_LOW_LOOKBACK).min()).fillna(False)
    return volatility_compress & no_lower_low


def calc_rs_layer(
    stock_close: pd.Series,
    idx_close: pd.Series,
    sector_close: pd.Series = None,
) -> pd.Series:
    """RS 层：相对指数（及可选相对行业）走强。"""
    idx_safe = idx_close.replace(0, np.nan)
    rs = stock_close / idx_safe
    rs_fast = rs.rolling(RS_FAST).mean()
    rs_slow = rs.rolling(RS_SLOW).mean()
    rs_ok = (rs_fast > rs_slow).fillna(False)
    if sector_close is not None and len(sector_close) > 0:
        sec_safe = sector_close.replace(0, np.nan)
        rs_vs_sector = stock_close / sec_safe
        rs_sector_ok = (rs_vs_sector.rolling(20).mean() > rs_vs_sector.rolling(50).mean()).fillna(False)
        rs_ok = rs_ok & rs_sector_ok
    return rs_ok


def calc_selection_layer(df: pd.DataFrame) -> pd.Series:
    """选择压：突破 120 前高 或 回撤不破前高。"""
    prior_high = df["High"].rolling(PRIOR_HIGH_LOOKBACK).max().shift(1)
    breakout = (df["Close"] > prior_high).fillna(False)
    pullback_safe = (df["Low"] > prior_high).fillna(False)
    return breakout | pullback_safe


def run_pcr_on_ohlc(
    stock_ohlc_df: pd.DataFrame,
    index_close_series: pd.Series = None,
    vix_close_series: pd.Series = None,
    sector_close_series: pd.Series = None,
) -> pd.DataFrame:
    """
    对已加载的 OHLC 做 PCR 四层筛选（不访问网络）。
    stock_ohlc_df: 列 Open/High/Low/Close，索引为日期。
    index_close_series: 指数收盘序列，索引为日期；为 None 时不做环境层与 RS 层。
    返回与 screen_one_stock 相同格式：Close, allow_entry, exit_signal。
    """
    df = stock_ohlc_df.copy()
    for c in ("Open", "High", "Low", "Close"):
        if c not in df.columns:
            return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if len(df) < max(LONG_MA, LOWEST_LOW_LOOKBACK, PRIOR_HIGH_LOOKBACK):
        return pd.DataFrame()

    idx_dates = df.index
    idx_close = None
    if index_close_series is not None and len(index_close_series) > 0:
        idx_close = index_close_series.reindex(idx_dates).ffill().bfill()

    if idx_close is not None:
        env_trend = calc_env_layer(idx_close)
        if vix_close_series is not None and len(vix_close_series) > 0:
            vix_aligned = vix_close_series.reindex(idx_dates).ffill().bfill()
            env_risk = calc_env_with_vix(vix_aligned)
            env_on = (env_trend & env_risk).fillna(False)
        else:
            env_on = env_trend.fillna(False)
    else:
        env_on = pd.Series(True, index=idx_dates).fillna(True)

    template_ok = calc_template_layer(df)
    if idx_close is not None:
        sector_ser = None
        if sector_close_series is not None and len(sector_close_series) > 0:
            sector_ser = sector_close_series.reindex(idx_dates).ffill().bfill()
        rs_ok = calc_rs_layer(df["Close"], idx_close, sector_ser)
    else:
        rs_ok = pd.Series(True, index=idx_dates).fillna(True)
    selection_ok = calc_selection_layer(df)

    allow_entry = (env_on & template_ok & rs_ok & selection_ok).fillna(False)
    if idx_close is not None:
        rs = df["Close"] / idx_close.replace(0, np.nan)
        exit_rs = (rs.rolling(RS_FAST).mean() < rs.rolling(RS_SLOW).mean()).fillna(False)
    else:
        exit_rs = pd.Series(False, index=idx_dates)
    exit_ma = (df["Close"] < df["Close"].rolling(50).mean()).fillna(False)
    exit_signal = (exit_rs | exit_ma).fillna(False)

    out = df[["Close"]].copy()
    out["allow_entry"] = allow_entry
    out["exit_signal"] = exit_signal
    return out


def screen_one_stock(
    symbol: str,
    start: str,
    end: str,
    index_symbol: str = None,
    sector_symbol: str = None,
    use_vix: bool = False,
    index_ohlc_df: pd.DataFrame = None,
    sector_ohlc_df: pd.DataFrame = None,
    vix_close_series: pd.Series = None,
) -> pd.DataFrame:
    """
    对单只股票做 PCR 四层筛选，返回带 allow_entry / exit_signal 的 DataFrame。
    index_symbol 为 None 且 index_ohlc_df 为 None 时不做环境层与 RS 层（仅模板层+选择压）。
    批量筛股时可传入 index_ohlc_df / sector_ohlc_df / vix_close_series 避免重复拉取指数。
    """
    df = fetch_ohlc(symbol, start, end)
    if df.empty or len(df) < max(LONG_MA, LOWEST_LOW_LOOKBACK, PRIOR_HIGH_LOOKBACK):
        return pd.DataFrame()

    idx_dates = df.index
    idx_close = None

    if index_ohlc_df is not None and not index_ohlc_df.empty and "Close" in index_ohlc_df.columns:
        idx_close = index_ohlc_df["Close"].reindex(idx_dates).ffill().bfill()
    elif index_symbol:
        idx_df = fetch_ohlc(index_symbol, start, end)
        if not idx_df.empty:
            idx_close = idx_df["Close"].reindex(idx_dates).ffill().bfill()

    # 1) 环境层
    if idx_close is not None:
        env_trend = calc_env_layer(idx_close)
        if use_vix and vix_close_series is not None and len(vix_close_series) > 0:
            vix_aligned = vix_close_series.reindex(idx_dates).ffill().bfill()
            env_risk = calc_env_with_vix(vix_aligned)
            env_on = (env_trend & env_risk).fillna(False)
        elif use_vix:
            try:
                vix_df = yf.download("^VIX", start=start, end=end, progress=False)
                if not vix_df.empty:
                    vix_df = _ensure_df(vix_df)
                    vix_close = vix_df["Close"].reindex(idx_dates).ffill().bfill()
                    env_risk = calc_env_with_vix(vix_close)
                    env_on = (env_trend & env_risk).fillna(False)
                else:
                    env_on = env_trend.fillna(False)
            except Exception:
                env_on = env_trend.fillna(False)
        else:
            env_on = env_trend.fillna(False)

    # 2) 模板层
    template_ok = calc_template_layer(df)

    # 3) RS 层
    if idx_close is not None:
        sector_ser = None
        if sector_ohlc_df is not None and not sector_ohlc_df.empty and "Close" in sector_ohlc_df.columns:
            sector_ser = sector_ohlc_df["Close"].reindex(idx_dates).ffill().bfill()
        elif sector_symbol:
            sec_df = fetch_ohlc(sector_symbol, start, end)
            if not sec_df.empty:
                sector_ser = sec_df["Close"].reindex(idx_dates).ffill().bfill()
        rs_ok = calc_rs_layer(df["Close"], idx_close, sector_ser)
    else:
        rs_ok = pd.Series(True, index=idx_dates)

    # 4) 选择压
    selection_ok = calc_selection_layer(df)

    # 门禁（统一布尔，避免 NaN）
    allow_entry = (env_on & template_ok & rs_ok & selection_ok).fillna(False)

    # 出场：RS 转弱 或 收盘跌破 50 日均线
    if idx_close is not None:
        rs = df["Close"] / idx_close
        rs_fast = rs.rolling(RS_FAST).mean()
        rs_slow = rs.rolling(RS_SLOW).mean()
        exit_rs = rs_fast < rs_slow
    else:
        exit_rs = pd.Series(False, index=idx_dates)
    exit_ma = (df["Close"] < df["Close"].rolling(50).mean()).fillna(False)
    exit_signal = (exit_rs | exit_ma).fillna(False)

    out = df[["Close"]].copy()
    out["allow_entry"] = allow_entry
    out["exit_signal"] = exit_signal
    return out


def screen_batch(
    symbols: list,
    start: str,
    end: str,
    index_symbol: str = None,
    sector_symbol: str = None,
    use_vix: bool = False,
) -> dict:
    """
    批量筛股。返回 { symbol: DataFrame with allow_entry, exit_signal }。
    """
    index_symbol = index_symbol or INDEX_SYMBOL
    sector_symbol = sector_symbol or SECTOR_SYMBOL
    use_vix = use_vix if index_symbol else False

    results = {}
    for sym in symbols:
        try:
            df = screen_one_stock(sym, start, end, index_symbol, sector_symbol, use_vix)
            if not df.empty:
                results[sym] = df
        except Exception as e:
            print(f"Skip {sym}: {e}")
    return results


# ========== 示例：单股 + 批量 ==========
if __name__ == "__main__":
    try:
        # 回测区间：结束日=运行当天，开始日=往前 400 天（约 1.1 年），美股
        end_d = datetime.now()
        start_d = end_d - timedelta(days=400)
        start = start_d.strftime("%Y-%m-%d")
        end = end_d.strftime("%Y-%m-%d")
        print("回测区间（美股）:", start, "~", end)

        # 单只
        one = screen_one_stock("AAPL", start, end, index_symbol=INDEX_SYMBOL, use_vix=USE_VIX)
        if not one.empty:
            passed = one["allow_entry"].sum()
            print("AAPL allow_entry 天数:", int(passed))
            print(one.tail())
        else:
            print("AAPL 无足够数据或返回为空")

        # 多只
        batch = screen_batch(["AAPL", "MSFT", "GOOGL"], start, end)
        for sym, df in batch.items():
            print(sym, "allow_entry 天数:", int(df["allow_entry"].sum()))
    except Exception as e:
        import traceback
        print("报错:", e)
        traceback.print_exc()
