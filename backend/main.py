from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import baostock as bs
import pandas as pd
import datetime
import akshare as ak
import efinance as ef
import time
import diskcache
from diskcache import Cache
from typing import List, Dict, Optional
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Cache for stock list and data
cache = Cache('./cache')
@cache.memoize(expire=86400)  # Cache for 24 hours
def get_stock_list():
    try:
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            return df[['代码', '名称']].to_dict('records')
    except Exception as e:
        print(f"Error fetching stock list: {e}")
    return []

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChipSettings(BaseModel):
    algorithm: str = "triangular"  # "triangular" or "average"
    decay: float = 1.0
    lookback: int = 250
    use_turnover: bool = True
    decay_factor: float = 1.0
    peakUpperPercent: float = 10.0
    peakLowerPercent: float = 10.0
    showPeakArea: bool = True
    longTermDays: int = 100
    mediumTermDays: int = 10
    shortTermDays: int = 2

class BacktestRequest(BaseModel):
    code: str
    buy_threshold: float
    sell_threshold: float
    source: str = "baostock"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    chip_settings: Optional[ChipSettings] = None
    strategy_type: str = "mean_reversion" # "breakout", "mean_reversion", or "buy_and_hold"

def calculate_advanced_chips(df: pd.DataFrame, settings: ChipSettings, include_all_distributions: bool = False):
    # Ensure columns are numeric and drop rows with missing essential data
    for col in ['close', 'high', 'low', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['close', 'high', 'low', 'volume'])
    
    if df.empty:
        return df, [], [], {}

    if 'turn' in df.columns:
        df['turn'] = pd.to_numeric(df['turn'], errors='coerce').fillna(0)
    
    # Define price buckets for the whole period
    min_price = df['low'].min()
    max_price = df['high'].max()
    
    if pd.isna(min_price) or pd.isna(max_price):
        return df, [], [], {}
        
    min_price *= 0.9
    max_price *= 1.1
    
    num_buckets = 500
    price_bins = np.linspace(min_price, max_price, num_buckets)
    bin_width = price_bins[1] - price_bins[0]
    
    chips = np.zeros(num_buckets)
    profit_ratios = []
    
    # To store historical distributions and stats
    last_distribution = []
    all_distributions = {}
    all_summary_stats = {}
    
    # Calculate simple volume distribution for the requested range
    range_volume_dist = np.zeros(num_buckets)

    # Pre-convert to numpy for speed
    lows = df['low'].values
    highs = df['high'].values
    closes = df['close'].values
    volumes = df['volume'].values
    turns = df['turn'].values if 'turn' in df.columns else np.ones(len(df)) * (settings.decay * 100)
    dates = df['date'].values

    def get_stats_for_chips(current_chips, current_close):
        total = current_chips.sum()
        if total <= 0:
            return None
        
        avg_cost = np.sum(current_chips * price_bins) / total
        cumsum = np.cumsum(current_chips)
        
        def get_conc(percent):
            target = total * (1 - percent / 100) / 2
            l_idx = np.searchsorted(cumsum, target)
            h_idx = np.searchsorted(cumsum, total - target)
            if h_idx >= len(price_bins): h_idx = len(price_bins) - 1
            l_p = price_bins[l_idx]
            h_p = price_bins[h_idx]
            width = h_p - l_p
            center = (h_p + l_p) / 2
            conc = (width / center) * 100 if center > 0 else 0
            return {
                "low": round(float(l_p), 2),
                "high": round(float(h_p), 2),
                "concentration": round(float(conc), 2)
            }

        asr_l = current_close * 0.925
        asr_h = current_close * 1.075
        asr_m = (price_bins >= asr_l) & (price_bins <= asr_h)
        asr_v = (current_chips[asr_m].sum() / total * 100)
        
        c50 = price_bins[np.searchsorted(cumsum, total * 0.5)]
        c90 = price_bins[np.searchsorted(cumsum, total * 0.9)]
        
        win_chips = current_chips[price_bins < current_close].sum()
        p_ratio = (win_chips / total * 100)

        return {
            "avg_cost": round(float(avg_cost), 2),
            "conc_90": get_conc(90),
            "conc_70": get_conc(70),
            "peak_price": round(float(price_bins[np.argmax(current_chips)]), 2),
            "profit_ratio": round(float(p_ratio), 2),
            "asr": round(float(asr_v), 2),
            "cost_50": round(float(c50), 2),
            "cost_90": round(float(c90), 2)
        }

    for i in range(len(df)):
        low, high, close, vol, turn = lows[i], highs[i], closes[i], volumes[i], turns[i]
        
        # Add to range volume distribution
        if high > low:
            mask = (price_bins >= low) & (price_bins <= high)
            if mask.any():
                range_volume_dist[mask] += vol / mask.sum()
        else:
            idx = np.abs(price_bins - close).argmin()
            range_volume_dist[idx] += vol

        # 1. Decay existing chips
        if settings.use_turnover:
            turnover_decimal = turn / 100.0
            current_decay = max(0.0, 1.0 - turnover_decimal * settings.decay_factor)
            chips *= current_decay
        else:
            chips *= settings.decay
        
        # 2. Add new chips
        if high > low:
            current_low = low
            current_high = high
            if settings.showPeakArea:
                range_width = high - low
                current_low = low + range_width * (settings.peakLowerPercent / 100.0)
                current_high = high - range_width * (settings.peakUpperPercent / 100.0)
                if current_high <= current_low:
                    current_low, current_high = low, high

            if settings.algorithm == "average":
                mask = (price_bins >= current_low) & (price_bins <= current_high)
                if mask.any():
                    chips[mask] += vol / mask.sum()
            else:
                peak = (current_low + current_high + 2 * close) / 4
                
                # Vectorized triangular weights
                weights = np.zeros(num_buckets)
                
                # Left side
                left_mask = (price_bins >= current_low) & (price_bins <= peak)
                if left_mask.any() and peak > current_low:
                    weights[left_mask] = (price_bins[left_mask] - current_low) / (peak - current_low)
                
                # Right side
                right_mask = (price_bins > peak) & (price_bins <= current_high)
                if right_mask.any() and current_high > peak:
                    weights[right_mask] = (current_high - price_bins[right_mask]) / (current_high - peak)
                
                sum_weights = weights.sum()
                if sum_weights > 0:
                    chips += (weights / sum_weights) * vol
        else:
            idx = np.abs(price_bins - close).argmin()
            chips[idx] += vol

        # 3. Calculate profit ratio
        total_chips = chips.sum()
        if total_chips > 0:
            winning_chips = chips[price_bins < close].sum()
            ratio = (winning_chips / total_chips * 100)
            profit_ratios.append(round(float(ratio), 2))
        else:
            profit_ratios.append(0)
        
        # 4. Store distribution and stats
        if include_all_distributions or i == len(df) - 1:
            # Optimization: only store non-zero chips
            valid_indices = chips > 1e-6 # Small threshold to avoid floating point noise
            current_dist = [
                {"price": round(float(price_bins[j]), 2), "volume": float(chips[j])}
                for j in np.where(valid_indices)[0]
            ]
            
            if include_all_distributions:
                all_distributions[str(dates[i])] = current_dist
                all_summary_stats[str(dates[i])] = get_stats_for_chips(chips, close)
            
            if i == len(df) - 1:
                last_distribution = current_dist

    df['profit_ratio'] = profit_ratios
    
    # Format range volume distribution
    range_dist_list = []
    for j in range(num_buckets):
        if range_volume_dist[j] > 0:
            range_dist_list.append({
                "price": round(float(price_bins[j]), 2),
                "volume": float(range_volume_dist[j])
            })

    # Calculate summary stats for the latest distribution
    summary_stats = get_stats_for_chips(chips, closes[-1])

    # Replace any NaN in the whole dataframe before returning
    df = df.fillna(0)
    return df, last_distribution, range_dist_list, all_distributions, summary_stats, all_summary_stats

@cache.memoize(expire=3600)
def get_fundamentals_cached(clean_code: str):
    info = {}
    
    # 1. 优先尝试从 efinance 获取最新行情（支持股票和 ETF）
    try:
        quote_df = ef.stock.get_latest_quote(clean_code)
        if quote_df is not None and not quote_df.empty:
            row = quote_df.iloc[0]
            info = {
                "股票代码": str(row['代码']),
                "股票简称": str(row['名称']),
                "最新价": row['最新价'],
                "涨跌幅": f"{row['涨跌幅']}%",
                "最高": row['最高'],
                "最低": row['最低'],
                "今开": row['今开'],
                "昨收": row['昨日收盘'],
                "成交量": row['成交量'],
                "成交额": row['成交额'],
                "换手率": f"{row['换手率']}%",
                "总市值": row['总市值'],
                "流通市值": row['流通市值'],
            }
    except Exception as e:
        print(f"Error fetching latest quote from efinance: {e}")

    # 2. 尝试从 akshare 获取补充信息
    is_etf = clean_code.startswith(('51', '56', '58', '15', '16', '50'))
    
    if is_etf:
        if "行业" not in info:
            info["行业"] = "ETF/基金"
        try:
            # 获取基金名称和类型
            df_name = ak.fund_name_em()
            fund_row = df_name[df_name['基金代码'] == clean_code]
            if not fund_row.empty:
                info["行业"] = fund_row.iloc[0]['基金类型']
                if "股票简称" not in info or not info["股票简称"]:
                    info["股票简称"] = fund_row.iloc[0]['基金简称']
        except:
            pass
    else:
        try:
            # 股票额外信息：行业、总股本等
            stock_info_df = ak.stock_individual_info_em(symbol=clean_code)
            if stock_info_df is not None and not stock_info_df.empty:
                for _, row in stock_info_df.iterrows():
                    item = row['item']
                    val = row['value']
                    # 如果 efinance 没有提供或者值为空，则使用 akshare 的
                    if item not in info or info[item] is None or info[item] == '':
                        info[item] = val
        except:
            pass

    if not info:
        return {}

    # 3. 统一格式化数值并处理 numpy 类型
    def convert_val(v):
        if v is None:
            return ""
        if hasattr(v, 'item'): # numpy scalar
            v = v.item()
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            if np.isnan(v):
                return ""
            return float(v)
        if isinstance(v, (np.ndarray, list)):
            return [convert_val(x) for x in v]
        return str(v)

    clean_info = {}
    for key, val in info.items():
        v = convert_val(val)
        
        if key in ['总市值', '流通市值', '总股本', '流通股本', '成交额']:
            try:
                num_val = float(v)
                if num_val >= 100000000:
                    clean_info[key] = f"{num_val / 100000000:.2f}亿"
                elif num_val >= 10000:
                    clean_info[key] = f"{num_val / 10000:.2f}万"
                else:
                    clean_info[key] = str(num_val)
            except:
                clean_info[key] = str(v)
        elif key in ['最新价', '最高', '最低', '今开', '昨收']:
            # 保留数字或转为字符串
            clean_info[key] = v
        else:
            clean_info[key] = str(v)
                
    return clean_info

@app.get("/api/stock/{code}/fundamentals")
async def get_stock_fundamentals(code: str):
    clean_code = code.split('.')[-1] if '.' in code else code
    try:
        info = get_fundamentals_cached(clean_code)
        return info
    except Exception as e:
        # Fallback if EM fails, try another source or return empty
        return {"error": str(e)}

@app.get("/api/search")
async def search_stocks(q: str):
    if not q:
        return []
    
    stocks = get_stock_list()
    results = []
    q = q.lower()
    
    for s in stocks:
        code = s['代码']
        name = s['名称']
        if q in code.lower() or q in name.lower():
            results.append({
                "value": code,
                "label": f"{code} - {name}"
            })
            if len(results) >= 10:  # Limit results
                break
                
    return results

@cache.memoize(expire=3600)
def fetch_history_data_cached(code: str, source: str, fetch_start: str, req_end: str):
    clean_code = code.split('.')[-1] if '.' in code else code
    if source == "baostock":
        # Login to baostock
        lg = bs.login()
        if lg.error_code != '0':
            return None
        
        try:
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                start_date=fetch_start,
                end_date=req_end,
                frequency="d",
                adjustflag="3" 
            )
            
            if rs.error_code != '0':
                return None
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
                
            if not data_list:
                return None
                
            df = pd.DataFrame(data_list, columns=rs.fields)
            return df
        finally:
            bs.logout()
    
    elif source == "ths":
        # Using akshare for THS/EastMoney data
        # Detect if it's an ETF
        is_etf = clean_code.startswith(('51', '56', '58', '15', '16'))
        prefix = "sh" if clean_code.startswith(('6', '9', '5')) else "sz"
        ak_code = f"{prefix}{clean_code}"
        
        df = None
        
        if is_etf:
            # Try Sina ETF source first (stable)
            for attempt in range(2):
                try:
                    df = ak.fund_etf_hist_sina(symbol=ak_code)
                    if df is not None and not df.empty:
                        df['date'] = df['date'].astype(str)
                        df = df[(df['date'] >= fetch_start) & (df['date'] <= req_end)]
                        break
                except:
                    time.sleep(1)
            
            # Fallback to EM ETF source
            if df is None or df.empty:
                for attempt in range(2):
                    try:
                        df = ak.fund_etf_hist_em(
                            symbol=clean_code, 
                            period="daily", 
                            start_date=fetch_start.replace('-', ''), 
                            end_date=req_end.replace('-', ''), 
                            adjust=""
                        )
                        if df is not None and not df.empty:
                            break
                    except:
                        time.sleep(1)
        else:
            # Stock logic
            for attempt in range(2):
                try:
                    df = ak.stock_zh_a_daily(
                        symbol=ak_code, 
                        start_date=fetch_start.replace('-', ''), 
                        end_date=req_end.replace('-', ''), 
                        adjust=""
                    )
                    if df is not None and not df.empty:
                        df = df.rename(columns={"turnover": "turn"})
                        break
                except:
                    time.sleep(1)
            
            if df is None or df.empty:
                for attempt in range(2):
                    try:
                        df = ak.stock_zh_a_hist(
                            symbol=clean_code, 
                            period="daily", 
                            start_date=fetch_start.replace('-', ''), 
                            end_date=req_end.replace('-', ''), 
                            adjust=""
                        )
                        if df is not None and not df.empty:
                            break
                    except:
                        time.sleep(1)
        
        # Fallback to efinance
        if df is None or df.empty:
            try:
                df = ef.stock.get_quote_history(
                    clean_code, 
                    beg=fetch_start.replace('-', ''), 
                    end=req_end.replace('-', '')
                )
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        "日期": "date", "开盘": "open", "收盘": "close",
                        "最高": "high", "最低": "low", "成交量": "volume", "换手率": "turn"
                    })
            except:
                pass

        if df is not None and not df.empty:
            # Standardize columns
            rename_map = {
                "日期": "date", "开盘": "open", "收盘": "close", 
                "最高": "high", "最低": "low", "成交量": "volume", "换手率": "turn",
                "turnover": "turn"
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            df['date'] = df['date'].astype(str)
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            if 'turn' not in df.columns:
                df['turn'] = 1.0 
            df['turn'] = pd.to_numeric(df['turn'], errors='coerce').fillna(1.0)
            if df['turn'].max() < 1.0 and df['turn'].max() > 0: 
                df['turn'] = df['turn'] * 100
            return df
            
    return None

@app.get("/api/stock/{code}")
async def get_stock_data(
    code: str, 
    source: str = "baostock",
    algorithm: str = "triangular", 
    decay: float = 1.0, 
    lookback: int = 250,
    use_turnover: bool = True,
    decay_factor: float = 1.0,
    peakUpperPercent: float = 10.0,
    peakLowerPercent: float = 10.0,
    showPeakArea: bool = True,
    longTermDays: int = 100,
    mediumTermDays: int = 10,
    shortTermDays: int = 2,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_all_distributions: bool = False
):
    settings = ChipSettings(
        algorithm=algorithm, 
        decay=decay, 
        lookback=lookback,
        use_turnover=use_turnover,
        decay_factor=decay_factor,
        peakUpperPercent=peakUpperPercent,
        peakLowerPercent=peakLowerPercent,
        showPeakArea=showPeakArea,
        longTermDays=longTermDays,
        mediumTermDays=mediumTermDays,
        shortTermDays=shortTermDays
    )
    
    # Default date range if not provided
    req_end = end_date or datetime.date.today().strftime('%Y-%m-%d')
    req_start = start_date or (datetime.date.today() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Fetch additional historical data for stabilization
    fetch_start = (datetime.datetime.strptime(req_start, '%Y-%m-%d') - datetime.timedelta(days=3*365)).strftime('%Y-%m-%d')
    
    try:
        result_df = fetch_history_data_cached(code, source, fetch_start, req_end)
        
        if result_df is None or result_df.empty:
            raise HTTPException(status_code=404, detail="No data found for the given stock code.")

        # Calculate profit ratio and distribution
        result_df, last_distribution, range_distribution, all_distributions, summary_stats, all_summary_stats = calculate_advanced_chips(
            result_df, settings, include_all_distributions
        )
        
        # Filter data based on user's requested date range for display
        mask = (result_df['date'] >= req_start) & (result_df['date'] <= req_end)
        display_df = result_df[mask]
        
        if len(display_df) < 2:
            display_df = result_df.tail(min(len(result_df), 260))
            
        display_data = display_df.to_dict(orient="records")
        
        response = {
            "history": display_data,
            "distribution": last_distribution,
            "range_distribution": range_distribution,
            "summary_stats": summary_stats
        }
        
        if include_all_distributions:
            response["all_distributions"] = all_distributions
            response["all_summary_stats"] = all_summary_stats
            
        return response
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    # Use provided chip settings or defaults
    settings = req.chip_settings or ChipSettings()
    
    # Default date range
    req_end = req.end_date or datetime.date.today().strftime('%Y-%m-%d')
    req_start = req.start_date or (datetime.date.today() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Fetch extra data for chip stabilization (3 years)
    fetch_start = (datetime.datetime.strptime(req_start, '%Y-%m-%d') - datetime.timedelta(days=3*365)).strftime('%Y-%m-%d')
    
    try:
        df = fetch_history_data_cached(req.code, req.source, fetch_start, req_end)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data found for the given stock code.")
        
        # Calculate profit ratios
        df, _, _, _, _, _ = calculate_advanced_chips(df, settings, include_all_distributions=False)
        
        # Filter for the backtest period
        mask = (df['date'] >= req_start) & (df['date'] <= req_end)
        test_df = df[mask].copy()
        
        if test_df.empty:
             raise HTTPException(status_code=400, detail="No data available for the selected date range.")
        
        # Simulation
        initial_cash = 100000.0
        cash = initial_cash
        shares = 0.0
        trades = []
        yield_curve = []
        current_cost_basis = 0.0
        
        # 策略买入配置
        if req.strategy_type == "buy_and_hold":
            # 只买不卖策略：单笔固定 1000 元，且不设总份数限制（允许无限买入）
            buy_unit_cash = 1000.0
        else:
            # 其他策略：每份买入资金设为初始资金的 10%
            buy_unit_cash = initial_cash * 0.1 
        
        # Final price of the backtest period
        final_price = float(test_df.iloc[-1]['close'])
        
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            ratio = row['profit_ratio']
            price = float(row['close'])
            date = str(row['date'])
            
            # Buy logic
            should_buy = False
            if req.strategy_type == "breakout":
                if ratio >= req.buy_threshold:
                    should_buy = True
            elif req.strategy_type == "buy_and_hold":
                # 只买不卖：只要获利比例低于阀值，就买入
                if ratio <= req.buy_threshold:
                    should_buy = True
            elif req.strategy_type == "mean_reversion":
                if ratio <= req.buy_threshold:
                    should_buy = True
            
            # 执行买入
            # 只买不卖策略不限制现金（无限买入），其他策略需有现金才买入
            can_buy = (req.strategy_type == "buy_and_hold") or (cash > 0)
            
            if should_buy and can_buy:
                if req.strategy_type == "buy_and_hold":
                    # 只买不卖：固定 1000 元，即使现金不足也买入（允许负债/无限份额）
                    amount_to_spend = buy_unit_cash
                else:
                    # 其他策略：不能超过剩余现金
                    amount_to_spend = min(cash, buy_unit_cash)
                    if cash < buy_unit_cash * 1.1: # 如果剩余现金不足 1.1 份，就一次性买完
                        amount_to_spend = cash
                
                shares_to_buy = amount_to_spend / price
                cash -= amount_to_spend
                shares += shares_to_buy
                current_cost_basis += amount_to_spend
                
                trade_data = {
                    "type": "buy",
                    "date": str(date),
                    "price": price,
                    "ratio": float(ratio),
                    "amount": round(amount_to_spend, 2)
                }
                
                # 为“只买不卖”策略的每一笔计算收益率（相对于回测结束时的价格）
                if req.strategy_type == "buy_and_hold":
                    trade_profit_pct = (final_price - price) / price
                    trade_data["profit"] = round(trade_profit_pct * 100, 2)
                
                trades.append(trade_data)
            
            # Sell logic
            should_sell = False
            if shares > 0:
                if req.strategy_type == "breakout":
                    if ratio <= req.sell_threshold:
                        should_sell = True
                elif req.strategy_type == "buy_and_hold":
                    # 只买不卖策略不主动止盈止损
                    should_sell = False
                elif req.strategy_type == "mean_reversion":
                    if ratio >= req.sell_threshold:
                        should_sell = True
                
                # 最后一天强制卖出（非只买不卖策略），用于结算
                is_last_day = (i == len(test_df) - 1)
                if is_last_day and req.strategy_type != "buy_and_hold":
                    should_sell = True
                
                if should_sell:
                    sell_revenue = shares * price
                    trade_profit_pct = (sell_revenue - current_cost_basis) / current_cost_basis if current_cost_basis > 0 else 0
                    
                    trades.append({
                        "type": "sell",
                        "date": str(date),
                        "price": price,
                        "ratio": float(ratio),
                        "profit": round(trade_profit_pct * 100, 2),
                        "amount": round(sell_revenue, 2)
                    })
                    
                    cash += sell_revenue
                    shares = 0
                    current_cost_basis = 0.0
            
            # Update yield curve (Daily total value)
            current_total_value = cash + (shares * price)
            
            if req.strategy_type == "buy_and_hold":
                # 只买不卖：累计收益率 = (当前持仓股数 * 走势图最后一天价格 - 累计投入) / 累计投入
                if current_cost_basis > 0:
                    # 使用走势图最后一天价格 final_price 计算收益
                    current_market_value_at_final = shares * final_price
                    # print(f"kim name {req.code} current_cost_basis ¥={current_cost_basis} current_market_value_at_final {current_market_value_at_final}");
                    cumulative_yield = (current_market_value_at_final - current_cost_basis) / current_cost_basis
                else:
                    cumulative_yield = 0.0
            else:
                # 其他策略：收益率 = (当前总资产 - 初始资金) / 初始资金
                cumulative_yield = (current_total_value - initial_cash) / initial_cash
            
            yield_curve.append({
                "date": date,
                "yield": round(cumulative_yield * 100, 2),
                "profit_ratio": float(ratio),
                "cash": round(cash, 2),
                "shares": round(shares, 4),
                "price": price
            })
        
        # Calculate maximum drawdown
        max_drawdown = 0
        peak = -999999.0
        for item in yield_curve:
            y = item['yield']
            if y > peak:
                peak = y
            drawdown = peak - y
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Debug trade count
        with open("backtest_debug.log", "a") as f:
            f.write(f"{datetime.datetime.now()}: {req.code} backtest done. Trades: {len(trades)}, Buys: {len([t for t in trades if t['type'] == 'buy'])}, Sells: {len([t for t in trades if t['type'] == 'sell'])}\n")

        # Final yield calculation logic
        if req.strategy_type == "buy_and_hold":
            # 最后一天收盘持仓价即为 final_price, 持仓份数为 shares, 累计投入为 current_cost_basis
            if current_cost_basis > 0:
                final_market_value = shares * final_price
                final_yield = (final_market_value - current_cost_basis) / current_cost_basis
            else:
                final_yield = 0.0
        else:
            final_yield = cumulative_yield

        # Win rate calculation
        if req.strategy_type == "buy_and_hold":
            # 只买不卖：胜率 = 盈利的买入笔数 / 总买入笔数
            buy_trades = [t for t in trades if t['type'] == 'buy']
            win_rate = round(len([t for t in buy_trades if t.get('profit', 0) > 0]) / max(1, len(buy_trades)) * 100, 2)
        else:
            # 其他策略：胜率 = 盈利的卖出笔数 / 总卖出笔数
            sell_trades = [t for t in trades if t['type'] == 'sell']
            win_rate = round(len([t for t in sell_trades if t.get('profit', 0) > 0]) / max(1, len(sell_trades)) * 100, 2)

        return {
            "code": req.code,
            "total_yield": round(final_yield * 100, 2),
            "trades": trades,
            "yield_curve": yield_curve,
            "max_drawdown": round(max_drawdown, 2),
            "buy_count": len([t for t in trades if t['type'] == 'buy']),
            "sell_count": len([t for t in trades if t['type'] == 'sell']),
            "trade_count": len(trades),
            "win_rate": win_rate
        }
        
    except Exception as e:
        print(f"Backtest error: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
