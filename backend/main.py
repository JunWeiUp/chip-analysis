from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import baostock as bs
import pandas as pd
import datetime
import akshare as ak
import efinance as ef
import time
from diskcache import Cache
from typing import Optional
import numpy as np
from pydantic import BaseModel

# 外部调用日志包装器
def log_external_api(api_name: str, func, *args, **kwargs):
    start_time = time.time()
    print(f"[{datetime.datetime.now()}] >>> EXTERNAL CALL START: {api_name}")
    try:
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"[{datetime.datetime.now()}] <<< EXTERNAL CALL END: {api_name} | Duration: {duration:.2f}s | Status: Success")
        return result
    except Exception as e:
        duration = time.time() - start_time
        print(f"[{datetime.datetime.now()}] <<< EXTERNAL CALL END: {api_name} | Duration: {duration:.2f}s | Status: Error | Msg: {str(e)}")
        raise e

app = FastAPI()

# Cache for stock list and data
cache = Cache('./cache')
@cache.memoize(expire=3600)
def get_stock_list():
    try:
        # 使用更稳定的接口获取股票列表
        df = log_external_api("akshare.stock_info_a_code_name", ak.stock_info_a_code_name)
        if df is not None and not df.empty:
            # 该接口返回的列名是 'code' 和 'name'
            return df.rename(columns={'code': '代码', 'name': '名称'})[['代码', '名称']].to_dict('records')
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

    # Calculate additional technical indicators
    # 1. RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # 2. MACD (12, 26, 9)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Replace any NaN in the whole dataframe before returning
    df = df.fillna(0)
    return df, last_distribution, range_dist_list, all_distributions, summary_stats, all_summary_stats

@cache.memoize(expire=3600)
def get_fundamentals_cached(clean_code: str):
    info = {}
    
    # 1. 优先尝试从 efinance 获取最新行情（支持股票和 ETF）
    try:
        quote_df = log_external_api(f"efinance.stock.get_latest_quote({clean_code})", ef.stock.get_latest_quote, clean_code)
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
            df_name = log_external_api("akshare.fund_name_em", ak.fund_name_em)
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
            stock_info_df = log_external_api(f"akshare.stock_individual_info_em({clean_code})", ak.stock_individual_info_em, symbol=clean_code)
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
                
    # 返回结构化数据以匹配前端接口
    important_keys = ["最新价", "涨跌幅", "换手率", "成交额", "行业"]
    groups = {
        "基本行情": ["最新价", "涨跌幅", "最高", "最低", "今开", "昨收", "成交量", "成交额", "换手率"],
        "市值股本": ["总市值", "流通市值", "总股本", "流通股本"],
        "公司信息": ["行业", "上市时间", "股票简称", "股票代码"]
    }
    
    # 过滤掉不存在的 key
    final_groups = {}
    for g_name, g_keys in groups.items():
        valid_g_keys = [k for k in g_keys if k in clean_info]
        if valid_g_keys:
            final_groups[g_name] = valid_g_keys

    return {
        "info": clean_info,
        "groups": final_groups,
        "important_keys": [k for k in important_keys if k in clean_info]
    }

@app.get("/api/stock/{code}/fundamentals")
async def get_stock_fundamentals(code: str):
    clean_code = code.split('.')[-1] if '.' in code else code
    try:
        info = get_fundamentals_cached(clean_code)
        return info
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/search")
async def search_stocks(q: str):
    if not q:
        return []
    
    stocks = get_stock_list()
    results = []
    q_lower = q.lower()
    
    # 提取纯代码部分进行代码搜索，兼容 SH.600000, sz000001 等格式
    clean_q = q_lower
    for prefix in ['sh.', 'sz.', 'sh', 'sz']:
        if q_lower.startswith(prefix):
            potential_clean = q_lower[len(prefix):]
            if potential_clean: # 确保不是只输了前缀
                clean_q = potential_clean
            break
    
    for s in stocks:
        code = s['代码']
        name = s['名称']
        # 匹配逻辑：
        # 1. 如果去掉了前缀的 clean_q 在股票代码中
        # 2. 或者原始输入的 q_lower 在名称中
        if clean_q in code.lower() or q_lower in name.lower():
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
    df = None
    
    if source == "baostock":
        # Auto-prefix for baostock if missing
        lower_code = code.lower()
        if not (lower_code.startswith('sh.') or lower_code.startswith('sz.')):
            prefix = "sh" if clean_code.startswith(('6', '9', '5')) else "sz"
            code = f"{prefix}.{clean_code}"
            
        # Login to baostock
        lg = log_external_api("baostock.login", bs.login)
        if lg.error_code != '0':
            return None
        
        try:
            rs = log_external_api(f"baostock.query_history_k_data_plus({code})", bs.query_history_k_data_plus,
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
        finally:
            log_external_api("baostock.logout", bs.logout)
    
    elif source == "efinance":
        try:
            df = log_external_api(f"efinance.stock.get_quote_history({clean_code})", ef.stock.get_quote_history,
                clean_code, 
                beg=fetch_start.replace('-', ''), 
                end=req_end.replace('-', '')
            )
        except Exception as e:
            print(f"efinance error: {e}")
            return None

    elif source == "ths":
        # Using akshare for THS/EastMoney data
        # Detect if it's an ETF
        is_etf = clean_code.startswith(('51', '56', '58', '15', '16'))
        prefix = "sh" if clean_code.startswith(('6', '9', '5')) else "sz"
        ak_code = f"{prefix}{clean_code}"
        
        if is_etf:
            # Try Sina ETF source first (stable)
            for _ in range(2):
                try:
                    df = log_external_api(f"akshare.fund_etf_hist_sina({ak_code})", ak.fund_etf_hist_sina, symbol=ak_code)
                    if df is not None and not df.empty:
                        df['date'] = df['date'].astype(str)
                        df = df[(df['date'] >= fetch_start) & (df['date'] <= req_end)]
                        break
                except:
                    time.sleep(1)
            
            # Fallback to EM ETF source
            if df is None or df.empty:
                for _ in range(2):
                    try:
                        df = log_external_api(f"akshare.fund_etf_hist_em({clean_code})", ak.fund_etf_hist_em,
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
            for _ in range(2):
                try:
                    df = log_external_api(f"akshare.stock_zh_a_daily({ak_code})", ak.stock_zh_a_daily,
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
                for _ in range(2):
                    try:
                        df = log_external_api(f"akshare.stock_zh_a_hist({clean_code})", ak.stock_zh_a_hist,
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
                df = log_external_api(f"efinance.stock.get_quote_history({clean_code})", ef.stock.get_quote_history,
                    clean_code, 
                    beg=fetch_start.replace('-', ''), 
                    end=req_end.replace('-', '')
                )
            except:
                pass

    # Standardize and return if data exists
    if df is not None and not df.empty:
        # Standardize columns
        rename_map = {
            "日期": "date", "开盘": "open", "收盘": "close", 
            "最高": "high", "最低": "low", "成交量": "volume", "换手率": "turn",
            "turnover": "turn", "pctChg": "pct_chg"
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

@cache.memoize(expire=600)
def get_stock_data_internal(
    code: str, 
    source: str,
    algorithm: str, 
    decay: float, 
    lookback: int,
    use_turnover: bool,
    decay_factor: float,
    peakUpperPercent: float,
    peakLowerPercent: float,
    showPeakArea: bool,
    longTermDays: int,
    mediumTermDays: int,
    shortTermDays: int,
    start_date: Optional[str],
    end_date: Optional[str],
    include_all_distributions: bool
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
    
    result_df = fetch_history_data_cached(code, source, fetch_start, req_end)
    
    if result_df is None or result_df.empty:
        return None

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
    try:
        response = get_stock_data_internal(
            code, source, algorithm, decay, lookback, use_turnover, decay_factor,
            peakUpperPercent, peakLowerPercent, showPeakArea,
            longTermDays, mediumTermDays, shortTermDays,
            start_date, end_date, include_all_distributions
        )
        
        if response is None:
            raise HTTPException(status_code=404, detail="No data found for the given stock code.")
            
        return response
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@cache.memoize(expire=600)
def run_backtest_internal(req_dict: dict):
    code = req_dict['code']
    buy_threshold = req_dict['buy_threshold']
    sell_threshold = req_dict['sell_threshold']
    source = req_dict.get('source', 'baostock')
    start_date = req_dict.get('start_date')
    end_date = req_dict.get('end_date')
    strategy_type = req_dict.get('strategy_type', 'mean_reversion')
    
    chip_settings_dict = req_dict.get('chip_settings')
    settings = ChipSettings(**chip_settings_dict) if chip_settings_dict else ChipSettings()
    
    req_end = end_date or datetime.date.today().strftime('%Y-%m-%d')
    req_start = start_date or (datetime.date.today() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_start = (datetime.datetime.strptime(req_start, '%Y-%m-%d') - datetime.timedelta(days=3*365)).strftime('%Y-%m-%d')
    
    df = fetch_history_data_cached(code, source, fetch_start, req_end)
    if df is None or df.empty: return None
    
    df, _, _, _, _, _ = calculate_advanced_chips(df, settings, include_all_distributions=False)
    mask = (df['date'] >= req_start) & (df['date'] <= req_end)
    test_df = df[mask].copy()
    if test_df.empty: return {"error": "No data available for the selected date range."}
    
    initial_cash, cash, shares, trades, yield_curve, current_cost_basis = 100000.0, 100000.0, 0.0, [], [], 0.0
    buy_unit_cash = 1000.0 if strategy_type == "buy_and_hold" else initial_cash * 0.1 
    final_price = float(test_df.iloc[-1]['close'])
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        ratio, price, date = row['profit_ratio'], float(row['close']), str(row['date'])
        
        should_buy = (strategy_type == "breakout" and ratio >= buy_threshold) or \
                     (strategy_type == "buy_and_hold" and ratio <= buy_threshold) or \
                     (strategy_type == "mean_reversion" and ratio <= buy_threshold)
        
        if should_buy and (strategy_type == "buy_and_hold" or cash > 0):
            amount_to_spend = buy_unit_cash if strategy_type == "buy_and_hold" else min(cash, buy_unit_cash)
            if strategy_type != "buy_and_hold" and cash < buy_unit_cash * 1.1: amount_to_spend = cash
            
            shares_to_buy = amount_to_spend / price
            cash -= amount_to_spend
            shares += shares_to_buy
            current_cost_basis += amount_to_spend
            
            trade_data = {"type": "buy", "date": date, "price": price, "ratio": float(ratio), "amount": round(amount_to_spend, 2)}
            if strategy_type == "buy_and_hold":
                trade_data["profit"] = round((final_price - price) / price * 100, 2)
            trades.append(trade_data)
        
        should_sell = False
        if shares > 0:
            if strategy_type == "breakout": should_sell = ratio <= sell_threshold
            elif strategy_type == "mean_reversion": should_sell = ratio >= sell_threshold
            
            if (i == len(test_df) - 1) and strategy_type != "buy_and_hold": should_sell = True
            
            if should_sell:
                sell_revenue = shares * price
                trade_profit_pct = (sell_revenue - current_cost_basis) / current_cost_basis if current_cost_basis > 0 else 0
                trades.append({"type": "sell", "date": date, "price": price, "ratio": float(ratio), "profit": round(trade_profit_pct * 100, 2), "amount": round(sell_revenue, 2)})
                cash += sell_revenue
                shares, current_cost_basis = 0, 0.0
        
        current_total_value = cash + (shares * price)
        if strategy_type == "buy_and_hold":
            cumulative_yield = (shares * final_price - current_cost_basis) / current_cost_basis if current_cost_basis > 0 else 0.0
        else:
            cumulative_yield = (current_total_value - initial_cash) / initial_cash
            
        yield_curve.append({"date": date, "yield": round(cumulative_yield * 100, 2), "profit_ratio": float(ratio), "cash": round(cash, 2), "shares": round(shares, 4), "price": price})
    
    max_drawdown, peak = 0, -999999.0
    for item in yield_curve:
        peak = max(peak, item['yield'])
        max_drawdown = max(max_drawdown, peak - item['yield'])
    
    if strategy_type == "buy_and_hold":
        final_yield = (shares * final_price - current_cost_basis) / current_cost_basis if current_cost_basis > 0 else 0.0
        buy_trades = [t for t in trades if t['type'] == 'buy']
        win_rate = round(len([t for t in buy_trades if t.get('profit', 0) > 0]) / max(1, len(buy_trades)) * 100, 2)
    else:
        final_yield = cumulative_yield
        sell_trades = [t for t in trades if t['type'] == 'sell']
        win_rate = round(len([t for t in sell_trades if t.get('profit', 0) > 0]) / max(1, len(sell_trades)) * 100, 2)

    return {"code": code, "total_yield": round(final_yield * 100, 2), "trades": trades, "yield_curve": yield_curve, "max_drawdown": round(max_drawdown, 2), "buy_count": len([t for t in trades if t['type'] == 'buy']), "sell_count": len([t for t in trades if t['type'] == 'sell']), "trade_count": len(trades), "win_rate": win_rate}

@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    try:
        result = run_backtest_internal(req.model_dump())
        if result is None: raise HTTPException(status_code=404, detail="No data found for the given stock code.")
        if "error" in result: raise HTTPException(status_code=400, detail=result["error"])
        with open("backtest_debug.log", "a") as f: f.write(f"{datetime.datetime.now()}: {req.code} backtest done. Trades: {result['trade_count']}\n")
        return result
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
