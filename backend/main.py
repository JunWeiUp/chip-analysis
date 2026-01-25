from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import baostock as bs
import pandas as pd
import datetime
import akshare as ak
import efinance as ef
import time
from typing import List, Dict, Optional
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Cache for stock list
stock_list_cache = []

def get_stock_list():
    global stock_list_cache
    if not stock_list_cache:
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                stock_list_cache = df[['代码', '名称']].to_dict('records')
        except Exception as e:
            print(f"Error fetching stock list: {e}")
    return stock_list_cache

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

@app.get("/api/stock/{code}/fundamentals")
async def get_stock_fundamentals(code: str):
    clean_code = code.split('.')[-1] if '.' in code else code
    try:
        # Get individual stock info from EastMoney
        df = ak.stock_individual_info_em(symbol=clean_code)
        if df is None or df.empty:
            return {}
        
        # Convert dataframe to a more usable dict
        info = {}
        for _, row in df.iterrows():
            info[row['item']] = row['value']
            
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
        if source == "baostock":
            # Login to baostock
            lg = bs.login()
            if lg.error_code != '0':
                raise HTTPException(status_code=500, detail=f"BaoStock login failed: {lg.error_msg}")
            
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
                    raise HTTPException(status_code=404, detail=f"Query failed: {rs.error_msg}")
                
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                    
                if not data_list:
                    raise HTTPException(status_code=404, detail="No data found for the given stock code.")
                    
                result_df = pd.DataFrame(data_list, columns=rs.fields)
            finally:
                bs.logout()
        
        elif source == "ths":
            # Using akshare for THS/EastMoney data
            clean_code = code.split('.')[-1] if '.' in code else code
            
            # Detect if it's an ETF
            is_etf = clean_code.startswith(('51', '56', '58', '15', '16'))
            prefix = "sh" if clean_code.startswith(('6', '9', '5')) else "sz"
            ak_code = f"{prefix}{clean_code}"
            
            df = None
            last_error = ""
            
            if is_etf:
                # Try Sina ETF source first (stable)
                for attempt in range(2):
                    try:
                        # Sina fund_etf_hist_sina returns full history
                        df = ak.fund_etf_hist_sina(symbol=ak_code)
                        if df is not None and not df.empty:
                            # Filter by date manually since this API doesn't support start/end
                            df['date'] = df['date'].astype(str)
                            df = df[(df['date'] >= fetch_start) & (df['date'] <= req_end)]
                            break
                    except Exception as e:
                        last_error = str(e)
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
                        except Exception as e:
                            last_error = f"Sina ETF error: {last_error}; EM ETF error: {str(e)}"
                            time.sleep(1)
            else:
                # Stock logic
                # Try akshare stock_zh_a_daily (Sina source) which is often more stable
                for attempt in range(2):
                    try:
                        df = ak.stock_zh_a_daily(
                            symbol=ak_code, 
                            start_date=fetch_start.replace('-', ''), 
                            end_date=req_end.replace('-', ''), 
                            adjust=""
                        )
                        if df is not None and not df.empty:
                            # stock_zh_a_daily already has standard names: date, open, high, low, close, volume, turnover
                            df = df.rename(columns={"turnover": "turn"})
                            break
                    except Exception as e:
                        last_error = str(e)
                        time.sleep(1)
                
                # Fallback to akshare stock_zh_a_hist (EastMoney source) if Sina fails
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
                        except Exception as e:
                            last_error = f"Sina error: {last_error}; EastMoney error: {str(e)}"
                            time.sleep(1)
            
            # Fallback to efinance if both akshare methods fail (works for both stocks and funds)
            if df is None or df.empty:
                try:
                    df = ef.stock.get_quote_history(
                        clean_code, 
                        beg=fetch_start.replace('-', ''), 
                        end=req_end.replace('-', '')
                    )
                    if df is not None and not df.empty:
                        df = df.rename(columns={
                            "日期": "date",
                            "开盘": "open",
                            "收盘": "close",
                            "最高": "high",
                            "最低": "low",
                            "成交量": "volume",
                            "换手率": "turn"
                        })
                except Exception as e:
                    last_error = f"{last_error}; efinance error: {str(e)}"
                    time.sleep(1)

            if df is None or df.empty:
                raise HTTPException(status_code=500, detail=f"THS Data fetch failed: {last_error}")
            
            # Standardize columns for any source
            rename_map = {
                "日期": "date", "开盘": "open", "收盘": "close", 
                "最高": "high", "最低": "low", "成交量": "volume", "换手率": "turn",
                "turnover": "turn"
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            
            # Ensure date is string format
            df['date'] = df['date'].astype(str)
            
            # Unify volume and turn
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Ensure turn exists (ETFs from Sina might miss it)
            if 'turn' not in df.columns:
                df['turn'] = 1.0 
            
            df['turn'] = pd.to_numeric(df['turn'], errors='coerce').fillna(1.0)
            
            # Unify turnover format: Sina is ratio (0.0026), EM is percentage (0.26)
            if df['turn'].max() < 1.0 and df['turn'].max() > 0: 
                df['turn'] = df['turn'] * 100
                
            result_df = df
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported data source: {source}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
