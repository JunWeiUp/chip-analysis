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
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
            
    # 3. 如果行业信息依然缺失，尝试从 Baostock 获取（作为最后的兜底）
    if not is_etf and ("行业" not in info or not info["行业"]):
        try:
            import baostock as bs
            bs.login()
            rs = bs.query_stock_industry(code=f"sh.{clean_code}" if clean_code.startswith('6') else f"sz.{clean_code}")
            if rs.error_code == '0' and rs.next():
                row_data = rs.get_row_data()
                # Baostock return list: updateDate, code, code_name, industry, industryClassification
                if len(row_data) >= 4:
                    info["行业"] = row_data[3]
            bs.logout()
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
                
    # 返回扁平化数据供内部使用
    return clean_info

@app.get("/api/stock/{code}/fundamentals")
async def get_stock_fundamentals(code: str):
    clean_code = code.split('.')[-1] if '.' in code else code
    try:
        clean_info = get_fundamentals_cached(clean_code)
        if not clean_info:
            return {}
            
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
def get_industry_stocks(industry_name: str, stock_code: str = None):
    if not industry_name or industry_name == "ETF/基金":
        return []
        
    # 尝试多种名称变体进行匹配
    names_to_try = [industry_name]
    if "行业" in industry_name:
        names_to_try.append(industry_name.replace("行业", ""))
    
    # 1. Try EM interface (original)
    for name in names_to_try:
        try:
            df = log_external_api(f"akshare.stock_board_industry_cons_em({name})", 
                                ak.stock_board_industry_cons_em, symbol=name)
            if df is not None and not df.empty:
                return df[['代码', '名称']].to_dict(orient="records")
        except Exception as e:
            print(f"Error fetching industry stocks from EM for {name}: {e}")
        
    # 2. Fallback to Baostock if stock_code is provided
    if stock_code:
        try:
            import baostock as bs
            bs.login()
            # 获取所有股票行业信息 (Baostock 不支持按行业名称查询，只能全量获取后过滤)
            # 优化：可以使用缓存或者只查询当前股票及其行业
            rs = bs.query_stock_industry()
            industry_list = []
            while (rs.error_code == '0') & rs.next():
                industry_list.append(rs.get_row_data())
            bs.logout()
            
            if industry_list:
                df_bs = pd.DataFrame(industry_list, columns=rs.fields)
                clean_code = stock_code.split('.')[-1] if '.' in stock_code else stock_code
                
                # 寻找当前股票所属行业
                match = df_bs[df_bs['code'].str.contains(clean_code)]
                if not match.empty:
                    target_industry = match.iloc[0]['industry']
                    peers = df_bs[df_bs['industry'] == target_industry]
                    results = []
                    for _, row in peers.iterrows():
                        p_code = row['code']
                        if '.' in p_code: p_code = p_code.split('.')[-1]
                        results.append({'代码': p_code, '名称': row['code_name']})
                    return results
        except Exception as e:
            print(f"Error fetching industry stocks from Baostock: {e}")
            
    return []

@app.get("/api/stock/{code}/sector/comparison")
async def get_sector_comparison(code: str):
    clean_code = code.split('.')[-1] if '.' in code else code
    try:
        # 1. Get current stock's industry
        fundamentals = get_fundamentals_cached(clean_code)
        industry = fundamentals.get("行业")
        
        # Fallback: if no industry, try to find it from board list
        if not industry or industry == "ETF/基金":
            try:
                # 尝试从个股所属板块信息中提取行业
                gn_df = log_external_api(f"akshare.stock_board_concept_cons_em({clean_code})", 
                                       ak.stock_board_concept_cons_em, symbol=clean_code)
                if gn_df is not None and not gn_df.empty:
                    # 优先寻找包含“行业”字样的板块
                    industry_boards = gn_df[gn_df['板块名称'].str.contains('行业', na=False)]
                    if not industry_boards.empty:
                        industry = industry_boards.iloc[0]['板块名称']
                    else:
                        industry = gn_df.iloc[0]['板块名称']
            except:
                pass
        
        if not industry or industry == "ETF/基金":
            return {
                "industry": "未知板块",
                "comparison": [],
                "message": "未找到所属行业板块，暂无法进行对比分析。"
            }

        # 2. Get other stocks in the same industry
        peer_stocks = get_industry_stocks(industry, clean_code)
        if not peer_stocks:
            return {
                "industry": industry,
                "comparison": [],
                "message": f"未找到 {industry} 板块的成份股数据。"
            }
        
        # 3. Get basic chip stats for peers
        comparison_data = []
        # 限制对比数量，避免过慢
        for peer in peer_stocks[:8]:
            p_code = peer['代码']
            if p_code == clean_code: continue
            
            try:
                # 优化：使用较短的 lookback 来加速计算
                p_data = get_stock_data_internal(
                    p_code, "efinance", "triangular", 1.0, 120, True, 1.0, 
                    10.0, 10.0, True, 60, 10, 2, None, None, False
                )
                if p_data and "summary_stats" in p_data:
                    comparison_data.append({
                        "code": p_code,
                        "name": peer['名称'],
                        "stats": p_data["summary_stats"]
                    })
            except:
                continue
        
        # 始终包含当前股票自身的数据进行对比
        try:
            current_data = get_stock_data_internal(
                clean_code, "efinance", "triangular", 1.0, 120, True, 1.0, 
                10.0, 10.0, True, 60, 10, 2, None, None, False
            )
            if current_data and "summary_stats" in current_data:
                comparison_data.insert(0, {
                    "code": clean_code,
                    "name": fundamentals.get("股票简称", "当前股票"),
                    "stats": current_data["summary_stats"],
                    "is_current": True
                })
        except:
            pass
                
        return {
            "industry": industry,
            "comparison": comparison_data
        }
    except Exception as e:
        print(f"Sector comparison error: {e}")
        return {
            "industry": "未知",
            "comparison": [],
            "error": str(e)
        }

@app.get("/api/stock/{code}/sector/money-flow")
async def get_sector_money_flow(code: str):
    clean_code = code.split('.')[-1] if '.' in code else code
    try:
        # 1. 获取个股资金流向 (优先使用 efinance get_history_bill，因为它更稳定且详细)
        stock_flow_list = []
        try:
            ef_flow = log_external_api(f"ef.stock.get_history_bill({clean_code})", ef.stock.get_history_bill, clean_code)
            if ef_flow is not None and not ef_flow.empty:
                # 统一列名映射，适配前端
                stock_flow_list = ef_flow.tail(30).to_dict(orient="records")
        except Exception as e:
            print(f"Sector money flow EF error: {e}")
            
        if not stock_flow_list:
            # Fallback to akshare
            try:
                stock_flow = log_external_api(f"akshare.stock_individual_fund_flow({clean_code})", 
                                            ak.stock_individual_fund_flow, stock=clean_code, market="sh" if clean_code.startswith('6') else "sz")
                if stock_flow is not None and not stock_flow.empty:
                    stock_flow = stock_flow.rename(columns={
                        '主力净流入-净额': '主力净流入',
                        '超大单净流入-净额': '超大单净流入',
                        '大单净流入-净额': '大单净流入',
                        '中单净流入-净额': '中单净流入',
                        '小单净流入-净额': '小单净流入'
                    })
                    stock_flow_list = stock_flow.tail(30).to_dict(orient="records")
            except Exception as e:
                print(f"Sector money flow AK error: {e}")
        
        # 2. 获取板块资金流向
        fundamentals = get_fundamentals_cached(clean_code)
        industry = fundamentals.get("行业")
        sector_flow = None
        
        # 尝试获取板块实时资金流
        try:
            # 优先使用 akshare (快)
            all_sectors_flow = log_external_api("akshare.stock_sector_fund_flow_rank", ak.stock_sector_fund_flow_rank)
            if all_sectors_flow is not None and not all_sectors_flow.empty:
                if industry and industry != "ETF/基金":
                    short_industry = industry.replace("行业", "")
                    match = all_sectors_flow[all_sectors_flow['名称'].str.contains(short_industry, na=False)]
                    if not match.empty:
                        sector_flow = match.iloc[0].to_dict()
                        sector_flow['主力净额'] = sector_flow.get('今日主力净流入-净额', 0)
        except:
            pass
            
        # 如果 akshare 失败或没匹配到，使用 efinance 获取行业历史资金流作为参考
        if (not sector_flow or '主力净额' not in sector_flow) and industry and industry != "ETF/基金":
            try:
                # 尝试用行业名称直接获取
                ef_sector_flow = log_external_api(f"ef.stock.get_history_bill({industry})", ef.stock.get_history_bill, industry)
                if ef_sector_flow is not None and not ef_sector_flow.empty:
                    latest = ef_sector_flow.iloc[-1]
                    sector_flow = {
                        "名称": industry,
                        "主力净额": float(latest.get('主力净流入', 0)),
                        "今日主力净流入-幅度": float(latest.get('主力净流入占比', 0)),
                        "涨跌幅": float(latest.get('涨跌幅', 0))
                    }
                else:
                    # 尝试去掉“行业”后缀再试一次
                    short_ind = industry.replace("行业", "")
                    if short_ind != industry:
                        ef_sector_flow = log_external_api(f"ef.stock.get_history_bill({short_ind})", ef.stock.get_history_bill, short_ind)
                        if ef_sector_flow is not None and not ef_sector_flow.empty:
                            latest = ef_sector_flow.iloc[-1]
                            sector_flow = {
                                "名称": short_ind,
                                "主力净额": float(latest.get('主力净流入', 0)),
                                "今日主力净流入-幅度": float(latest.get('主力净流入占比', 0)),
                                "涨跌幅": float(latest.get('涨跌幅', 0))
                            }
            except:
                pass

        return {
            "stock_flow": stock_flow_list,
            "sector_flow": sector_flow,
            "industry": industry
        }
    except Exception as e:
        print(f"Sector money flow error: {e}")
        return {
            "stock_flow": [],
            "sector_flow": None,
            "error": str(e)
        }

def get_single_sector_flow(sector_name):
    """获取单个板块的最新资金流向"""
    try:
        df = log_external_api(f"ef.stock.get_history_bill({sector_name})", ef.stock.get_history_bill, sector_name)
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            return {
                "板块名称": sector_name,
                "主力净额": float(latest.get('主力净流入', 0)),
                "今日主力净流入-幅度": float(latest.get('主力净流入占比', 0))
            }
    except Exception as e:
        print(f"Error fetching flow for {sector_name}: {e}")
    return None

@app.get("/api/sector/rotation")
async def get_sector_rotation():
    try:
        # 1. 获取行业板块行情数据 (优先 EM)
        df = None
        try:
            df = log_external_api("akshare.stock_board_industry_name_em", ak.stock_board_industry_name_em)
        except:
            pass
            
        # 备选：同花顺行业板块
        if df is None or df.empty:
            try:
                ths_df = log_external_api("akshare.stock_board_industry_name_ths", ak.stock_board_industry_name_ths)
                if ths_df is not None and not ths_df.empty:
                    # 转换列名以匹配后续逻辑
                    df = ths_df.rename(columns={'name': '板块名称'})
                    # THS 接口不带涨跌幅，我们需要额外获取或者只展示名称
                    try:
                        summary_df = log_external_api("akshare.stock_board_industry_summary_ths", ak.stock_board_industry_summary_ths)
                        if summary_df is not None and not summary_df.empty:
                            if '板块' in summary_df.columns:
                                summary_df = summary_df.rename(columns={'板块': '板块名称'})
                            elif 'name' in summary_df.columns:
                                summary_df = summary_df.rename(columns={'name': '板块名称'})
                            
                            df = pd.merge(df, summary_df[['板块名称', '涨跌幅']], on='板块名称', how='left')
                    except Exception as e:
                        print(f"Rotation THS merge error: {e}")
                        if '涨跌幅' not in df.columns: df['涨跌幅'] = 0
            except:
                pass

        if df is not None and not df.empty:
            # 2. 尝试获取板块资金流向
            flow_df = None
            try:
                # 优先使用 akshare 排名接口 (快)
                flow_df = log_external_api("akshare.stock_sector_fund_flow_rank", ak.stock_sector_fund_flow_rank)
                if flow_df is not None and not flow_df.empty:
                    flow_df = flow_df.rename(columns={'今日主力净流入-净额': '主力净额'})
                    df = pd.merge(df, flow_df[['名称', '主力净额', '今日主力净流入-幅度']], 
                                left_on='板块名称', right_on='名称', how='left')
            except:
                pass
            
            # 如果 akshare 失败，针对涨幅前15的板块，使用 efinance 补充资金流数据
            if '主力净额' not in df.columns or df['主力净额'].isna().all() or (df['主力净额'] == 0).all():
                # 排序获取前15个板块
                if '涨跌幅' in df.columns:
                    top_sectors = df.sort_values('涨跌幅', ascending=False).head(15)['板块名称'].tolist()
                else:
                    top_sectors = df.head(15)['板块名称'].tolist()
                
                # 并行获取资金流向
                with ThreadPoolExecutor(max_workers=5) as executor:
                    loop = asyncio.get_event_loop()
                    tasks = [loop.run_in_executor(executor, get_single_sector_flow, name) for name in top_sectors]
                    flow_results = await asyncio.gather(*tasks)
                
                # 合并结果
                flow_results = [r for r in flow_results if r is not None]
                if flow_results:
                    extra_flow_df = pd.DataFrame(flow_results)
                    # 如果之前 merge 失败了（没列），现在加上
                    if '主力净额' not in df.columns:
                        df = pd.merge(df, extra_flow_df, on='板块名称', how='left')
                    else:
                        # 如果之前 merge 了但是没数据，更新它
                        for _, row in extra_flow_df.iterrows():
                            mask = df['板块名称'] == row['板块名称']
                            df.loc[mask, '主力净额'] = row['主力净额']
                            if '今日主力净流入-幅度' in df.columns:
                                df.loc[mask, '今日主力净流入-幅度'] = row['今日主力净流入-幅度']
                            else:
                                # 如果列不存在，这里可能需要特殊处理，但通常 merge 之后列会存在
                                pass

            # 3. 清理并排序
            df['主力净额'] = pd.to_numeric(df.get('主力净额', 0), errors='coerce').fillna(0)
            df['涨跌幅'] = pd.to_numeric(df.get('涨跌幅', 0), errors='coerce').fillna(0)
            
            # 统一单位：主力净额如果是从 ef 获取的，单位是元，我们可能需要转换为“亿”或保持一致
            # akshare 的单位通常也是元或万，这里我们确保前端显示一致即可
            
            df = df.fillna(0)
            df = df.replace({np.nan: None, np.inf: 0, -np.inf: 0})
            df = df.sort_values('涨跌幅', ascending=False)
            
            return df.head(15).to_dict(orient="records")
        else:
            return []
    except Exception as e:
        print(f"Sector rotation error: {e}")
        return []

@app.get("/api/market/sentiment")
async def get_market_sentiment():
    """
    获取大盘“恐惧与贪婪”择时指数
    通过计算上证指数的获利比例来衡量大盘情绪
    """
    try:
        # 上证指数代码
        index_code = "sh000001"
        
        # 优先尝试 akshare 的 stock_zh_index_daily (通常最稳定)
        df = None
        try:
            df = log_external_api(f"ak.stock_zh_index_daily({index_code})", 
                                ak.stock_zh_index_daily, symbol=index_code)
        except Exception as e:
            print(f"ak.stock_zh_index_daily error: {e}")
            # 兜底尝试 akshare 的 em 接口
            try:
                df = log_external_api(f"ak.stock_zh_index_daily_em({index_code})", 
                                    ak.stock_zh_index_daily_em, symbol=index_code)
            except:
                pass
        
        if df is not None and not df.empty:
            # 统一列名以匹配 chip 计算函数
            # ak.stock_zh_index_daily 返回: date, open, high, low, close, volume
            # ak.stock_zh_index_daily_em 可能返回不同列名，统一处理
            column_map = {
                '日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume',
                'date': 'date', 'open': 'open', 'close': 'close', 'high': 'high', 'low': 'low', 'volume': 'volume'
            }
            df = df.rename(columns=column_map)
            
            # 指数通常没有换手率，或者换手率不适用于筹码计算
            # 我们使用固定衰减系数来模拟筹码流动
            settings = ChipSettings(use_turnover=False, decay=0.96) # 96% 的筹码留存率
            
            # 确保数据按日期升序
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 只取最近一年的数据进行计算，提高速度
            if len(df) > 300:
                df = df.tail(300)
            
            _, _, _, _, summary_stats, _ = calculate_advanced_chips(df, settings)
            
            if summary_stats:
                profit_ratio = summary_stats["profit_ratio"]
                
                # 情绪评估逻辑
                sentiment = "中性"
                if profit_ratio >= 90:
                    sentiment = "极度贪婪"
                elif profit_ratio >= 75:
                    sentiment = "贪婪"
                elif profit_ratio <= 10:
                    sentiment = "极度恐惧"
                elif profit_ratio <= 25:
                    sentiment = "恐惧"
                
                return {
                    "index": profit_ratio,
                    "sentiment": sentiment,
                    "description": f"上证指数当前获利比例约 {profit_ratio}%，市场情绪处于{sentiment}状态。",
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
    except Exception as e:
        print(f"Error calculating market sentiment: {e}")
    
    # 返回默认值以防崩溃
    return {
        "index": 0,
        "sentiment": "未知",
        "description": "暂无情绪数据，请稍后再试。",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/api/stock/{code}/financial-radar")
async def get_financial_radar(code: str):
    """
    获取个股财务雷达图数据
    包含：盈利能力、成长性、资产质量、现金流、估值水平
    """
    clean_code = code.split('.')[-1] if '.' in code else code
    
    # 尝试多种数据源
    df = None
    source_used = ""
    
    # 优先尝试同花顺摘要接口 (目前最稳定)
    try:
        df = log_external_api(f"ak.stock_financial_abstract_ths({clean_code})", 
                            ak.stock_financial_abstract_ths, symbol=clean_code, indicator="主要指标")
        if df is not None and not df.empty:
            source_used = "ths"
    except Exception as e:
        print(f"ths financial error: {e}")

    if df is None or df.empty:
        # 尝试多种股票代码格式
        code_formats = [clean_code, f"sh{clean_code}", f"sz{clean_code}"]
        for fmt in code_formats:
            try:
                df = log_external_api(f"ak.stock_financial_analysis_indicator({fmt})", 
                                    ak.stock_financial_analysis_indicator, symbol=fmt)
                if df is not None and not df.empty:
                    source_used = "ak"
                    break
            except:
                continue
    
    if (df is None or df.empty) and source_used == "":
        try:
            df = log_external_api(f"ak.stock_financial_analysis_indicator_em({clean_code})", 
                                ak.stock_financial_analysis_indicator_em, symbol=clean_code)
            if df is not None and not df.empty:
                source_used = "em"
        except:
            pass
    
    if df is None or df.empty:
        # 返回默认值
        return {
            "score": 0,
            "data": [
                {"subject": "盈利能力", "value": 0, "fullMark": 100, "original": "暂无数据"},
                {"subject": "成长性", "value": 0, "fullMark": 100, "original": "暂无数据"},
                {"subject": "资产质量", "value": 0, "fullMark": 100, "original": "暂无数据"},
                {"subject": "现金流", "value": 0, "fullMark": 100, "original": "暂无数据"},
                {"subject": "估值水平", "value": 0, "fullMark": 100, "original": "暂无数据"}
            ],
            "period": "暂无数据"
        }
    
    # 解析函数
    def parse_val(val):
        if val is None or val == "False" or val is False: return 0
        if isinstance(val, (int, float)): return val
        s = str(val).strip()
        if not s: return 0
        if s.endswith('%'):
            try: return float(s[:-1])
            except: return 0
        if s.endswith('亿'):
            try: return float(s[:-1])
            except: return 0 # 这里保持单位亿，因为归一化函数会处理
        if s.endswith('万'):
            try: return float(s[:-1]) / 10000
            except: return 0
        try: return float(s)
        except: return 0

    # 获取最新一期的财报数据 (ths 接口通常最后一行是最新，ak 接口通常第一行是最新)
    if source_used == "ths":
        latest = df.iloc[-1]
        # 清理列名中的空格
        latest.index = [c.strip() for c in latest.index]
        roe = parse_val(latest.get('净资产收益率', latest.get('净资产收益率-摊薄', 0)))
        growth_val = parse_val(latest.get('净利润同比增长率', 0))
        debt_ratio = parse_val(latest.get('资产负债率', 50))
        cash_val = parse_val(latest.get('每股经营现金流', 0))
        period = str(latest.get('报告期', '未知'))
    else:
        latest = df.iloc[0]
        # 清理列名中的空格
        latest.index = [c.strip() for c in latest.index]
        roe = parse_val(latest.get('净资产收益率(%)', 0))
        growth_val = parse_val(latest.get('净利润比上年同期增长(%)', 0))
        debt_ratio = parse_val(latest.get('资产负债率(%)', 50))
        cash_val = parse_val(latest.get('每股经营性现金流(元)', 0))
        period = str(latest.get('日期', '未知'))
    
    # 2. 获取估值指标 (PE/PB)
    # 使用东方财富的实时行情接口
    quote_df = None
    pe = 0
    pb = 0
    try:
        quote_df = log_external_api(f"ef.stock.get_latest_quote({clean_code})", ef.stock.get_latest_quote, clean_code)
        if quote_df is not None and not quote_df.empty:
            pe = quote_df.iloc[0].get('动态市盈率', 0)
            pb = quote_df.iloc[0].get('市净率', 0)
    except:
        pass

    # 3. 归一化计算函数 (0-100)
    def normalize(val, min_val, max_val, reverse=False):
        try:
            v = float(val)
            if np.isnan(v): return 50
            if reverse:
                score = 100 - ((v - min_val) / (max_val - min_val) * 100)
            else:
                score = (v - min_val) / (max_val - min_val) * 100
            return max(0, min(100, score))
        except:
            return 50

    # 提取关键指标并评分
    # 盈利能力: ROE (净资产收益率)
    profit_score = normalize(roe, 0, 20) # 0-20% 对应 0-100分
    
    # 成长性: 净利润增长率
    growth_score = normalize(growth_val, -10, 40) # -10% 到 40% 对应 0-100分
    
    # 资产质量: 资产负债率 (越低分越高)
    asset_score = normalize(debt_ratio, 20, 80, reverse=True) # 20%-80% 对应 100-0分
    
    # 现金流: 每股经营性现金流
    cash_score = normalize(cash_val, -0.5, 2.0) # -0.5 到 2.0 对应 0-100分
    
    # 估值水平: PE (越低分越高)
    valuation_score = normalize(pe, 10, 60, reverse=True) # 10-60 对应 100-0分

    radar_data = [
        {"subject": "盈利能力", "value": round(profit_score, 1), "fullMark": 100, "original": f"ROE: {roe}%"},
        {"subject": "成长性", "value": round(growth_score, 1), "fullMark": 100, "original": f"净利增长: {growth_val}%"},
        {"subject": "资产质量", "value": round(asset_score, 1), "fullMark": 100, "original": f"负债率: {debt_ratio}%"},
        {"subject": "现金流", "value": round(cash_score, 1), "fullMark": 100, "original": f"每股现金流: {cash_val}元"},
        {"subject": "估值水平", "value": round(valuation_score, 1), "fullMark": 100, "original": f"PE: {pe}"}
    ]

    # 计算综合评分
    total_score = sum(d['value'] for d in radar_data) / len(radar_data)
    
    return {
        "score": round(total_score, 1),
        "data": radar_data,
        "period": latest.get('日期', '未知')
    }

@app.get("/api/stock/{code}/diagnosis")
async def get_stock_diagnosis(code: str):
    clean_code = code.split('.')[-1] if '.' in code else code
    try:
        # Get full data for diagnosis
        data = get_stock_data_internal(
            clean_code, "efinance", "triangular", 1.0, 250, True, 1.0, 
            10.0, 10.0, True, 100, 10, 2, None, None, False
        )
        if not data:
            return {"error": "Could not fetch stock data"}
            
        stats = data["summary_stats"]
        history = data["history"]
        latest = history[-1]
        
        # Rule-based diagnosis logic
        diagnosis = []
        suggestions = ""
        risk_level = "中"
        
        # 1. Profit Ratio Analysis
        p_ratio = stats["profit_ratio"]
        if p_ratio > 90:
            diagnosis.append(f"当前获利比例高达{p_ratio}%，筹码高度获利，谨防高位回调。")
            risk_level = "高"
            suggestions = "建议逢高减仓，锁定利润。"
        elif p_ratio < 10:
            diagnosis.append(f"当前获利比例仅为{p_ratio}%，处于超跌区域，大部分筹码被套。")
            risk_level = "中"
            suggestions = "分批布局，等待反弹。"
        else:
            diagnosis.append(f"获利比例为{p_ratio}%，筹码分布相对均衡。")
            suggestions = "持有观察，关注成交量变化。"

        # 2. Concentration Analysis
        conc = stats["conc_90"]["concentration"]
        if conc < 10:
            diagnosis.append(f"筹码集中度极高({conc}%)，属于单峰密集状态，主力控盘迹象明显。")
        elif conc > 20:
            diagnosis.append(f"筹码集中度较低({conc}%)，筹码较为分散。")

        # 3. Cost Analysis
        avg_cost = stats["avg_cost"]
        curr_price = float(latest["close"])
        if curr_price > avg_cost * 1.15:
            diagnosis.append("当前价格远高于平均成本，存在短线超买风险。")
        elif curr_price < avg_cost * 0.85:
            diagnosis.append("当前价格远低于平均成本，属于空头陷阱或极度超跌。")

        # 4. Technical Indicators
        if "rsi" in latest and latest["rsi"] > 70:
            diagnosis.append("RSI指标进入超买区。")
        elif "rsi" in latest and latest["rsi"] < 30:
            diagnosis.append("RSI指标进入超卖区。")

        return {
            "diagnosis": diagnosis,
            "suggestions": suggestions,
            "risk_level": risk_level,
            "stats": stats
        }
    except Exception as e:
        return {"error": str(e)}

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

@cache.memoize(expire=600)
def get_money_flow_cached(clean_code: str):
    """获取资金流向数据"""
    try:
        # 1. 优先使用 efinance 获取详细的主力资金流向数据
        try:
            # 获取历史单子流入数据
            flow_df = log_external_api(f"ef.stock.get_history_bill({clean_code})", 
                                      ef.stock.get_history_bill, 
                                      clean_code)
            
            if flow_df is not None and not flow_df.empty:
                # 取最近30天数据
                recent_data = flow_df.tail(30)
                
                money_flow_data = []
                for _, row in recent_data.iterrows():
                    date = row.get('日期', '')
                    if isinstance(date, pd.Timestamp):
                        date = date.strftime('%Y-%m-%d')
                    
                    # 映射 efinance 的列名到前端需要的格式
                    # 注意：efinance 的主力净流入单位通常是元
                    money_flow_data.append({
                        "date": str(date),
                        "net_inflow": float(row.get('主力净流入', 0)),
                        "super_large_inflow": float(row.get('超大单净流入', 0)),
                        "large_inflow": float(row.get('大单净流入', 0)),
                        "medium_inflow": float(row.get('中单净流入', 0)),
                        "small_inflow": float(row.get('小单净流入', 0)),
                        "change_pct": float(row.get('涨跌幅', 0)),
                        "close": float(row.get('收盘价', 0)),
                    })
                
                # 计算汇总统计
                total_inflow = sum(d['net_inflow'] for d in money_flow_data if d['net_inflow'] > 0)
                total_outflow = sum(abs(d['net_inflow']) for d in money_flow_data if d['net_inflow'] < 0)
                net_flow = sum(d['net_inflow'] for d in money_flow_data)
                
                # 汇总大单数据
                summary_large = {
                    "super_large": sum(d['super_large_inflow'] for d in money_flow_data),
                    "large": sum(d['large_inflow'] for d in money_flow_data),
                    "medium": sum(d['medium_inflow'] for d in money_flow_data),
                    "small": sum(d['small_inflow'] for d in money_flow_data)
                }
                
                return {
                    "daily_flow": money_flow_data,
                    "summary": {
                        "total_inflow": round(total_inflow, 2),
                        "total_outflow": round(total_outflow, 2),
                        "net_flow": round(net_flow, 2),
                        "inflow_days": len([d for d in money_flow_data if d['net_inflow'] > 0]),
                        "outflow_days": len([d for d in money_flow_data if d['net_inflow'] < 0]),
                    },
                    "large_orders": {k: round(v, 2) for k, v in summary_large.items()}
                }
        except Exception as e:
            print(f"Efinance history bill error for {clean_code}: {e}")

        # 2. 备选方案：使用原有的简化计算逻辑
        flow_df = log_external_api(f"efinance.stock.get_quote_history({clean_code})", 
                                   ef.stock.get_quote_history, 
                                   clean_code, 
                                   klt=101)
        
        if flow_df is None or flow_df.empty:
            return {"error": "无法获取资金流向数据"}
        
        recent_data = flow_df.tail(30)
        money_flow_data = []
        for _, row in recent_data.iterrows():
            date = row.get('日期', '')
            if isinstance(date, pd.Timestamp):
                date = date.strftime('%Y-%m-%d')
            
            amount = float(row.get('成交额', 0))
            change_pct = float(row.get('涨跌幅', 0))
            
            # 简化计算：上涨视为流入，下跌视为流出
            net_inflow = amount * (change_pct / 100) if change_pct != 0 else 0
            
            money_flow_data.append({
                "date": str(date),
                "net_inflow": round(net_inflow, 2),
                "change_pct": change_pct,
                "amount": amount
            })
        
        total_inflow = sum(d['net_inflow'] for d in money_flow_data if d['net_inflow'] > 0)
        total_outflow = sum(abs(d['net_inflow']) for d in money_flow_data if d['net_inflow'] < 0)
        
        return {
            "daily_flow": money_flow_data,
            "summary": {
                "total_inflow": round(total_inflow, 2),
                "total_outflow": round(total_outflow, 2),
                "net_flow": round(total_inflow - total_outflow, 2),
                "inflow_days": len([d for d in money_flow_data if d['net_inflow'] > 0]),
                "outflow_days": len([d for d in money_flow_data if d['net_inflow'] < 0]),
            },
            "large_orders": {"super_large": 0, "large": 0, "medium": 0, "small": 0}
        }
    except Exception as e:
        print(f"Error fetching money flow for {clean_code}: {e}")
        return {"error": str(e)}

@app.get("/api/stock/{code}/money-flow")
async def get_money_flow(code: str):
    """获取股票资金流向数据"""
    clean_code = code.split('.')[-1] if '.' in code else code
    try:
        result = get_money_flow_cached(clean_code)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
