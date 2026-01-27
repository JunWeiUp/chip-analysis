
import pandas as pd

class MockReq:
    def __init__(self, strategy_type, buy_threshold, sell_threshold):
        self.strategy_type = strategy_type
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

def run_sim(strategy_type, buy_threshold, sell_threshold):
    test_df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'profit_ratio': [10, 4, 3, 6, 2],
        'close': [100, 95, 90, 92, 88]
    })
    
    req = MockReq(strategy_type, buy_threshold, sell_threshold)
    
    initial_cash = 2000.0  # 设小一点，方便测试“无限”买入
    cash = initial_cash
    shares = 0.0
    trades = []
    
    # 只买不卖：单笔 1000 元
    buy_unit_cash = 1000.0
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        ratio = row['profit_ratio']
        price = float(row['close'])
        date = str(row['date'])
        
        should_buy = False
        if req.strategy_type == "buy_and_hold":
            if ratio <= req.buy_threshold:
                should_buy = True
        
        # 只买不卖不限制现金
        can_buy = (req.strategy_type == "buy_and_hold") or (cash > 0)
        
        if should_buy and can_buy:
            amount_to_spend = buy_unit_cash
            
            shares_to_buy = amount_to_spend / price
            cash -= amount_to_spend
            shares += shares_to_buy
            trades.append({"type": "buy", "date": date, "price": price, "amount": amount_to_spend})
    
    print(f"Strategy: {strategy_type}, Initial Cash: {initial_cash}, Final Cash: {cash}, Trades: {len(trades)}")
    # 计算累计收益率
    # 按照（走势图最后一天收盘持仓价*持仓份数 - 投入）/ 投入
    current_cost_basis = sum([t['amount'] for t in trades])
    final_price = float(test_df.iloc[-1]['close'])
    if current_cost_basis > 0:
        final_market_value = shares * final_price
        final_yield = (final_market_value - current_cost_basis) / current_cost_basis
        print(f"Final Yield (based on final price {final_price}): {round(final_yield * 100, 2)}%")
    
    # 模拟每日收益率更新
    print("\nDaily Yield Curve (using Final Price):")
    temp_shares = 0
    temp_cost = 0
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        p = float(row['close'])
        # 简化模拟买入
        if i > 0: # 假设从第二天开始买
            temp_shares += 1000 / p
            temp_cost += 1000
        
        if temp_cost > 0:
            y = (temp_shares * final_price - temp_cost) / temp_cost
            print(f"Date: {row['date']}, Yield: {round(y*100, 2)}%")
    
    print("\nTrades:")
    for t in trades:
        print(t)

if __name__ == "__main__":
    run_sim("buy_and_hold", 5, 50)
