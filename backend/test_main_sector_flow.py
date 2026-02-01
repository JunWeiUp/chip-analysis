
import sys
import os
import pandas as pd
import numpy as np
import asyncio

# Add parent directory to path to import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import get_sector_money_flow

async def test_sector_money_flow():
    code = "002468"  # 神州泰岳
    print(f"Testing get_sector_money_flow for {code}...")
    try:
        result = await get_sector_money_flow(code)
        print(f"Industry identified: {result.get('industry')}")
        
        stock_flow = result.get('stock_flow', [])
        if stock_flow:
            print(f"Found {len(stock_flow)} days of stock flow data.")
            latest = stock_flow[-1]
            print(f"Latest Stock Flow: {latest.get('日期')} | 主力净流入: {latest.get('主力净流入')}")
            
        sector_flow = result.get('sector_flow')
        if sector_flow:
            print(f"Sector Flow Info: {sector_flow.get('名称')} | 主力净额: {sector_flow.get('主力净额')} | 涨跌幅: {sector_flow.get('涨跌幅')}%")
            if sector_flow.get('message'):
                print(f"Note: {sector_flow.get('message')}")
        else:
            print("No sector flow data found.")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_sector_money_flow())
