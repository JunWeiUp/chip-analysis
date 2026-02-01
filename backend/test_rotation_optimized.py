
import sys
import os
import asyncio
import time

# Add parent directory to path to import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import get_sector_rotation

async def test_optimized_rotation():
    print("Testing optimized get_sector_rotation with parallel efinance fetching...")
    start_time = time.time()
    try:
        result = await get_sector_rotation()
        duration = time.time() - start_time
        
        if isinstance(result, dict) and "error" in result:
            print(f"Error: {result['error']}")
            return

        data = result if isinstance(result, list) else []
        print(f"Fetched {len(data)} sectors in {duration:.2f}s")
        
        if len(data) > 0:
            print("\nTop 15 Sectors (Sorted by Gain):")
            # Display sectors and check if 主力净额 is non-zero
            for i, sector in enumerate(data):
                net_inflow = sector.get('主力净额', 0)
                # Convert to 100M if it's large
                if abs(net_inflow) > 100000000:
                    net_inflow_str = f"{net_inflow/100000000:.2f}亿"
                elif abs(net_inflow) > 10000:
                    net_inflow_str = f"{net_inflow/10000:.2f}万"
                else:
                    net_inflow_str = f"{net_inflow:.2f}元"
                    
                print(f"{i+1:2}. {sector.get('板块名称'):<10}: 涨跌幅 {sector.get('涨跌幅'):>5}% | 主力净额: {net_inflow_str}")
            
            # Count how many have non-zero fund flow
            non_zero = sum(1 for s in data if s.get('主力净额', 0) != 0)
            print(f"\nSectors with non-zero fund flow: {non_zero}/{len(data)}")
            
            if non_zero == 0:
                print("Warning: All sectors have 0 fund flow. Optimization might have failed or data is missing.")
            else:
                print("Success: Real fund flow data captured.")

    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_optimized_rotation())
