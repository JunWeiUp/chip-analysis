
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import get_money_flow_cached

def test_optimized_money_flow():
    code = "002468"
    print(f"Testing optimized get_money_flow_cached for {code}...")
    try:
        result = get_money_flow_cached(code)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("Summary:")
            print(result.get("summary"))
            print("\nLarge Orders:")
            print(result.get("large_orders"))
            print("\nDaily Flow (last 3 days):")
            daily = result.get("daily_flow", [])
            for d in daily[-3:]:
                print(d)
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_optimized_money_flow()
