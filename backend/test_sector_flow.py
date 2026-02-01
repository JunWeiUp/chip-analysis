import akshare as ak
import pandas as pd

def test_sector_flow():
    print("Testing akshare sector fund flow rank...")
    try:
        df = ak.stock_sector_fund_flow_rank()
        if df is not None and not df.empty:
            print("Columns:")
            print(df.columns.tolist())
            print("\nSample data (top 5):")
            print(df.head(5))
        else:
            print("No data returned.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_sector_flow()
