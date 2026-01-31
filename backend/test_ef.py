import efinance as ef
import datetime

code = '600000'
end = datetime.date.today().strftime('%Y%m%d')
start = '20220131'
print(f"Fetching history for {code} from {start} to {end}")
df = ef.stock.get_quote_history(code, beg=start, end=end)
print(f"Result type: {type(df)}")
if df is not None:
    print(f"Result columns: {df.columns}")
    print(f"Result length: {len(df)}")
    print(df.head())
else:
    print("Result is None")
