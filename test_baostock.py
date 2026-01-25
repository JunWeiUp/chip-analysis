import baostock as bs
import pandas as pd
import datetime

def test_stock_data(code):
    lg = bs.login()
    print(f"Login result: {lg.error_code}, {lg.error_msg}")
    
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    # Try to get 3 years of data to see how far back it goes
    start_date = (datetime.date.today() - datetime.timedelta(days=365*3)).strftime('%Y-%m-%d')
    
    print(f"Querying {code} from {start_date} to {end_date}")
    
    rs = bs.query_history_k_data_plus(
        code,
        "date,code,open,high,low,close,volume,turn,tradestatus",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"
    )
    
    print(f"Query result: {rs.error_code}, {rs.error_msg}")
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
        
    if data_list:
        df = pd.DataFrame(data_list, columns=rs.fields)
        print(f"Found {len(df)} records")
        print("First 5 records:")
        print(df.head())
        print("Last 5 records:")
        print(df.tail())
    else:
        print("No data found.")
        
    bs.logout()

if __name__ == "__main__":
    print("Testing sh.512890 (ETF):")
    test_stock_data("sh.512890")
    print("\nTesting sh.600000 (Stock):")
    test_stock_data("sh.600000")
