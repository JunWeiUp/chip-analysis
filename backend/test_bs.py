import baostock as bs
import pandas as pd

lg = bs.login()
print(f"Login error_code: {lg.error_code}")

rs = bs.query_history_k_data_plus("sh.600000",
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
    start_date='2025-01-01', end_date='2026-01-31',
    frequency="d", adjustflag="3")
print(f"Query error_code: {rs.error_code}")

data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())

print(f"Result length: {len(data_list)}")
if data_list:
    print(data_list[0])

bs.logout()
