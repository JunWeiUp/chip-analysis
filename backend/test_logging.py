import requests
import time

def test_api_logging():
    base_url = "http://localhost:8001/api"
    
    print("\n--- Testing Stock Fundamentals (EF/AK) ---")
    start = time.time()
    resp = requests.get(f"{base_url}/stock/600519/fundamentals")
    print(f"Status: {resp.status_code}, Time: {time.time() - start:.2f}s")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Name: {data.get('股票简称')}")
    
    print("\n--- Testing Stock Data (Baostock) ---")
    start = time.time()
    # 强制不使用缓存 (通过修改参数或重启后端)
    resp = requests.get(f"{base_url}/stock/sh.600519?source=baostock")
    print(f"Status: {resp.status_code}, Time: {time.time() - start:.2f}s")
    
    print("\n--- Testing Stock Data (THS/AK) ---")
    start = time.time()
    resp = requests.get(f"{base_url}/stock/600519?source=ths")
    print(f"Status: {resp.status_code}, Time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    test_api_logging()
