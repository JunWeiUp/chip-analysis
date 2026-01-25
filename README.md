# BaoStock 股票筹码分析工具

这是一个基于 Vite + React 和 FastAPI 的股票分析软件，主要功能是分析指定股票每日的筹码收盘获利比例。

## 核心功能

1.  **股票代码输入**：支持输入上海（如 `sh.600000`）和深圳（如 `sz.000001`）股票代码。
2.  **数据获取**：通过 Python 后端调用 `BaoStock` API 获取历史日线行情数据。
3.  **筹码分析**：后端根据历史成交量和收盘价，计算每日的筹码收盘获利比例（简化模型）。
4.  **数据可视化**：使用 Ant Design Charts 展示获利比例的时间序列折线图。
5.  **详细数据展示**：使用 Ant Design Table 列出每日的开盘、收盘、成交量及获利比例。

## 技术栈

-   **前端**：Vite, React, TypeScript, Ant Design, Ant Design Charts (@ant-design/plots), Axios.
-   **后端**：Python, FastAPI, BaoStock, Pandas, Uvicorn.

## 项目启动

### 1. 后端启动

确保已安装 Python 3.5+。

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows 使用 venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

后端将运行在 `http://localhost:8000`。

### 2. 前端启动

```bash
cd frontend
npm install
npm run dev
```

前端将运行在 `http://localhost:5173`。

## 计算逻辑说明

**筹码收盘获利比例** 的计算采用了简化模型：
1.  选取过去 250 个交易日（约一年）作为筹码分布窗口。
2.  统计该窗口内，所有收盘价低于当前交易日收盘价的成交量总和。
3.  获利比例 = (低价成交量之和 / 窗口内总成交量) * 100%。

该指标反映了当前价格水平下，持仓筹码中盈利部分的占比。

## 使用方法

1.  在输入框中输入股票代码（格式：`sh.xxxxxx` 或 `sz.xxxxxx`）。
2.  点击“分析数据”按钮。
3.  查看上方的获利比例走势图和下方的详细数据表格。
