# 修改记录 (AGENTS.md)

## 2026-02-01 12:30: 添加项目根目录 .gitignore

### 修改背景
为了保持代码库整洁，防止缓存文件、日志文件及本地配置被提交到版本控制系统中。

### 修改内容
1.  **新增根目录 .gitignore**:
    *   明确忽略了 `cache/` 和 `backend/cache/` 目录（diskcache 产生的缓存）。
    *   忽略了 Python 编译文件 `__pycache__` 和 `*.pyc`。
    *   忽略了后端生成的日志文件 `*.log` 和进程文件 `backend.pid`。
    *   增加了对 `node_modules`、`dist` 以及操作系统/编辑器特定文件的常规忽略规则。

### 涉及文件
- [.gitignore](file:///Users/mac/Documents/code1/stock/.gitignore) (新建)

---

## 2026-02-01 12:15: 修复大盘情绪指数数据缺失问题

### 修改背景
用户反馈大盘“恐惧与贪婪”指数没有数据。经排查，原因是后端接口使用了不兼容的代码格式请求 `efinance` 接口，且未针对指数数据优化筹码计算逻辑。

### 修改内容
1.  **更换数据源 (main.py)**:
    *   将 `/api/market/sentiment` 的数据源从 `efinance` 切换为更稳定的 `akshare.stock_zh_index_daily`。
    *   增加了 `akshare.stock_zh_index_daily_em` 作为二级兜底，确保指数数据的获取成功率。
2.  **优化指数筹码计算逻辑**:
    *   针对指数缺乏换手率数据的特点，将计算模式调整为“固定衰减”模式（`use_turnover=False`, `decay=0.96`）。
    *   增加了数据清洗和排序逻辑，确保参与计算的数据格式符合算法要求。
    *   限制计算范围为最近 300 个交易日，在保证准确性的同时提升了响应速度。
3.  **完善错误处理**:
    *   统一了列名映射（处理中英文列名差异），确保算法能正确识别价格和成交量字段。

### 涉及文件
- [backend/main.py](file:///Users/mac/Documents/code1/stock/backend/main.py)

---

## 2026-02-01 11:58: 模块结构优化与自选股抽屉化集成

### 修改背景
优化系统布局，将自选股管理模块整合为侧边抽屉，移除冗余模块，并全面提升前端代码的类型安全性。

### 修改内容

#### 1. 自选股模块抽屉化 (Sheet)
- **新增组件**: [frontend/src/components/ui/sheet.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/ui/sheet.tsx)
  - 引入了 Radix UI 的 Dialog 原语，实现高性能、可访问的右侧滑出抽屉。
- **UI 集成 (App.tsx)**:
  - 在页面头部导航栏右侧新增“我的自选”按钮作为触发器。
  - 将 `FavoriteStocks` 组件封装在 `Sheet` 中，点击按钮时从右侧滑出。
  - 从主页面流中移除了原有的自选股展示区域，使界面更加简洁聚焦。
- **组件适配 (FavoriteStocks.tsx)**:
  - 移除了外部 `Card` 容器依赖，改为 flex 布局以适应抽屉的高度。
  - 优化了内部标题和刷新按钮的布局，提升了在窄屏（抽屉宽度）下的展示效果。

#### 2. 界面清理
- **移除历史交易模块**: 从 [App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx) 中彻底删除了“历史交易明细”卡片及其对应的 `CoolStockTable` 引用，并物理删除了 `frontend/src/components/CoolStockTable.tsx` 文件，保持代码库整洁。

#### 3. 类型安全深度优化 (App.tsx)
- **消除 any 隐患**: 定义了多个核心数据接口，替代了原有的 `Record<string, any>`：
  - `SectorComparison` & `SectorComparisonItem`: 行业对比数据。
  - `SectorMoneyFlow` & `StockFlowItem`: 资金流向数据。
  - `SectorRotationItem`: 板块轮动项。
  - `Diagnosis`: 智能诊断报告。
  - `FinancialRadarData`: 财务雷达图数据。
- **跨组件类型共享**: 将 `SentimentData` 接口从 [MarketSentiment.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/MarketSentiment.tsx) 导出，并在 `App.tsx` 中通过 `import type` 正确引用，修复了类型不匹配导致的 linter 错误。

### 涉及文件
- [frontend/src/App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx) (修改)
- [frontend/src/components/FavoriteStocks.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/FavoriteStocks.tsx) (修改)
- [frontend/src/components/ui/sheet.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/ui/sheet.tsx) (新建)
- [frontend/src/components/MarketSentiment.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/MarketSentiment.tsx) (修改)
- [AGENTS.md](file:///Users/mac/Documents/code1/stock/AGENTS.md) (修改)

---

## 2026-02-01 11:35: 全面支持自选股功能集成与状态同步

### 修改背景
支持用户在股票分析系统中方便地添加和移除自选股，并确保分析界面与自选股管理列表之间的状态实时同步。

### 修改内容

#### 1. 前端状态管理升级 (App.tsx)
- **自选状态全局化**: 将 `favorites` 状态从 `FavoriteStocks.tsx` 提升至 `App.tsx`，作为全局状态管理。
- **类型安全增强**: 引入了 `FavoriteStock` 接口定义，替代原有的 `any[]` 类型，并修复了相关的 linter 错误。
- **快捷操作逻辑**: 实现了 `toggleFavorite` 函数，支持根据当前分析的股票代码自动判断并切换自选状态。

#### 2. 交互界面优化 (App.tsx)
- **新增“加入/移除自选”按钮**: 在主分析操作区（“开始分析”旁）集成了星标按钮。
- **实时视觉反馈**: 按钮图标颜色根据自选状态动态切换（已加入显示琥珀色实心星标，未加入显示灰色空心星标）。
- **智能名称获取**: 在添加自选时，自动从基本面数据中提取“股票简称”，确保列表展示更友好。

#### 3. 组件通信重构 (FavoriteStocks.tsx)
- **Props 驱动同步**: 将 `FavoriteStocks` 组件重构为受控组件，通过 `props` 接收 `favorites` 和 `setFavorites`。
- **接口导出**: 导出了 `FavoriteStock` 接口，供其他组件（如 `App.tsx`）进行类型引用。
- **清理冗余代码**: 修复了 `lucide-react` 中 `X` 图标定义但未使用的 linter 错误。

#### 4. 数据持久化
- **双重存储保障**: 保留并优化了 `localStorage` 的读写逻辑，确保用户刷新页面后自选股列表依然存在。

### 涉及文件
- [frontend/src/App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx) (修改)
- [frontend/src/components/FavoriteStocks.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/FavoriteStocks.tsx) (修改)

---

## 2026-02-01 10:55: 主力资金流向数据源深度优化与并行加速

### 修改背景
用户需要增强股票分析系统的功能，添加自选股管理、筹码峰值追踪和资金流向分析三大核心功能模块。

### 修改内容

#### 1. 自选股管理功能
**新增文件**: [frontend/src/components/FavoriteStocks.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/FavoriteStocks.tsx)

**功能特性**:
- 添加/删除自选股
- 自选股列表展示（股票代码、名称）
- 一键刷新所有自选股数据
- 显示实时价格、涨跌幅、获利比例、筹码集中度
- 点击自选股快速切换分析
- 本地存储（localStorage）持久化

**组件接口**:
```typescript
interface FavoriteStocksProps {
  onSelectStock: (code: string) => void;
}
```

**主要功能**:
- `addFavorite()`: 添加股票到自选列表
- `removeFavorite()`: 从自选列表移除股票
- `refreshData()`: 批量刷新所有自选股的实时数据

#### 2. 筹码峰值追踪功能
**新增文件**: [frontend/src/components/ChipPeakAnalysis.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/ChipPeakAnalysis.tsx)

**功能特性**:
- 显示当前筹码峰值价格
- 峰值与收盘价的距离分析
- 近30日峰值迁移趋势计算
- 筹码集中度变化监测
- 峰值/收盘价/平均成本三线对比图表
- 智能分析建议

**核心算法**:
- 峰值迁移趋势 = (最新峰值 - 30日前峰值) / 30日前峰值 × 100%
- 峰值距离 = (峰值价格 - 收盘价) / 收盘价 × 100%
- 集中度变化 = 最新集中度 - 30日前集中度

**分析提示**:
- 价格低于峰值 > 5%: 可能存在上行空间
- 价格高于峰值 > 5%: 注意获利回吐风险
- 峰值持续上移 > 3%: 处于上升趋势
- 峰值持续下移 > 3%: 处于下跌趋势
- 筹码集中度 < -5%: 筹码高度集中

#### 3. 资金流向分析功能
**新增文件**: 
- 前端: [frontend/src/components/MoneyFlowAnalysis.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/MoneyFlowAnalysis.tsx)
- 后端接口: [backend/main.py](file:///Users/mac/Documents/code1/stock/backend/main.py) 中的 `/api/stock/{code}/money-flow`

**功能特性**:
- 近30日资金净流向统计
- 流入/流出天数统计
- 大单、中单、小单资金分析
- 每日资金流向柱状图
- 资金活跃度指标（流入占比）
- 智能资金流向分析建议

**后端接口**:
```python
@app.get("/api/stock/{code}/money-flow")
async def get_money_flow(code: str):
    """获取股票资金流向数据"""
```

**数据结构**:
```python
{
  "daily_flow": [
    {
      "date": "2026-01-31",
      "volume": 1000000,
      "amount": 50000000,
      "change_pct": 2.5,
      "turnover": 3.2,
      "net_inflow": 1250000
    }
  ],
  "summary": {
    "total_inflow": 5000000,
    "total_outflow": 3000000,
    "net_flow": 2000000,
    "inflow_days": 18,
    "outflow_days": 12
  },
  "large_orders": {
    "super_large": 2000000,
    "large": 1500000,
    "medium": 1000000,
    "small": 500000
  }
}
```

**资金流向计算逻辑**:
- 净流入 = 成交额 × (涨跌幅 / 100)
- 上涨时视为资金流入，下跌时视为资金流出
- 大单分类基于成交额相对于平均成交额的倍数

#### 4. App.tsx 集成
**修改文件**: [frontend/src/App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx)

**集成位置**:
1. 自选股组件：放置在基本面信息卡片之后
2. 筹码峰值分析：放置在右侧栏筹码统计概览之后
3. 资金流向分析：放置在右侧栏最底部

**新增依赖**:
```bash
npm install chart.js react-chartjs-2
```

### 涉及文件
* [frontend/src/components/FavoriteStocks.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/FavoriteStocks.tsx) (新建)
* [frontend/src/components/ChipPeakAnalysis.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/ChipPeakAnalysis.tsx) (新建)
* [frontend/src/components/MoneyFlowAnalysis.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/MoneyFlowAnalysis.tsx) (新建)
* [frontend/src/App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx) (修改)
* [backend/main.py](file:///Users/mac/Documents/code1/stock/backend/main.py) (修改：新增资金流向接口)
* [frontend/package.json](file:///Users/mac/Documents/code1/stock/frontend/package.json) (修改：新增依赖)

### 功能截图说明

1. **自选股管理**
   - 位置：基本面信息下方
   - 特点：卡片式布局，支持添加/删除/刷新操作
   - 数据：显示价格、涨跌幅、获利比例、筹码集中度

2. **筹码峰值追踪**
   - 位置：右侧栏中部
   - 特点：双指标卡片 + 趋势图表 + 智能分析
   - 数据：峰值价格、迁移趋势、集中度变化

3. **资金流向分析**
   - 位置：右侧栏底部
   - 特点：净流向统计 + 大单分析 + 每日柱状图
   - 数据：流入/流出金额、天数、大单占比

## 2026-01-31: 新增板块分析与智能诊断报告

### 修改背景
为了提供多维度的分析视角和辅助决策建议，新增了板块分析模块和基于规则的智能诊断报告。

### 修改内容

#### 1. 板块分析功能
**后端接口**:
- `GET /api/stock/{code}/sector/comparison`: 获取同行业股票筹码对比数据。
- `GET /api/stock/{code}/sector/money-flow`: 获取行业板块及个股的资金流向数据。
- `GET /api/sector/rotation`: 获取全行业板块轮动（今日涨幅榜）数据。

**前端展示**:
- **同板块对比**: 展示同行业前10名股票的获利比例、平均成本、筹码集中度。
- **资金流向**: 展示板块主力净流入、涨跌幅及个股近5日主力资金趋势。
- **板块轮动**: 以网格形式展示今日表现最强劲的行业板块及其资金动向。

#### 2. 智能诊断报告
**后端接口**:
- `GET /api/stock/{code}/diagnosis`: 综合筹码、价格、指标进行多维度诊断。

**核心逻辑**:
- **筹码状态**: 分析获利比例（高位风险/超跌反弹）、集中度（主力控盘/筹码分散）。
- **成本位置**: 计算价格与平均成本的乖离率，识别超买/超卖。
- **技术指标**: 结合 RSI 等指标识别极端行情。
- **买卖建议**: 提供针对性的操作策略建议（如逢高减仓、分批布局等）。
- **风险评级**: 根据综合得分给出“高、中、低”风险评级。

**前端展示**:
- 采用 Tabs 切换，将“筹码分析”、“板块分析”、“智能诊断”解耦。
- 使用进度条和仪表盘可视化展示筹码健康度指标。

### 涉及文件
- [backend/main.py](file:///Users/mac/Documents/code1/stock/backend/main.py)

## 2026-02-01 00:30: 解决板块分析数据缺失问题（针对 EM 接口不稳定）

### 修改背景
用户反馈 `002468` 等股票在板块分析中依然显示无数据。经排查，原因是 `akshare` 内部调用的东方财富 (EM) 接口在当前环境下频繁出现连接中断，导致行业成份股及板块行情获取失败。

### 修改内容

#### 1. 板块成份股获取增加 Baostock 兜底
- **逻辑优化**: 在 `get_industry_stocks` 函数中引入了多级回退机制。
- **Baostock 整合**: 当 EM 接口失效时，自动调用 `baostock` 的行业分类数据。通过当前股票代码反查所属行业（如 `002468` 对应“邮政业”），并提取同行业的所有成份股。
- **参数扩展**: 接口支持传入 `stock_code`，确保在行业名称匹配模糊时仍能精准定位所属板块。

#### 2. 板块轮动数据增加同花顺 (THS) 备选
- **多源行情**: 在 `get_sector_rotation` 中，如果 EM 的行业涨幅榜获取失败，系统将自动切换至同花顺的行业指数数据。
- **数据对齐**: 整合了同花顺的行业名称与简要行情，确保即使在极端网络环境下，用户仍能看到基本的板块热度排名。

## 2026-02-01 10:55: 主力资金流向数据源深度优化与并行加速

### 修改背景
解决主力净流入显示 `NaN` 或 `0` 的问题，通过接入多数据源 fallback 机制并引入并行请求，提升数据的准确性与接口稳定性。

### 修改内容

#### 1. 个股资金流向优化 (main.py)
- **接入 efinance 实时账单**: 在 `get_money_flow_cached` 和 `get_sector_money_flow` 中，优先使用 `efinance.get_history_bill` 获取个股主力资金流向。相比 `akshare`，该接口返回的数据包含更精确的超大单/大单分类，且稳定性更高。
- **历史数据扩充**: 将个股资金流向的返回天数从 5 天增加至 30 天，为前端提供更丰富的趋势分析数据。

#### 2. 板块轮动并行加速与兜底 (main.py)
- **并行数据抓取**: 在 `get_sector_rotation` 中引入 `ThreadPoolExecutor`。当 `akshare` 行业排名接口连接断开时，自动并发请求 `efinance` 获取涨幅前 15 个板块的实时主力净流入数据。
- **智能名称匹配**: 实现了行业名称的自动清洗逻辑，在请求 `efinance` 时自动处理行业名称后缀，提高数据匹配率。
- **数据一致性保证**: 统一了 `akshare` 和 `efinance` 的字段映射（如 `主力净流入`、`主力净额`），确保前端展示逻辑无缝衔接。

#### 3. 健壮性与性能提升
- **连接错误自动容错**: 针对 `akshare` 频繁出现的 `RemoteDisconnected` 错误，建立了“AK排名 -> EF单点 -> THS行情”的三级兜底机制。
- **响应耗时优化**: 虽然引入了多数据源和并行请求，但通过限制并发数量（`max_workers=5`）和只针对 Top 板块补充数据，将整体接口响应时间控制在合理范围内（约 3-6s，含超时重试）。

### 验证结果
- **数据准确性**: 验证了个股（如 `002468`）和板块（如 `半导体`、`通信设备`）的主力净流入数据均能正常获取，不再显示 `NaN` 或全为 `0`。
- **并发性能**: 在 `akshare` 故障期间，系统能通过并发 EF 请求在 6 秒内补全核心板块的资金流数据。

### 涉及文件
- [backend/main.py](file:///Users/mac/Documents/code1/stock/backend/main.py)
- [backend/test_main_money_flow.py](file:///Users/mac/Documents/code1/stock/backend/test_main_money_flow.py)
- [backend/test_main_sector_flow.py](file:///Users/mac/Documents/code1/stock/backend/test_main_sector_flow.py)
- [backend/test_rotation_optimized.py](file:///Users/mac/Documents/code1/stock/backend/test_rotation_optimized.py)

## 2026-02-01 10:45: 深度优化板块分析与行业数据获取

### 修改背景
解决行业名称不匹配、API 冗余调用以及前端组件对数据类型处理不够健壮的问题。

### 修改内容

#### 1. 后端 API 逻辑重构 (main.py)
- **行业名称模糊匹配**: 在 `get_industry_stocks` 中实现了名称变体匹配（如“物流行业”自动尝试“物流”），显著提升了东方财富接口的匹配成功率。
- **数据结构扁平化**: 重构 `get_fundamentals_cached` 移除冗余的 `info` 层级，统一返回扁平化字典，降低内部调用复杂度。
- **板块分析接口加固**: 
  - `get_sector_comparison`: 移除了对已废弃 `info` 字典的引用，增加了从个股板块列表中提取行业的兜底逻辑。
  - `get_sector_money_flow`: 优化了 THS 兜底逻辑中的字段映射（`板块` -> `板块名称`），解决了合并导致的 `KeyError`。
- **Baostock 稳定性增强**: 优化了 Baostock 行业查询逻辑，通过全量数据过滤实现对 `002468` 等股票行业成份股的精准提取（成功识别“邮政业”板块）。

#### 2. 前端健壮性提升 (App.tsx & Components)
- **类型安全检查**: 在 `App.tsx` 中增加了对 `sectorRotation` 的 `Array.isArray` 检查，防止接口返回非数组导致页面崩溃。
- **数值处理防错**: 在 `MarketSentiment.tsx` 等组件中增加了对 `undefined` 的空值合并处理，防止 `toFixed` 报错。
- **板块轮动数值展示**: 修复了板块轮动监控中主力净流入显示 `NaN` 的问题，通过后端数据预处理（`fillna(0)`）和前端空值合并（`|| 0`）双重保障。

### 验证结果
- **板块轮动**: “今日主力净流入”不再显示 `NaN`，即使在数据源合并失败时也能显示为 `0.00亿`。
- **002468 (申通快递)**: 行业成份股已能通过 Baostock 正常获取，板块资金流向已能通过 THS 模糊匹配正确显示。
- **稳定性**: 系统在东方财富接口连接中断时，能平滑切换至 Baostock 或 THS，确保功能不中断。

#### 3. 系统鲁棒性提升
- **错误捕获**: 细化了外部 API 调用的异常捕获，确保单个接口超时不会导致整个分析流程中断。
- **性能兼顾**: 依然保留了 1 小时的分钟级缓存，减少对兜底接口（如 Baostock 登录/登出）的频繁调用。

### 涉及文件
- [backend/main.py](file:///Users/mac/Documents/code1/stock/backend/main.py)

## 2026-02-01 00:15: 板块分析功能全面补全与性能优化

### 修改背景
用户反馈板块分析数据不完整。经检查发现，板块对比计算缓慢、部分股票缺失行业归属、资金流向接口偶尔失效。

### 修改内容

#### 1. 板块对比 (Sector Comparison) 深度优化
- **计算加速**: 将对比股票的筹码计算追溯天数从 250 天缩减至 120 天，大幅提升接口响应速度。
- **数据增强**: 对比列表中始终包含当前分析的股票（标记为 `is_current`），方便用户直观对比。
- **容错处理**: 增加了行业缺失时的提示逻辑，并限制对比股票数量为前 8 名，兼顾性能与参考价值。

#### 2. 资金流向 (Money Flow) 鲁棒性增强
- **多源备份**: 针对个股资金流向，引入 `efinance` 作为 `akshare` 接口失效时的备份数据源。
- **智能降级**: 若无法匹配到个股的具体行业资金流向，自动展示今日领涨的前 3 个板块作为参考背景，避免页面出现空白。

#### 3. 板块轮动 (Sector Rotation) 维度扩充
- **多维合并**: 结合了行业板块的“涨跌幅”行情数据与“主力净流入”资金数据。
- **数据清洗**: 统一处理了合并过程中的 `NaN` 值，确保前端接收到的 JSON 数据格式规范，防止渲染崩溃。

### 涉及文件
- [backend/main.py](file:///Users/mac/Documents/code1/stock/backend/main.py)
- [frontend/src/App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx)

## 2026-01-31: 移除获利比例风险提示

### 修改背景
用户希望去掉页面上的风险提示信息（包括极高/极低获利比例及筹码高度集中的预警）。

### 修改内容
1.  **移除 UI 组件**:
    *   在 [App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx) 中移除了 `renderAlerts` 函数定义。
    *   在 [App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx) 的 JSX 渲染逻辑中移除了对 `{renderAlerts()}` 的调用。

### 涉及文件
*   [frontend/src/App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx)

## 2026-01-31: 股票搜索记录增加中文名称

### 修改背景
用户希望在前端的股票搜索历史记录中，除了显示股票代码外，还能显示对应的中文名称。

### 修改内容
1.  **数据模型定义**:
    *   在 [App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx) 中增加了 `SearchRecord` 接口，包含 `code` (股票代码) 和 `name` (股票名称)。
2.  **状态管理更新**:
    *   将 `searchHistory` 的状态类型从 `string[]` 更新为 `SearchRecord[]`。
    *   增加了对 `localStorage` 中旧版本字符串格式数据的兼容性转换逻辑。
3.  **数据抓取与存储逻辑优化**:
    *   修改了 `fetchData` 函数。现在不再在发起请求前立即保存搜索记录，而是在 `fundamentalsRes` (基本面数据) 成功返回后，从中提取“股票简称”字段。
    *   调用更新后的 `addToHistory(code, name)` 将代码和名称一同存入历史记录。
4.  **UI 组件渲染**:
    *   更新了搜索历史标签的渲染逻辑。如果记录中存在 `name`，则以 `代码 | 名称` 的格式展示，并优化了按钮的样式（使用 `flex` 布局和 `gap`）。

### 涉及文件
*   [frontend/src/App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx)

## 2026-01-31: 统一后端外部 API 调用日志

### 修改背景
为了方便监控后端对第三方接口（akshare, efinance, baostock）的调用情况，需要将所有外部请求记录到控制台。

### 修改内容
1.  **引入统一日志包装器**:
    *   在 [main.py](file:///Users/mac/Documents/code1/stock/backend/main.py) 中实现了 `log_external_api` 函数。
    *   该包装器会自动记录调用的开始时间、结束时间、总耗时以及执行状态（成功/失败）。
2.  **重构外部调用点**:
    *   移除了之前散落在代码各处的 `print` 语句。
    *   将所有对 `ak.xxx`, `ef.xxx`, `bs.xxx` 的直接调用替换为通过 `log_external_api` 进行包装调用。
3.  **标准化输出格式**:
    *   输出格式统一为：`[时间] >>> EXTERNAL CALL START: 接口名` 和 `[时间] <<< EXTERNAL CALL END: 接口名 | Duration: 耗时 | Status: 状态`。

### 涉及文件
*   [backend/main.py](file:///Users/mac/Documents/code1/stock/backend/main.py)

## 2026-01-31: 后端数据接口引入分钟级缓存

### 修改背景
股票历史数据及相关筹码计算频率不高，为了提升响应速度并减轻后端计算压力，对相同代码和时间段的请求引入缓存机制。

### 修改内容
1.  **引入端到端缓存**:
    *   在 `backend/main.py` 中利用 `diskcache` 对核心计算接口进行了重构。
    *   新增 `get_stock_data_internal` 函数，并使用 `@cache.memoize(expire=600)` 进行 10 分钟缓存，涵盖了筹码分布计算等耗时操作。
    *   新增 `run_backtest_internal` 函数，同样引入 10 分钟缓存，通过对请求参数进行哈希化处理实现相同回测配置的快速响应。
2.  **优化接口逻辑**:
    *   将 FastAPI 的路由函数重构为调用内部缓存函数的模式，确保了代码的整洁与缓存的一致性。

### 涉及文件
*   [backend/main.py](file:///Users/mac/Documents/code1/stock/backend/main.py)

## 2026-01-31: 优化股票代码搜索功能与 UI

### 修改背景
提升用户搜索股票时的交互体验，支持键盘导航，并优化搜索建议列表的展示效果。同时修复了基本面信息未展示的问题。

### 修改内容
1.  **搜索功能增强**:
    *   在 [App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx) 中优化了 `onKeyDown` 处理逻辑，支持 `ArrowUp` / `ArrowDown` 切换建议项，并实现了 `scrollIntoView` 确保选中项可见。
    *   支持 `Enter` 键直接选择当前高亮的建议项并触发数据加载。
    *   支持 `Escape` 键快速关闭建议列表。
2.  **UI 体验优化**:
    *   改进了搜索建议项的样式，使用等宽字体显示代码，并优化了名称的截断处理。
    *   增加了高亮项的背景色和字体加粗效果。
3.  **修复基本面展示**:
    *   将 `renderFundamentals` 函数重新挂载到页面渲染流程中，确保股票的基本面数据（如行业、概念等）能够正常显示。
4.  **增强基本面渲染逻辑**:
    *   在 [App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx) 中优化了 `renderFundamentals` 的错误处理和数据兼容逻辑，增加了对结构化数据和扁平数据的双重支持。
    *   将基本面请求与历史数据请求解耦，避免其中一个失败导致整体流程中断。
    *   添加了详细的调试日志以便追踪数据流。
5.  **清理后端缓存与优化**:
    *   由于后端修改了基本面数据的返回结构，手动清理了 `backend/cache` 下的旧缓存文件，确保新结构立即生效。
    *   修复了 `get_fundamentals_cached` 在某些情况下返回格式不一致的问题。

### 涉及文件
*   [frontend/src/App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx)
*   [backend/main.py](file:///Users/mac/Documents/code1/stock/backend/main.py)
*   [backend/cache/*](file:///Users/mac/Documents/code1/stock/backend/cache/) (清理操作)

## 2026-01-31: 修复代码报错与清理冗余代码

### 修改背景
清理前端和后端代码中的 linter 报错（如未使用的变量）以及冗余的导入项，提升代码质量。

### 修改内容
1.  **修复前端图表组件报错**:
    *   在 [ProfitRatioChart.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/ProfitRatioChart.tsx) 中，移除了 `macdHistData.map` 函数中未使用的索引变量 `i`。
2.  **后端代码清理与优化**:
    *   在 [main.py](file:///Users/mac/Documents/code1/stock/backend/main.py) 中移除了未使用的导入项 `diskcache` 以及 `typing` 中的 `List` 和 `Dict`。
    *   移除了 `calculate_advanced_chips` 函数中计算但未使用的变量 `bin_width`。
    *   将 `get_stock_data_internal` 中多个重试循环里的未使用变量 `attempt` 替换为下划线 `_`。

### 涉及文件
*   [frontend/src/components/ProfitRatioChart.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/ProfitRatioChart.tsx)
*   [backend/main.py](file:///Users/mac/Documents/code1/stock/backend/main.py)

## 2026-01-31: 修复股票搜索接口对前缀的支持及稳定性

### 修改背景
用户反馈搜索带前缀的股票代码（如 `SH.600000`）时返回空结果。同时发现原有的股票列表接口不稳定。

### 修改内容
1.  **优化搜索匹配逻辑**:
    *   在 [main.py](file:///Users/mac/Documents/code1/stock/backend/main.py) 的 `search_stocks` 接口中，增加了对 `sh.`, `sz.`, `sh`, `sz` 前缀的自动剥离逻辑。
    *   即使输入带前缀，也能正确匹配到纯数字代码的股票。
2.  **提升股票列表获取稳定性**:
    *   将 `get_stock_list` 使用的接口从 `akshare.stock_zh_a_spot_em` 切换为更稳定的 `akshare.stock_info_a_code_name`。
    *   为 `get_stock_list` 增加了 1 小时的缓存，避免频繁请求导致被封禁或性能下降。

### 涉及文件
*   [backend/main.py](file:///Users/mac/Documents/code1/stock/backend/main.py)

## 2026-01-31 23:30: 新增大盘情绪指数与个股财务雷达图

### 修改背景
为了增强系统对市场整体风险的把握能力以及对个股基本面的多维度评估，新增了大盘“恐惧与贪婪”择时指数和个股财务雷达图功能。

### 修改内容

#### 1. 大盘“恐惧与贪婪”择时指数
**后端接口**:
- `GET /api/market/sentiment`: 通过计算上证指数的实时获利比例来衡量市场情绪。
- 逻辑：基于筹码分布算法计算全市场代表性指数（000001.SH）的获利比例，将其划分为“极度恐惧、恐惧、中性、贪婪、极度贪婪”五个等级。

**前端展示**:
- **MarketSentiment 组件**: 位于“筹码分析”标签页底部。
- 特性：进度条可视化展示情绪值，配合颜色提醒和智能操作建议（如“机会大于风险”、“注意风险”等）。

#### 2. 个股财务雷达图与基本面评分
**后端接口**:
- `GET /api/stock/{code}/financial-radar`: 获取个股五个维度的财务评分。
- 维度：盈利能力、成长性、资产质量、现金流、估值水平。
- 逻辑：抓取最新的财务指标并进行标准化处理（0-100分），同时计算综合基本面评分。

**前端展示**:
- **FinancialRadar 组件**: 位于“智能诊断”标签页右侧。
- 特性：使用雷达图直观展示个股各项指标的强弱，帮助用户识别个股是否有扎实的基本面支撑。

#### 3. 系统集成与优化
- **App.tsx**: 
  - 集成了上述两个新组件。
  - 优化了 `fetchExtraData` 逻辑，支持并发获取所有分析数据。
  - 修复了 `renderSmartDiagnosis` 中的 JSX 嵌套结构错误。
  - 移除了未使用的 `TrendingDown` 图标导入。
  - 为 `StockData` 接口增加了索引签名，修复了与 `ChipPeakAnalysis` 组件的类型不匹配问题。

### 涉及文件
- [backend/main.py](file:///Users/mac/Documents/code1/stock/backend/main.py) (新增接口)
- [frontend/src/components/MarketSentiment.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/MarketSentiment.tsx) (新建)
- [frontend/src/components/FinancialRadar.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/FinancialRadar.tsx) (新建)
- [frontend/src/App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx) (集成与修复)

## 2026-01-31 23:45: 修复前端报错并增强系统鲁棒性

### 修改背景
用户反馈在使用新功能时出现多处前端崩溃（Runtime Error），涉及雷达图不显示、板块分析 map 报错以及大盘情绪指数 toFixed 报错。

### 修改内容

#### 1. 修复大盘情绪指数报错
- **前端**: 在 [MarketSentiment.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/MarketSentiment.tsx) 中增加了对 `data.index` 和 `data.timestamp` 的空值校验。使用 `(data.index || 0).toFixed(1)` 确保数值计算安全。
- **后端**: 修改了 [main.py](file:///Users/mac/Documents/code1/stock/backend/main.py) 中的 `/api/market/sentiment` 接口。当数据抓取或计算失败时，不再返回错误对象，而是返回一个包含默认值的合法结构，防止前端解析崩溃。

#### 2. 修复板块分析渲染报错
- **前端**: 在 [App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx) 的 `renderSectorAnalysis` 中增加了 `Array.isArray(sectorRotation)` 检查。
- **后端**: 确保 `/api/sector/rotation` 在捕获异常时返回空列表 `[]` 而非错误信息。

#### 3. 增强财务雷达图数据稳定性
- **后端**: 在 [main.py](file:///Users/mac/Documents/code1/stock/backend/main.py) 的 `/api/stock/{code}/financial-radar` 接口中，增加了对多种股票代码格式（如 `sh600000`, `sz000001`）的适配。
- **后端**: 若财务数据无法获取，接口现在会返回一个标准的“暂无数据”结构（分值为0），确保雷达图能够正常渲染空状态而非报错。

### 涉及文件
- [backend/main.py](file:///Users/mac/Documents/code1/stock/backend/main.py)
- [frontend/src/App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx)
- [frontend/src/components/MarketSentiment.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/components/MarketSentiment.tsx)

## 2026-01-31 23:58: 财务雷达图数据源深度优化与修复

### 修改背景
用户反馈“基本面评分雷达没有数据”。经排查发现，原有的 `akshare` 财务指标接口在某些股票代码下返回空数据或格式不兼容，导致归一化计算失败。

### 修改内容

#### 1. 引入更稳定的同花顺财务摘要接口
- **后端**: 在 [main.py](file:///Users/mac/Documents/code1/stock/backend/main.py) 的 `/api/stock/{code}/financial-radar` 接口中，将 `ak.stock_financial_abstract_ths` 设为首选数据源。该接口数据更全面且更新及时。
- **多级回退机制**: 如果同花顺接口失败，系统会依次尝试 `stock_financial_analysis_indicator` (含多种代码格式适配) 和 `stock_financial_analysis_indicator_em`，确保最大程度获取数据。

#### 2. 增强数据解析与鲁棒性
- **通用解析函数**: 实现了 `parse_val` 工具函数，自动处理百分号（%）、单位“亿”、“万”以及布尔值，统一转换为数值格式。
- **字段动态适配**: 针对不同数据源（ths, ak, em）的列名差异（如“净资产收益率”与“净资产收益率(%)”）进行了映射兼容。
- **归一化算法优化**: 改进了 ROE、净利润增长、负债率等指标的评分映射逻辑，使其在不同数据源下表现一致。

#### 3. 修复代码质量问题
- **缩进修复**: 修复了 `main.py` 中因多次修改导致的 API 函数内部缩进错误。
- **空值处理**: 确保在所有数据源均不可用时，返回结构完整的默认零值对象，彻底杜绝前端“无数据”导致的显示异常。

### 涉及文件
- [backend/main.py](file:///Users/mac/Documents/code1/stock/backend/main.py)
