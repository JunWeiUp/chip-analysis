# 修改记录 (AGENTS.md)

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

## 2026-01-31: 优化 600000 股票数据展示与 BaoStock 适配

### 涉及文件
*   [frontend/src/App.tsx](file:///Users/mac/Documents/code1/stock/frontend/src/App.tsx)
*   [backend/main.py](file:///Users/mac/Documents/code1/stock/backend/main.py)
*   [backend/cache/*](file:///Users/mac/Documents/code1/stock/backend/cache/) (清理操作)
