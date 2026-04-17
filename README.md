# Autism Search 自闭症内容搜索服务

基于 PostgreSQL + pgvector 的混合搜索 REST API，在语义 + 关键词检索之上叠加 `claude -p` 智能体回答、SSE 流式响应、家长日志个性化、以及围绕 `mzhu_test_*` 表的分析报告（洞察、周报、临床报告）。

---

## 功能概述

- **混合搜索**：`fastembed` 向量相似度 + PostgreSQL 全文检索，归一化评分后加权合并
- **意图分类**：基于同义词/规则的意图识别（BEHAVIORAL / MEDICAL / SAFETY / GENERAL），替代脆弱的关键词匹配，安全场景自动升级
- **5 因子排序**：source_authority (0.40) + trigger_match (0.30) + context_match (0.15) + language_match (0.10) + recency (0.05)，权重可通过 `config/ranking.json` 调整
- **源注册表**：`config/sources.json` 定义 28 个可信来源（Tier 1/2/3），替代硬编码 boost 列表，驱动权限过滤与排序
- **实时站内搜索**：对 28 个官方站点直接搜索（JSON API / HTML 抓取 / Sitemap），不依赖 Google/Bing
- **搜索路由引擎**：根据意图分类和本地结果质量，智能选择 LOCAL_ONLY / HYBRID / LIVE_ONLY 模式；安全场景强制触发实时搜索
- **本地上下文准入**：4 项质量门控（相关性 ≥0.75、质量 ≥0.6、时效 ≥0.5、覆盖 ≥2 条），确保本地结果质量
- **证据缓存**：按 normalized_query_hash + language + time_bucket 缓存搜索证据，支持 trigger_key 维度失效
- **多语言搜索**：查询翻译 → 本地 DB 目标语言检索 → 结果摘要回译英文，支持 fr/de/ja/es
- **触发策略**：智能判断是否需要搜索（新触发因素、安全敏感、缓存过期等），避免重复查询
- **策展证据**：面向 Insights 标签页的干净证据卡片（仅可信来源，无原始分数泄露）
- **Claude 智能体回答**：调用 `claude -p`，开放 `Read + Bash` 工具；智能体可读入预取结果的临时 JSON 文件，必要时自主调用 CLI 搜索工具继续迭代
- **流式响应**：`/api/search/stream` 基于 SSE 分阶段推送 `metadata → stage → results → agent_activity → summary → done`
- **个性化上下文**：从用户 DB 中读取近 30 天日志与每日检查，拼成精简上下文注入 LLM 提示
- **安全检测**：两层策略——意图分类器（同义词/规则语义匹配）+ 提示词注入让 Claude 以 `⚠ SAFETY:` 前缀响应
- **分析接口**：`/api/insights`、`/api/weekly-summary`、`/api/clinician-report` 三个围绕 `mzhu_test_logs / _interventions / _daily_checks / _summaries` 的端点，全部走 SQL 聚合 + LLM 叙述，结果带缓存
- **PubMed 直连**：`src.tools.pubmed` 封装 E-utilities，智能体可在本地库证据不足时实时查询
- **HTTPS + Chrome PNA**：`setup.sh` 生成自签证书，`PrivateNetworkAccessMiddleware` 为公网页面访问内网 API 放行

---

## 搜索架构

```
User Query
   ↓
[1] Intent Classification (BEHAVIORAL / MEDICAL / SAFETY / GENERAL)
   ↓
[2] Trigger Policy (should_search? — cache check, safety override)
   ↓
[3] Routing Engine (LOCAL_ONLY / HYBRID / LIVE_ONLY)
   │
   ├─ LOCAL path:
   │   ├─► fastembed (nomic-ai/nomic-embed-text-v1.5, 768d)
   │   │       └─► pgvector 余弦相似度
   │   ├─► PostgreSQL tsvector 全文检索
   │   └─► Local Context Qualification (4-factor gate)
   │
   ├─ LIVE path:
   │   └─► Site Search Engine (28 sites: API / HTML / Sitemap)
   │
   └─ MULTILINGUAL path:
       └─► translate query → local DB search → translate snippets
   ↓
[4] merge_and_rerank (5-factor ranking formula)
        source_authority × 0.40 + trigger_match × 0.30
      + context_match × 0.15 + language_match × 0.10 + recency × 0.05
   ↓
[5] Evidence Cache (store/retrieve by query_hash + lang + date)
   ↓
[6] safety check + fetch_log_context (用户 DB, ≥5 条日志才注入)
   ↓
[7] run_agent (claude -p, Read+Bash, 60s 超时)
   │   └─ 读 /tmp/autism_agent_<uuid>.json
   │   └─ 按需调用 python -m src.tools.search | src.tools.pubmed
   │
   └─► (失败回退) summarize() ← 单次 claude -p
```

若语义向量不可用（`embed_query` 返回 None），自动降级为 `keyword_only` 模式。

流式端点 `/api/search/stream` 使用 `run_agent_stream`，会额外把 Claude 的每次工具调用转为 `agent_activity` 事件推送给前端。

---

## 快速开始

### 1. 准备虚拟环境与依赖

`setup.sh` 假设虚拟环境在 `/home/test/.virtualenvs/search`。菜单选项 `5) Setup cert / .env / deps` 会一次性完成：
- 生成自签证书到 `../certs/{cert,key}.pem`
- 交互式创建 `.env`（同时配置 `DATABASE_URL` 和 `USER_DATABASE_URL`）
- `pip install -r requirements.txt`

也可以手动安装：

```bash
pip install -r requirements.txt
cp .env.example .env        # 然后编辑
```

### 2. 环境变量（`.env`）

```
DATABASE_URL=postgresql://user:pass@localhost:5432/autism_crawler
USER_DATABASE_URL=postgresql://user:pass@localhost:5432/autism_users
COLLECT_BASE_URL=https://localhost:18001
DEFAULT_RESULT_LIMIT=10
MAX_RESULT_LIMIT=50
LOG_LEVEL=INFO
NCBI_API_KEY=               # 可选：PubMed E-utilities 配额
RANKING_CONFIG_PATH=        # 可选：自定义排序权重 JSON 路径（默认 config/ranking.json）
TRANSLATION_API=            # 可选："google" | "deepl"，多语言查询翻译
TRANSLATION_API_KEY=        # 可选：翻译 API 密钥（无则降级为英文查询搜非英文内容）
```

- `DATABASE_URL` — 爬虫库，读取 `crawled_items`、`embedding` 等
- `USER_DATABASE_URL` — 用户库，读取 `mzhu_test_logs / _interventions / _daily_checks / _insights_cache / _summaries`；留空则个性化与分析端点全部降级为 503
- `COLLECT_BASE_URL` — collect 服务地址，周报生成后 POST 回写

### 3. 数据库迁移

```bash
psql "$USER_DATABASE_URL" -f migrations/001_evidence_cache.sql
```

创建 `evidence_cache` 表（查询缓存）并为 `mzhu_test_insights_cache` 添加 `cache_type` 列。

### 4. 启动服务

```bash
./setup.sh      # 菜单驱动：start / stop / status / url / setup
```

或直接跑 uvicorn（HTTPS）：

```bash
uvicorn src.main:app --host 0.0.0.0 --port 3001 \
  --ssl-certfile ../certs/cert.pem \
  --ssl-keyfile  ../certs/key.pem
```

Swagger 文档：`https://<host>:3002/docs`

---

## API 接口

所有路由都挂在 `/api` 前缀下。

### `GET /api/search`

同步接口，返回完整 JSON（含 LLM 摘要）。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `q` | string | ✅ | 查询文本 |
| `limit` | int |  | 返回条数（默认 10，最多 50）|
| `source` | string |  | 按来源过滤（如 `pubmed`、`reddit`）|
| `days` | int |  | 只返回最近 N 天内发布的内容 |
| `lang` | string |  | 搜索语言（默认 `en`；`all` 搜全部语言）|
| `audience` | string |  | 受众过滤：`parent` / `clinician` |
| `refresh` | bool |  | 跳过缓存强制重新搜索 |

返回体包含 `results`、`search_mode`（`hybrid` / `keyword_only`）、`search_time_ms`、`summary`、`llm_time_ms`、`agent_iterations`、`safety_flag`。

### `GET /api/search/stream`

相同参数，返回 `text/event-stream`。事件序列：

```
metadata       × 1    (safety_flag)
stage          × 4-5  (embedding → semantic → keyword → merge → agent)
results        × 1    (搜索卡片，先于 LLM 发送)
agent_activity × 0-N  (智能体每次工具调用)
summary        × 0-1  (最终答案；两层回退后仍可能缺失)
done           × 1    (总耗时 / agent_iterations / llm_time_ms)
```

### `GET /api/sources`

返回源注册表中所有活跃来源列表（source_id, organization_name, authority_tier, source_type, audience_type, language, country, domain）。

### `GET /api/evidence/{chunk_id}`

按 chunk_id 返回完整证据详情（来源元数据、全文、权威等级）。实时搜索结果（is_live_result=true）无本地存储，UI 应直接链接原始 URL。

### `GET /api/insights/evidence`

按触发因素/结果获取策展证据卡片（仅可信来源，无原始分数）。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `trigger` | string | ✅ | 触发因素名称 |
| `outcome` | string |  | 结果名称 |
| `limit` | int |  | 返回条数（1-10，默认 5）|

### `GET /api/insights/full`

带证据和建议的完整洞察报告。对 top 3 模式附加策展证据卡片 + LLM 生成的可操作建议。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `days` | int |  | 分析天数（默认 30）|
| `refresh` | bool |  | 强制刷新缓存 |

### `GET /api/stats`

爬虫库统计（总数、已向量化数、按来源分布、最新采集/向量化时间、缓存命中率）。

### `GET /api/health`

数据库 ping。

### `GET /api/insights?days=30&refresh=false`

纯 SQL 聚合洞察，缓存 1 小时（`mzhu_test_insights_cache`）。返回：
- `top_triggers` / `top_outcomes` — 触发因素与结果分布
- `patterns` — trigger×outcome 共现率，附 `confidence_level`：`strong_pattern` (≥80% & n≥20) / `possible_pattern` (≥50% & n≥10) / `insufficient_data`
- `intervention_effectiveness` — 已采纳干预前后 meltdown 发生率对比（最长 14 天窗口）
- `daily_check_trends` — 每日检查平均值 + 低睡眠日 vs 其他日的 meltdown 相关性
- `log_count` / `date_range`

### `GET /api/weekly-summary`

当前自然周的周报，24 小时缓存。先跑 SQL 计算统计，再让 `claude -p` 用自然语言叙述（不允许编造数字），最后 POST 到 collect 做持久化。

### `GET /api/clinician-report?days=90&refresh=false`

面向就诊的结构化报告，2 小时缓存：
- 日期范围 + 事件频率（总数 + 分周）
- Top triggers / outcomes / patterns / 干预效果
- `daily_check_summary` — 含周趋势
- `key_concerns_text` — LLM 生成的「关注点」段落，以 SQL 统计 + 最高频 trigger 的 top 3 证据结果为依据

---

## 项目结构

```
search/
├── config/
│   ├── sources.json               # 28 个 MVP 来源定义（Tier 1/2/3，含多语言）
│   └── ranking.json               # 5 因子排序权重配置
├── migrations/
│   └── 001_evidence_cache.sql     # evidence_cache 表 + insights_cache 扩展
├── src/
│   ├── main.py                    # FastAPI app + lifespan + PNA 中间件
│   ├── config.py                  # pydantic-settings (.env)
│   ├── db.py / user_db.py         # asyncpg 连接池（双库）
│   ├── embedder.py                # fastembed 单例
│   ├── safety.py                  # 两层安全检测（意图分类器 + LLM）
│   ├── api/
│   │   ├── routes.py              # 所有 API 端点
│   │   ├── stream.py              # SSE 生成器
│   │   └── models.py              # Pydantic 请求/响应模型
│   ├── search/
│   │   ├── semantic.py            # pgvector 余弦相似度
│   │   ├── keyword.py             # tsvector 全文检索（支持多语言 FTS config）
│   │   ├── hybrid.py              # 归一化 + 合并重排（registry-driven boost）
│   │   ├── ranking.py             # 5 因子排序公式（搜索结果 + 证据排序）
│   │   ├── cache.py               # 证据缓存（query_hash + trigger_key）
│   │   ├── intent_classifier.py   # 意图分类（BEHAVIORAL/MEDICAL/SAFETY/GENERAL）
│   │   ├── trigger_policy.py      # 搜索触发策略
│   │   ├── local_qualifier.py     # 本地上下文 4 因子准入门控
│   │   ├── live_fallback.py       # 搜索路由引擎（LOCAL/HYBRID/LIVE 决策）
│   │   ├── site_search.py         # 28 站实时站内搜索引擎
│   │   ├── multilingual.py        # 多语言搜索（翻译 → 检索 → 回译）
│   │   └── pubmed.py              # PubMed E-utilities 客户端
│   ├── sources/
│   │   └── registry.py            # 源注册表单例（加载 config/sources.json）
│   ├── evidence/
│   │   ├── search.py              # 策展证据检索（仅可信来源）
│   │   └── sources.py             # 证据来源可信度判定
│   ├── llm/
│   │   ├── agent.py               # claude -p + Read/Bash 智能体
│   │   ├── agent_stream.py        # SSE 版智能体
│   │   └── summarize.py           # 单次 claude -p 回退
│   ├── tools/
│   │   ├── search.py              # CLI: 本地混合搜索（智能体 Bash 调用）
│   │   └── pubmed.py              # CLI: 实时 PubMed 搜索
│   └── analytics/
│       ├── patterns.py            # /insights + 个性化上下文
│       ├── summary.py             # /weekly-summary
│       ├── clinician.py           # /clinician-report
│       └── daily_checks.py        # mzhu_test_daily_checks SQL 聚合
├── setup.sh                       # 服务管理 + cert/.env/deps 安装
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## 数据来源

28 个 MVP 来源，按可信等级分层：

| 等级 | 来源 | 数量 |
|------|------|------|
| Tier 1 | PubMed, NICE, NHS, CDC, NIMH, AAP, CDC Data | 7 |
| Tier 2 | Mayo Clinic, Europe PMC, Semantic Scholar, OpenAlex, Crossref, bioRxiv, DOAJ, ClinicalTrials.gov, CORE | 9 |
| Tier 3 | Autism Society, ASF, SPARK, Autism Navigator, Autism Spectrum News, Spectrum News, ASAN, Wikipedia | 8 |
| 非英文 | France HAS (fr), Germany Neuro (de), Japan DDIS (ja), Spain Autismo (es) | 4 |

爬取与入库由姊妹项目 `collect` 负责；本服务只读 `crawled_items` 与 `embedding`。实时搜索通过站内 API/HTML 直接查询各站点。

---

## 核心设计原则

1. **意图优先于关键词** — 语义意图分类替代脆弱的精确关键词匹配
2. **安全优先于一切** — 安全场景强制触发实时搜索，永不仅依赖本地数据
3. **实时确保正确性** — 直接搜索官方来源，不依赖第三方搜索引擎
4. **本地实现个性化** — 用户日志上下文注入 LLM 提示
5. **优雅降级，不静默失败** — 实时失败回退本地 + 警告；本地失败仍返回实时；均失败返回安全兜底消息

---

## 依赖说明

| 包 | 用途 |
|----|------|
| `fastapi` / `uvicorn[standard]` | Web 框架与 ASGI 服务器 |
| `asyncpg` / `pgvector` | PostgreSQL 异步驱动 + 向量类型 |
| `fastembed` | 本地文本向量化（nomic-embed-text-v1.5，无需 API Key）|
| `pydantic` / `pydantic-settings` | 数据校验与 `.env` 读取 |
| `httpx` | PubMed / collect / 站内搜索的异步 HTTP 客户端 |
| `claude` CLI | 外部依赖——智能体 / 摘要 / 周报 / 临床报告都 shell out 到它，不在 PATH 时自动降级 |
