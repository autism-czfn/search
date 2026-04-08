# Autism Search 自闭症内容搜索服务

基于 PostgreSQL + pgvector 的混合搜索 REST API，支持语义搜索与关键词搜索，覆盖多来源自闭症相关内容。 

---

## 功能概述

- **语义搜索**：将查询文本转换为向量，通过余弦相似度在数据库中匹配最相关内容
- **关键词搜索**：基于 PostgreSQL 全文检索（tsvector）进行精确匹配
- **混合模式**：同时运行两种搜索，归一化评分后加权合并，返回最优排名结果
- **AI 摘要**：调用 `claude -p` 对搜索结果自动生成简洁摘要
- **多源支持**：涵盖 PubMed、Reddit、DOAJ、ClinicalTrials、CORE 等多个数据来源

---

## 搜索原理

### 第一步：语义搜索

用户查询文本通过本地 `fastembed` 模型（`nomic-ai/nomic-embed-text-v1.5`）转换为 768 维向量，与数据库中预存的文章向量进行余弦距离计算，找出语义最接近的内容。

```
用户查询 → fastembed → 向量[768] → pgvector 余弦相似度查询 → 语义结果
```

### 第二步：关键词搜索

同时对 `title`、`description`、`content_body` 三个字段执行 PostgreSQL 全文检索，基于词频和相关性打分排序。

```
用户查询 → PostgreSQL tsvector @@ tsquery → 关键词结果
```

### 第三步：混合合并与重排

将两组结果合并去重，分别对语义分数和关键词分数进行归一化（0~1），按以下权重计算最终分数：

```
combined_score = 0.7 × semantic_score + 0.3 × keyword_score
```

若语义搜索不可用（向量未生成），自动降级为纯关键词模式。

### 第四步：AI 摘要

取排名前 5 的结果，调用 `claude -p` 生成 2~4 句话的综合摘要，随搜索结果一并返回。若 Claude 调用失败，仍正常返回搜索结果，摘要字段为 `null`。

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

复制配置模板并填写数据库地址：

```bash
cp .env.example .env
```

编辑 `.env`：

```
DATABASE_URL=postgresql://用户名:密码@localhost:5432/autism_crawler
DEFAULT_RESULT_LIMIT=10
MAX_RESULT_LIMIT=50
LOG_LEVEL=INFO
```

### 启动服务

```bash
./setup.sh
```

选择 `1) Start / Restart service`，服务将在端口 **3001** 启动。

或直接运行：

```bash
uvicorn src.main:app --reload --port 3001
```

---

## API 接口

### `GET /api/search` — 搜索

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `q` | string | ✅ | 搜索关键词 |
| `limit` | int | 否 | 返回条数（默认 10，最多 50）|
| `source` | string | 否 | 按来源过滤（如 `pubmed`、`reddit`）|
| `days` | int | 否 | 只返回最近 N 天内发布的内容 |

**请求示例：**
```bash
curl "http://localhost:3001/api/search?q=自闭症症状&source=pubmed&days=90"
```

**返回示例：**
```json
{
  "results": [
    {
      "id": 429,
      "source": "pubmed",
      "title": "Social Difficulties in Youth with Autism...",
      "url": "https://pubmed.ncbi.nlm.nih.gov/...",
      "description": "...",
      "published_at": "2024-03-15T00:00:00Z",
      "semantic_score": 0.87,
      "keyword_score": 0.65,
      "combined_score": 0.80
    }
  ],
  "total": 10,
  "search_mode": "hybrid",
  "search_time_ms": 254,
  "summary": "自闭症的核心症状包括社交障碍、重复性行为以及感知觉异常...",
  "llm_time_ms": 9298
}
```

`search_mode` 取值：
- `hybrid`：语义 + 关键词混合模式（推荐）
- `keyword_only`：纯关键词模式（向量未就绪时自动降级）

---

### `GET /api/stats` — 数据库统计

```bash
curl "http://localhost:3001/api/stats"
```

```json
{
  "total_items": 58320,
  "embedded_items": 54100,
  "items_by_source": {
    "pubmed": 42000,
    "reddit": 16320
  },
  "last_collected_at": "2026-04-07T22:00:00Z",
  "last_embedded_at": "2026-04-07T23:00:00Z"
}
```

---

### `GET /api/health` — 健康检查

```bash
curl "http://localhost:3001/api/health"
```

```json
{ "status": "ok", "db": "connected" }
```

---

## 项目结构

```
autism-search/
├── src/
│   ├── api/
│   │   ├── models.py        # 请求/响应 Pydantic 数据模型
│   │   └── routes.py        # FastAPI 路由：/search、/stats、/health
│   ├── search/
│   │   ├── semantic.py      # pgvector 语义搜索
│   │   ├── keyword.py       # PostgreSQL 全文关键词搜索
│   │   └── hybrid.py        # 归一化、合并、重排
│   ├── llm/
│   │   └── summarize.py     # 调用 claude -p 生成摘要
│   ├── embedder.py          # fastembed 查询向量化
│   ├── db.py                # asyncpg 数据库连接池
│   ├── config.py            # 环境变量配置
│   └── main.py              # FastAPI 应用入口
├── setup.sh                 # 服务管理脚本（启动/停止/状态/URL）
├── requirements.txt         # Python 依赖
├── pyproject.toml           # 包配置
└── .env.example             # 环境变量模板
```

---

## 数据来源

| 来源 | 类型 |
|------|------|
| PubMed / Europe PMC | 医学学术论文 |
| DOAJ / CORE / OpenAlex | 开放获取学术内容 |
| ClinicalTrials.gov | 临床试验数据 |
| Reddit | 社区讨论帖子 |
| bioRxiv / Semantic Scholar | 预印本与引用数据 |

---

## 依赖说明

| 包 | 用途 |
|----|------|
| `fastapi` | Web 框架 |
| `uvicorn` | ASGI 服务器 |
| `asyncpg` | 异步 PostgreSQL 驱动 |
| `pgvector` | 向量数据类型支持 |
| `fastembed` | 本地文本向量化（无需 API Key）|
| `pydantic` | 数据校验与序列化 |
