# Pipeline · Local Qdrant + Gemini Embeddings

Thư mục thực thi cho bài làm Day 06 (`2A202604013` — Nguyễn Bảo Trung). Pipeline đọc PDF từ `starter/data/`, chuẩn hóa Unicode và ranh giới đoạn, chạy 3 chiến lược (`A` fixed-size `500/50`, `B` paragraph-aware `650`, `C` như `B` + metadata pre-filter), tạo vector bằng Google Gemini API (`gemini-embedding-001`, `3072` chiều, `RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY`), lưu bền vững bằng `qdrant-client` trên đĩa và xuất toàn bộ trace/log ra `../results/`.

*(Các file `docker-compose.yml`, `init.sql` và `scripts/check_setup.py` là khung PostgreSQL + pgvector ban đầu của đề, không dùng trong pipeline Qdrant đã chọn.)*

## Cấu trúc

```text
starter/
├── requirements.txt       # pypdf, python-dotenv, google-genai, qdrant-client, pytest
├── .env.example           # mẫu biến môi trường; copy thành .env (không commit .env)
├── data/                  # 4 bài báo arXiv CC BY 4.0 (2404.04302, 2506.16988, 2506.18027, 2603.05698)
├── src/
│   ├── models.py          # dataclass DocumentPage, Chunk
│   ├── ingestion.py       # đọc PDF, normalize_text, gán section_type cấp trang, page_logs
│   ├── chunking.py        # fixed_size và by_paragraph
│   ├── embeddings.py      # MyEmbedder gọi Gemini API (không fallback mock)
│   ├── store.py           # QdrantVectorStore local disk, cosine, filter whitelist, xóa chunk cũ
│   ├── evaluation.py      # evaluate_query (Hit@1, Hit@3, split_across_chunks) và summarize
│   └── run.py             # chạy A/B/C, kiểm tra reload idempotency, xuất ../results/*.json
├── tests/
│   └── test_pipeline.py   # 5 unit test offline
└── results/
    └── qdrant_db/         # Qdrant local storage (gitignored, tự tạo lại khi chạy src.run)
```

## 1. Cài đặt môi trường

Cần Python **3.11–3.14** và khóa `GEMINI_API_KEY`. Từ thư mục `starter/`:

```bash
cd starter
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env               # điền GEMINI_API_KEY
```

Biến môi trường sử dụng trong `starter/.env`:
- `GEMINI_API_KEY`: bắt buộc cho `python -m src.run` (không dùng cho `pytest`).
- `EMBEDDING_MODEL`: tùy chọn, mặc định `gemini-embedding-001`.

## 2. Chạy test và pipeline

Kiểm thử đơn vị offline (không cần `GEMINI_API_KEY`, không gọi mạng):

```bash
python -m pytest -q
```

Chạy toàn bộ pipeline (tái tạo `starter/results/qdrant_db/` và ghi `../results/ingestion_log.json`, `../results/traces_all.json`, `../results/coherence_samples.json`, `../results/test_audit_samples.json`; không ghi đè `../results/manual_audit.json`):

```bash
python -m src.run
```

Chi tiết tham số, số liệu dev/test và đánh giá thủ công xem [../report/individual/2A202604013_NguyenBaoTrung.md](../report/individual/2A202604013_NguyenBaoTrung.md).
