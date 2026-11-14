# Day 06 · Data Ingestion Engineering

**Bài nộp cá nhân** · **Nguyễn Bảo Trung (`2A202604013` · Lớp `L3B`)**

Pipeline Data Ingestion bằng Python cho 4 bài báo arXiv CC BY 4.0 (`36/42` trang nội dung hợp lệ), so sánh có kiểm soát 3 chiến lược (`A` fixed-size `500/50`, `B` paragraph-aware `650`, `C` như `B` + pre-retrieval metadata filter) với embedding thật `gemini-embedding-001` (`3072` chiều) và vector store `qdrant-client` lưu bền vững trên đĩa.

| 📄 Dữ liệu | 🧩 Chiến lược | 🎯 Benchmark |
|---|---|---|
| **36/42 trang nội dung** (4 PDF arXiv CC BY 4.0) | **3 cấu hình** (`A` / `B` / `C`) | **20 câu hỏi** (`15 dev` / `5 test`, `6` câu filter) |

## 1. Đường dẫn dữ liệu & cấu trúc bài nộp

- Thông tin học viên: [STUDENT.md](STUDENT.md)
- Dữ liệu PDF nguồn: [starter/data/](starter/data/) (`2404.04302.pdf`, `2506.16988.pdf`, `2506.18027.pdf`, `2603.05698.pdf`)
- Manifest & quy tắc tính trang: [data/manifest.json](data/manifest.json)
- Bộ câu hỏi benchmark & gold evidence: [benchmark/questions.json](benchmark/questions.json)
- Mã nguồn & unit test: [starter/src/](starter/src/), [starter/tests/test_pipeline.py](starter/tests/test_pipeline.py)
- Nhật ký cấu hình & chỉ số A/B/C: [artifacts/experiment_log.csv](artifacts/experiment_log.csv)
- Kết quả chạy & audit thủ công: [results/ingestion_log.json](results/ingestion_log.json), [results/traces_all.json](results/traces_all.json), [results/coherence_samples.json](results/coherence_samples.json), [results/test_audit_samples.json](results/test_audit_samples.json), [results/manual_audit.json](results/manual_audit.json)
- Báo cáo cá nhân: [report/individual/2A202604013_NguyenBaoTrung.md](report/individual/2A202604013_NguyenBaoTrung.md)

## 2. Cách setup & chạy tái lập

Yêu cầu Python `3.11–3.14` và `GEMINI_API_KEY` (không commit `.env`; chỉ số vector cục bộ `starter/results/qdrant_db/` được gitignore và tự động tạo lại khi chạy pipeline). Xem thêm [starter/README.md](starter/README.md).

```bash
cd starter
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env               # điền GEMINI_API_KEY (tuỳ chọn EMBEDDING_MODEL=gemini-embedding-001)
```

Chạy kiểm thử đơn vị (offline, không gọi API):

```bash
(cd starter && .venv/bin/python -m pytest -q)
```

Chạy toàn bộ pipeline (parse PDF, chunk A/B/C, embed qua Gemini API, lưu Qdrant local, kiểm tra nạp lại không trùng, đánh giá 20 câu và xuất `results/`):

```bash
(cd starter && .venv/bin/python -m src.run)
```

## Tài liệu hướng dẫn

| Bạn cần tìm | Tài liệu |
|---|---|
| Setup baseline PostgreSQL + pgvector | [starter/README.md](starter/README.md) |
| Các bước làm và cách tính chỉ số | [GUIDE.md](GUIDE.md) |
| Tiến độ trong 4 giờ | [CHECKPOINTS.md](CHECKPOINTS.md) |
| Năm tiêu chí benchmark và bộ câu hỏi | [BENCHMARK.md](BENCHMARK.md) |
| Tiêu chí chấm điểm | [RUBRIC.md](RUBRIC.md) |
| Quy tắc sử dụng dữ liệu, AI và API | [RULES.md](RULES.md) |
| Cách nộp bài | [SUBMISSION.md](SUBMISSION.md) |

[data/format-example.json](data/format-example.json) chỉ minh họa định dạng, không phải corpus đủ điều kiện. Phần xử lý và benchmark do bạn tự hiện thực; không bắt buộc xây chatbot.
