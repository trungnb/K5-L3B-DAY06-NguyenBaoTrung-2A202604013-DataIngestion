# Báo cáo cá nhân — Day 06: Data Ingestion Engineering

- Họ và tên: Nguyễn Bảo Trung
- MSSV: 2A202604013
- Lớp: L3B
- Nhóm: Cá nhân (`Data Ingestion Engineering`)
- Repo bài nộp: [K5-L3B-DAY06-NguyenBaoTrung-2A202604013-DataIngestion](https://github.com/trungnb/K5-L3B-DAY06-NguyenBaoTrung-2A202604013-DataIngestion)
- Nguồn dữ liệu: 4 bài báo arXiv về RAG phát hành theo giấy phép **CC BY 4.0** (`arxiv-rag-ccby4-mini`, tải về `starter/data/`)

---

## 1. Dữ liệu và pipeline

### Corpus & tiền kiểm tra bản quyền
Do sách `Grokking Machine Learning` có bản quyền Manning *All rights reserved* (cấm ingest/tái sử dụng cho AI/benchmark theo `RULES.md`), pipeline được xây dựng trên corpus **4 bài báo arXiv CC BY 4.0** (kiểm tra trực tiếp trường license trên trang arXiv):

| Document ID | Chủ đề (`topic`) | Năm (`year`) | Tổng trang PDF | Trang đưa vào index (`content_pages`) | Trang loại khỏi index |
|---|---|---:|---:|---|---|
| `2404.04302` (CBR-RAG) | `legal_qa` | 2024 | 17 | 1–15 | 16–17 (`references_only`) |
| `2506.16988` (RAGentA) | `attributed_qa` | 2025 | 5 | 1–4 | 5 (`references_only`) |
| `2506.18027` (PIER-QA) | `pdf_qa` | 2026 | 14 | 1–8, 12–14 | 9–11 (`references_only` & trang mở đầu references) |
| `2603.05698` (GraphRAG vs RGB) | `kg_robustness` | 2026 | 6 | 1–6 | Không |

Tổng cộng có **42 trang vật lý**, lọc qua [data/manifest.json](../../data/manifest.json) còn **36 trang nội dung** (**117,393 ký tự chuẩn hóa**, thời gian parse `0.666s`).

### Pipeline Ingestion, Normalization & Incremental Load
Pipeline nằm trong [starter/src/](../../starter/src):
- **Trích xuất & chuẩn hóa ([starter/src/ingestion.py](../../starter/src/ingestion.py)):** Dùng `pypdf.PdfReader` chế độ mặc định (loại bỏ `extraction_mode="layout"` vì gộp ngang 2 cột IEEE/ACM). Hàm `normalize_text` thực hiện chuẩn hóa Unicode NFKC, nối từ bị gạch nối cuối dòng (`word-\nnext`), và tái tạo ranh giới đoạn `\n\n` từ các dòng đơn của `pypdf` dựa trên heading và độ dài dòng. Hàm `infer_page_section` gán tất định nhãn `section_type` (`intro_related`, `methodology`, `evaluation`, `appendix`) theo heading và tỷ lệ vị trí ký tự `<= 0.45` trên trang (không sử dụng thông tin nhãn câu hỏi).
- **Lỗi PDF thực tế còn tồn tại:** Một số từ dính khoảng trắng từ `pypdf` trong bảng/từ viết tắt (`forCUDA`, `SURY A_OCR_DPI`, `ACCREPRESENTS`, `35.35100.0085.85` ở Table II của `2603.05698` p5), cùng running header lặp ở đầu trang (`2 N. Wiratunga et al.`, venue line SIGIR) và một số trường hợp ngắt trang giữ lại dấu gạch nối (`Lan-\n\nguage`).
- **Embedding, Vector Store & Lệnh chạy ([starter/src/embeddings.py](../../starter/src/embeddings.py), [starter/src/store.py](../../starter/src/store.py)):** Sử dụng Gemini `gemini-embedding-001` (`3072` chiều, `google-genai==3.3.0`) với `task_type="RETRIEVAL_DOCUMENT"` cho chunk và `RETRIEVAL_QUERY` cho câu hỏi (gọi API thật, retry backoff 4 lần, không dùng mock). Vector store dùng `qdrant-client==1.20.0` khởi tạo qua `QdrantClient(path="starter/results/qdrant_db")` với 3 collection riêng biệt `day6_A`, `day6_B`, `day6_C` (`Distance.COSINE`, exact scan trên local Python mode) thay cho Docker server để phù hợp giới hạn phần cứng (Apple M1, 8 GB RAM). Môi trường & lệnh chạy: `pypdf==6.21.4`, `python-dotenv==1.2.4`, `pytest==9.2.0`; pipeline chạy xuất `results/` lúc `2026-11-14T10:05+07:00` trên Python `3.12.16`, `.venv` hiện tại kiểm thử lại trên Python `3.13.5` (đều thuộc dải `3.11–3.14` của starter) bằng `(cd starter && .venv/bin/python -m pytest -q)` và `(cd starter && .venv/bin/python -m src.run)`.
- **Idempotency & cập nhật tài liệu:** Mỗi chunk có UUID5 tất định từ `(strategy, chunk_id)`. Trước khi upsert tài liệu, `QdrantVectorStore.upsert` xóa toàn bộ điểm cũ của `document_id` đó (`Filter` theo `document_id`). Thử nghiệm reload `2404.04302` trong [results/ingestion_log.json](../../results/ingestion_log.json) giữ nguyên `272 -> 272` points (`no_duplicates: true`); hành vi xóa chunk thừa khi tài liệu thay đổi giảm số chunk được kiểm chứng trong [starter/tests/test_pipeline.py](../../starter/tests/test_pipeline.py).

**Schema Metadata lưu trên mỗi chunk & Qdrant payload (`8` trường):**

| Trường | Kiểu | Nguồn tạo | Lý do thiết kế |
| :--- | :--- | :--- | :--- |
| `document_id` | `str` | `manifest.json` (`source_id`) | Định danh bài báo ổn định cho truy vết nguồn, xóa/upsert lại theo tài liệu và kiểm soát rò rỉ. |
| `source` | `str` | `manifest.json` (`local_path`) | Đường dẫn file PDF gốc để đối chiếu trực tiếp trang nguồn khi kiểm định. |
| `page` | `int` | Chỉ số trang PDF (1-based) | Phục vụ đánh giá Evidence Hit theo đúng trang vàng và trích dẫn trang chính xác. |
| `year` | `int` | `manifest.json` (năm bản arXiv trang 1) | Cho phép lọc thời gian (ví dụ các hệ thống công bố năm `2026`) trong Chiến lược C. |
| `topic` | `str` | `manifest.json` (chủ đề bài báo) | Phân nhóm bài toán (`legal_qa`, `attributed_qa`, `pdf_qa`, `kg_robustness`) để thu hẹp phạm vi tìm kiếm khi cần. |
| `section_type` | `str` | `infer_page_section` (tất định từ heading & vị trí trang) | Phân tách ngữ cảnh (`intro_related`, `methodology`, `evaluation`, `appendix`) giúp Chiến lược C chặn nhiễu giữa các phần cùng bài. |
| `normalized_chars` | `int` | `len(norm)` sau `normalize_text` | Theo dõi mật độ ký tự sạch trên từng trang để kiểm soát chất lượng extraction và phát hiện trang rỗng/bất thường. |
| `chunk_index` | `int` | Thứ tự tăng dần trên trang trong `chunking.py` | Giữ thứ tự gốc các chunk trong cùng trang để phục vụ ghép ngữ cảnh hoặc audit tính liên kết. |

---

## 2. Cấu hình và benchmark

### Ba chiến lược so sánh ([artifacts/experiment_log.csv](../../artifacts/experiment_log.csv))
- **Chiến lược A (Baseline):** `fixed_size(size=500, overlap=50)`, không metadata filter (`use_filter=false`). Sinh **272 chunks** (min `68`, median `500`, mean `474.6`, max `500` ký tự).
- **Chiến lược B (Paragraph-aware):** `by_paragraph(max_size=650)`, gom các đoạn `\n\n` đến tối đa `650` ký tự và chỉ tách đoạn vượt ngưỡng theo ranh giới câu `(?<=[.!?])\s+`, không filter (`use_filter=false`). Sinh **245 chunks** (min `68`, median `518`, mean `477.8`, max `648` ký tự).
- **Chiến lược C (Paragraph-aware + Pre-retrieval Metadata Filter):** Dùng chung hoàn toàn **245 chunks** và vectors với B (`embed` không chạy lại), bật pre-retrieval payload filter Qdrant trên các trường whitelist (`year`, `section_type`) khi câu hỏi có trường `filter != null`.

### Bộ câu hỏi benchmark ([benchmark/questions.json](../../benchmark/questions.json))
- **Chia dev/test theo tài liệu (tránh document leakage):**
  - **Dev (`15` câu, `q01`–`q15`):** nằm trên `2404.04302`, `2506.16988`, `2506.18027`; gồm 4 câu có filter metadata (`q05`, `q10`, `q14`, `q15`).
  - **Test giữ kín (`5` câu, `q16`–`q20`):** hoàn toàn tách riêng trên tài liệu `2603.05698`; gồm 2 câu có filter metadata (`q17` với `{year: 2026}`, `q19` với `{year: 2026, section_type: "evaluation"}`). Không chỉnh sửa câu hỏi hay tham số chiến lược sau khi chạy `test`.
- **Phân loại câu hỏi:** Có đủ 3 nhóm `truc_tiep`, `dien_dat_lai`, `dieu_kien_ngoai_le` (kèm biến thể `_filter`).
- **Quy ước Grounding Quality:** Thang điểm thủ công `0/1/2` trên tập `test` đo **độ đầy đủ bằng chứng truy hồi (retrieval evidence sufficiency)** trong top-3 chunk để trả lời chính xác câu hỏi nguồn, không phải đánh giá sinh câu trả lời của LLM.
- **Tiêu chí chọn chiến lược chốt trước khi xem test ([benchmark/questions.json](../../benchmark/questions.json)):** (1) Ưu tiên cao nhất `Evidence Hit@3` trên tập `dev`; (2) Xử lý đồng hạng bằng `Hit@1` trên `dev` kết hợp điểm thủ công `coherence`; (3) Đối chiếu mức độ tuân thủ filter metadata trên tập `dev` (`dev filter compliance`) và chi phí tài nguyên nạp/truy vấn.

---

## 3. Kết quả

### Bảng 1: Hiệu quả truy hồi và chất lượng ngữ nghĩa (A / B / C)

| Chiến lược | Dev Hit@1 | Dev Hit@3 | Test Hit@1 | Test Hit@3 | Chunk Coherence (10 vị trí, /20) | Metadata completeness (15 test hits) | Filter compliance theo câu (`test` / all) & theo hit | Test Grounding (/10) | Ghi chú ngắn |
|---|---|---|---|---|---|---|---|---|---|
| **A** (`fixed_size 500, overlap 50`) | `5/15` (`33.3%`) | `10/15` (`66.7%`) | `2/5` (`40.0%`) | `2/5` (`40.0%`) | `4/20` (`0.40/2`) *(0:6, 1:4, 2:0)* | `15/15` (`100%`) | `2/2` / `4/6` câu (`6/6` / `16/18` hits) | `6/10` (`1.20/2`) *(0:1, 1:2, 2:2)* | Cắt ngang từ ở 6/10 mẫu; làm mất vế đích `q19` (`led t`) và `q20` (`all b`) |
| **B** (`by_paragraph 650`, no filter) | `6/15` (`40.0%`) | `10/15` (`66.7%`) | `2/5` (`40.0%`) | `3/5` (`60.0%`) | `13/20` (`1.30/2`) *(0:0, 1:7, 2:3)* | `15/15` (`100%`) | `1/2` / `3/6` câu (`5/6` / `15/18` hits) | `8/10` (`1.60/2`) *(0:0, 1:2, 2:3)* | Sạch cắt cụt từ; sửa `q13`, `q20`, nhưng rò chunk `methodology` ở `q19` rank 3 |
| **C** (`by_paragraph 650` + filter) | `6/15` (`40.0%`) | `10/15` (`66.7%`) | `2/5` (`40.0%`) | `4/5` (`80.0%`) | `13/20` (`1.30/2`) *(chung chunk B)* | `15/15` (`100%`) | `2/2` / `6/6` câu (`6/6` / `18/18` hits) | `9/10` (`1.80/2`) *(0:0, 1:1, 2:4)* | Pre-filter loại `p03:9` khỏi `q19`, đẩy `para:p05:2` vào top-3 |

> Lưu ý: Số câu `split_across_chunks` trên tập `dev` là `2` ở cả A, B, C (các ca bỏ sót đa ý); trên tập `test` là `0` ở cả A, B, C (nguồn [results/traces_all.json](../../results/traces_all.json) và [results/manual_audit.json](../../results/manual_audit.json)).

### Bảng 2: Khối lượng xử lý, thời gian và chi phí

| Chiến lược | Số chunk | Tổng ký tự index | Thời gian parse (s) | Thời gian embed (s) / Số API call | Thời gian upsert (s) | Thời gian query 20 câu (s) | Chi phí / ghi chú tài nguyên |
|---|---:|---:|---:|---:|---:|---:|---|
| **A** | `272` | `129,099` | `0.666` *(dùng chung)* | `10.977` / `9` calls | `0.242` | `0.054` | Nhiều hơn B `+27` chunks (`+12,034` chars) do overlap 50 |
| **B** | `245` | `117,065` | `0.666` *(dùng chung)* | `9.261` / `8` calls | `0.189` | `0.037` | Giảm ~10% chunk và ~15.6% thời gian embed so với A |
| **C** | `245` | `117,065` | `0.666` *(dùng chung)* | `0` *(dùng chung vectors B)* | `0.201` *(index bản C)* | `0.042` | Chi phí embed biên bằng `0` nhờ dùng chung biểu diễn của B |

- **Tổng chi phí toàn pipeline (`run.py`):** **37 API calls**, **248,637 ký tự** đã embed (bao gồm probe dimension, document batches của A/B và 20 query embeddings tốn `8.011s`), `$0` trong hạn mức free tier Google AI Studio (nếu chạy tài khoản trả phí cần tính theo biểu giá `gemini-embedding-001` tại thời điểm sử dụng).

### Chi tiết đánh giá thủ công (Coherence 10 mẫu & Grounding 5 câu test)

- **Cách chọn mẫu thủ công ([starter/src/run.py](../../starter/src/run.py)):** Hàm `_select_coherence_positions` chọn tất định `10` vị trí neo trước khi chấm điểm theo hạn mức từng tài liệu (`2404.04302`: 3, `2506.16988`: 2, `2506.18027`: 3, `2603.05698`: 2), sắp theo `(document_id, page)`, lọc đoạn `>= 120` ký tự và ưu tiên đoạn thứ hai (`paras[1]`) ở `page == 1` (nếu có) để tránh tiêu đề/abstract đầu trang; phần Grounding kiểm tra toàn bộ `5` câu `test` (`q16`–`q20`) trên top-3 kết quả của cả 3 chiến lược.
- Chi tiết đầy đủ được lưu trong [results/manual_audit.json](../../results/manual_audit.json) đối chiếu trực tiếp từ [results/coherence_samples.json](../../results/coherence_samples.json) và [results/test_audit_samples.json](../../results/test_audit_samples.json):

- **Chunk Coherence (`0` = vỡ, `1` = tạm, `2` = mạch lạc):**
  - `coh_01` (`2404.04302:p01`): **A = 1** (lẫn địa chỉ email/tác giả, đứt câu cuối), **B/C = 1** (đầu chunk mở bằng mảnh ghép `"guage Model"` từ ngắt dòng `"Lan-\n\nguage"`, phần thân cuối câu hoàn chỉnh).
  - `coh_02` (`2404.04302:p02`): **A = 0** (header trang + đứt giữa từ `"context f"`), **B/C = 1** (còn header `2 N. Wiratunga et al.` đầu trang, cuối câu trọn vẹn).
  - `coh_03` (`2404.04302:p03`): **A = 0** (header trang + đứt giữa từ `"in address"`), **B/C = 1** (còn header trang, kết thúc trọn câu).
  - `coh_04` (`2506.16988:p01`): **A = 1** (dính affiliation và cụt ở `"architecture that"`), **B/C = 2** (trọn đoạn Abstract sạch nhiễu).
  - `coh_05` (`2506.16988:p02`): **A = 0** (venue header SIGIR + cắt từ `"agent specia"`), **B/C = 1** (vẫn dính venue line đầu trang nhưng không cắt từ/câu ở cuối).
  - `coh_06` (`2506.18027:p01`): **A = 1** (gộp author header, cắt giữa liệt kê `"vector diagrams,"`), **B/C = 2** (3 câu Abstract mạch lạc).
  - `coh_07` (`2506.18027:p02`): **A = 1** (cụt cuối cụm `"Integration of RAG with"`), **B/C = 1** (trọn đoạn Section 2 nhưng kéo theo tiêu đề mồ côi `"2.1 Retrieval-Augmented Generation (RAG)"` không có nội dung).
  - `coh_08` (`2506.18027:p03`): **A = 0** (trộn sang subsection mới và cắt từ `"Fahim [2022] wh"`), **B/C = 2** (trọn Section 3.2 + caption Figure 1 + đoạn mô tả PIER-QA).
  - `coh_09` (`2603.05698:p01`): **A = 0** (`fixed:p01:1` mở đầu bằng chữ cái rơi `"t"` và cắt từ `"Generati"`), **B/C = 1** (kết thúc trọn câu nhưng câu đầu thiếu chủ ngữ tách dòng trước đó).
  - `coh_10` (`2603.05698:p02`): **A = 0** (lẫn caption + cắt từ `"some level of c"`), **B/C = 1** (trộn caption Fig. 1 với vế vắt sang từ trang 1, nhưng kết thúc trọn câu).

- **Grounding Quality trên 5 câu `test` (`0` = sai hướng, `1` = thiếu 1 ý/bị cắt/mờ bảng, `2` = đủ dữ kiện):**
  - `q16` (4 khả năng của RGB): **A = 2**, **B = 2**, **C = 2** (đều chứa đủ 4 mục ở Rank 1 của A hoặc Rank 2 của B/C).
  - `q17` (ngưỡng `0.7`, Leiden `1.0`, `5` docs, `year=2026`): **A = 2**, **B = 2**, **C = 2** (cả 3 đạt Rank 1).
  - `q18` (ngoại lệ `GPT-4o-mini` chỉ hơn tại nhiễu `0.2`): **A = 0** (3 hits lạc sang trang 5 Table III), **B = 1**, **C = 1** (Rank 1–2 vào đúng phần Noise Robustness trang 4 nhưng thiếu đoạn kế tiếp chứa con số `0.2`).
  - `q19` (GRcomb vs RGB ở `ACC_doc` & `ED`, `year=2026, section_type=evaluation`): **A = 1** (`fixed:p05:2` ở Rank 3 có `71% vs 35.35%` nhưng cắt `"led t"` ngay trước `ED`); **B = 1** (Table II ở Rank 1 dính cột `"35.35100.0085.85"` khó đọc, còn đoạn văn xác nhận `para:p05:2` bị đẩy ra Rank 4 bởi chunk methodology `para:p03:9` ở Rank 3); **C = 2** (nhờ filter loại `para:p03:9`, kéo `para:p05:2` vào Rank 3).
  - `q20` (vì sao `GRdef` `12.66%` và giới hạn `< 50%`): **A = 1** (Rank 2 `fixed:p06:1` cắt ngang `(all b`, còn Rank 1 Conclusion `fixed:p06:7` chỉ tóm tắt `"remained below 50%"` tách mảnh -> automated exact clause match đánh trượt nhưng người đọc vẫn ghép được phần lớn ý); **B = 2**, **C = 2** (Rank 1 `para:p06:1` chứa trọn vẹn cả hai vế).

---

## 4. Phân tích và kết luận

### So sánh tác động của từng bước dữ liệu (A -> B -> C)
1. **Chuyển từ Fixed-size (A) sang Paragraph-aware (B):**
   - **Cải thiện rõ nhất ở chất lượng đoạn và không còn chunk vỡ:** Điểm coherence tăng từ **4/20** lên **13/20** (xóa hoàn toàn 6 trường hợp cắt giữa từ). Trên `dev`, Hit@1 tăng `5/15 -> 6/15`; trên `test`, Hit@3 tăng `2/5 -> 3/5` và Grounding tăng `6/10 -> 8/10`, đồng thời giảm ~10% chi phí lưu trữ/embed (`272 -> 245` chunks).
   - **Lý do B chưa đạt điểm coherence cao hơn (`13/20` chứ không `20/20`):** `7/10` vị trí B/C dừng ở điểm `1` đều bắt nguồn từ tầng **ingestion PDF** chứ không phải thuật toán chunking (running header/footer lặp lại đầu trang PDF, câu bị ngắt qua ranh giới trang, hoặc gạch nối `Lan-\n\nguage` tách đoạn). Điều này chứng minh chunking tốt không thể tự sửa hết lỗi từ bước trích xuất văn bản gốc.
2. **Bổ sung Metadata + Pre-filtering (B -> C):**
   - Không làm phát sinh chi phí embed mới (tái sử dụng 100% vector của B) nhưng loại bỏ triệt để hiện tượng rò chunk sai phần (`methodology` lọt vào câu hỏi kết quả `evaluation` hoặc `appendix`): filter compliance tính theo câu tăng từ **3/6** (`15/18` hits) ở B lên **6/6** (`18/18` hits, `100%`) ở C; riêng trên `test` tăng từ **1/2** (`5/6` hits) lên **2/2** (`6/6` hits). Trường `source` + `page` hỗ trợ truy vết trực tiếp về trang PDF gốc, còn `section_type` + `year` hữu dụng để khoanh vùng đúng ngữ cảnh học thuật (như tách `methodology` khỏi `evaluation`), nhưng có hạn chế là `section_type` gán ở cấp trang bằng regex theo cấu trúc corpus (`SECTION_RULES` cần cập nhật nếu thêm tài liệu khác format; ở trang chuyển tiếp như `2603.05698:p04`, các chunk thuộc phần cuối `III. EXPERIMENTAL SETUP` trước `IV. EXPERIMENTAL RESULTS` cũng kế thừa `section_type="evaluation"` của trang đó) và Chiến lược C giả định bộ lọc do ngữ cảnh câu hỏi truyền đúng vào trước khi truy xuất.
   - Trên tập `test`, tác động quyết định thể hiện trực diện ở **`q19`**: chunk `2603.05698:para:p03:9` (section `methodology`, score `0.7855`) chiếm mất vị trí Rank 3 của `para:p05:2` (section `evaluation`, score `0.7827`). Pre-filter trong C chặn `p03:9` trước khi ranking, nâng Test Hit@3 từ `3/5` lên `4/5` và Test Grounding lên `9/10`.

### Ví dụ lỗi cụ thể và nguyên nhân gốc
- **Lỗi 1 — Ranh giới ký tự cố định cắt mất bằng chứng (`q19` ở Chiến lược A, `2603.05698:p05`):**
  Chunk `2603.05698:fixed:p05:2` (Rank 3) mang đủ nửa đầu về `ACC_doc` (`71 %` vs `35.35 %`), nhưng chạm ngưỡng 500 ký tự nên cắt đúng chuỗi `"led t"` ngay trước số liệu `ED = 94.94% / 100%` và `7% / 77%`. Một lỗi tương tự xảy ra ở `q20` (`fixed:p06:1` cắt tại `"(all b"`). Đây là nhược điểm cố hữu của `fixed_size`.
- **Lỗi 2 — Hồi quy trên dữ liệu bảng và đoạn mô tả mâu thuẫn (`q15` ở Chiến lược B/C, `2506.18027:p12–p13`):**
  Ở câu `q15` (cấu hình mặc định của Marker trong Table 4), **A** tìm thấy ở Rank 3 (`fixed:p13:1`), nhưng **B/C** trượt cả top-3. Nguyên nhân kép: (i) `para:p12:3` (văn xuôi trang 12 viết *"...configured to run OCR on all pages"*) đạt cosine cao nhất (`0.8107`) hút mất Rank 1 dù mâu thuẫn với `OCR_ALL_PAGES FALSE` trong Table 4; (ii) `by_paragraph` cắt bảng tại ranh giới trang 13 sao cho `para:p13:0` (`0.7494`) ôm tiêu đề bảng và các tham số đầu, đẩy dòng `OCR_ENGINE`/`OCR_ALL_PAGES` sang `para:p13:1`. Bảng cấu hình cần parser/chunker bảo toàn nguyên khối bảng hoặc lặp lại header bảng.
- **Lỗi 3 — Dense embedding bỏ sót điều kiện ngoại lệ hẹp (`q18` ở cả A/B/C, `2603.05698:p04`):**
  Ở câu hỏi tìm ngưỡng nhiễu `0.2`, B và C vào đúng mục Noise Robustness ở Rank 1–2 (`para:p04:4`, `para:p04:3`), nhưng đoạn ngay sau đó chứa chuỗi quyết định `"only for the lowest noise ratio of 0.2"` bị chunk bảng `para:p05:5` (Table III, score `0.8218`) đẩy khỏi top-3. Dense semantic retrieval dễ bị hấp dẫn bởi đoạn tổng hợp dài hơn là câu chốt chứa hằng số hẹp; cần kết hợp hybrid search (BM25 + dense) hoặc reranker khi mở rộng.

### Khuyến nghị triển khai & giới hạn
- **Chọn Chiến lược C (`by_paragraph(650)` + `Qdrant` pre-filter trên `year`, `section_type`)** theo đúng tiêu chí đã chốt trước khi xem `test`: trên `dev`, cả 3 chiến lược hòa `Evidence Hit@3 = 10/15`; xử lý đồng hạng bằng `dev Hit@1` (`6/15` vs `5/15`) và `coherence` (`13/20` vs `4/20`) loại A; giữa B và C hòa hai chỉ số này nhưng C vượt trội về `dev filter compliance` (`4/4` câu, `12/12` hits so với `2/4` câu, `10/12` hits của B) mà không tốn thêm chi phí embed (`0` API calls bổ sung). Kết quả `test` sau đó xác nhận lựa chọn này (`Hit@3` tăng `3/5 -> 4/5`, `Grounding` tăng `8/10 -> 9/10`).
- **Hướng cải tiến ưu tiên tiếp theo:** (1) Lọc running header/footer lặp ở tầng ingestion PDF trước khi chunk; (2) Nhận diện khối bảng riêng (không gộp/cắt giữa chừng bảng thông số); (3) Kết hợp hybrid sparse+dense cho các truy vấn chứa tên tham số hoặc con số ngoại lệ.
- **Giới hạn đánh giá & nguy cơ thiên lệch:** Tập `test` chỉ gồm 5 câu hỏi trên 1 tài liệu giữ kín (`2603.05698`), nên mỗi câu chiếm tới 20% tỷ lệ `Hit@k`; ngoài ra Chiến lược C giả định phía người dùng hoặc router truy vấn truyền đúng filter ngữ cảnh (`year`, `section_type`). Các nhãn thủ công (`coherence` 10 vị trí và `grounding` 5 câu test) đều do chính tác giả (Nguyễn Bảo Trung) tự chấm nên tồn tại nguy cơ thiên lệch người chấm (self-scoring bias); mọi kết luận cần được đối chiếu cùng tập `dev` (15 câu) và trace thô trong `results/`.
- **Công cụ AI hỗ trợ & cách tự kiểm tra ([RULES.md](../../RULES.md)):** Sử dụng trợ lý lập trình AI (Antigravity) để hỗ trợ triển khai mã nguồn pipeline, viết unit test và tổng hợp bảng đối chiếu. Toàn bộ kết quả được kiểm tra lại bằng: (1) bộ unit test `pytest` trong [starter/tests/test_pipeline.py](../../starter/tests/test_pipeline.py); (2) kiểm chứng tự động chuỗi trích dẫn và `required_evidence_clauses` của 20 câu hỏi trên văn bản chuẩn hóa từ `load_pdf`; (3) chạy pipeline đầy đủ gọi API Gemini thật và lưu log/trace kiểm chứng; và (4) đọc chấm thủ công `10` mẫu trong [results/coherence_samples.json](../../results/coherence_samples.json) cùng `5` câu test trên top-3 của 3 chiến lược trong [results/test_audit_samples.json](../../results/test_audit_samples.json), ghi vào [results/manual_audit.json](../../results/manual_audit.json).
