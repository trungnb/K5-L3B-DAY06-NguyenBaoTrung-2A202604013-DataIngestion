# Báo cáo cá nhân — Day 06: Data Ingestion Engineering

- Họ và tên: Nguyễn Bảo Trung
- MSSV: 2A202604013
- Lớp: L3B
- Repo bài nộp: [K5-L3B-DAY06-NguyenBaoTrung-2A202604013-DataIngestion](https://github.com/trungnb/K5-L3B-DAY06-NguyenBaoTrung-2A202604013-DataIngestion)
- Nguồn dữ liệu: 4 bài báo arXiv về RAG phát hành theo giấy phép **CC BY 4.0** (`arxiv-rag-ccby4-mini`, tải về `starter/data/`)

## Tóm tắt các bước đã thực hiện
1. **Kiểm tra quyền & chốt corpus:** Loại phương án dùng sách có bản quyền (`Grokking Machine Learning` — Manning *All rights reserved*); chọn 4 bài báo arXiv CC BY 4.0 và lập [data/manifest.json](../../data/manifest.json) (`42` trang vật lý -> `36` trang nội dung).
2. **Thiết kế & kiểm chứng benchmark trước khi chạy:** Sử dụng AI gợi ý nháp `20` câu hỏi trong [benchmark/questions.json](../../benchmark/questions.json), sau đó học viên tự kiểm tra và xác minh đầy đủ cả `20/20` câu đối chiếu trực tiếp trên 4 PDF nguồn; tách theo tài liệu (`15 dev` trên 3 bài đầu / `5 test` giữ kín trên `2603.05698`), có `6` câu filter (`4 dev`, `2 test`), xác minh từng `evidence_quote` trên văn bản sau `normalize_text`, và chốt tiêu chí chọn chiến lược trước khi xem `test`.
3. **Hiện thực pipeline & unit test:** Hoàn thiện ingestion, chunking (`A` fixed-size, `B/C` paragraph-aware), embedding Gemini thật, Qdrant local bền vững, idempotent upsert và `5` unit test offline trong [starter/tests/test_pipeline.py](../../starter/tests/test_pipeline.py).
4. **Chạy thực nghiệm A / B / C & test nạp lại:** Chạy `(cd starter && .venv/bin/python -m src.run)` qua Gemini API thật, lưu toàn bộ trace/log trong `results/`, xác nhận reload `2404.04302` không nhân đôi (`272 -> 272`), và dùng kết quả `dev` + coherence để chọn `C` trước khi đối chiếu trên `test`.
5. **Audit thủ công & phân tích:** Dùng AI gợi ý nháp điểm/nhận xét từ mẫu đầu ra, sau đó học viên tự rà soát lại toàn bộ `10` mẫu coherence cùng top-3 của `5` câu `test` trên cả 3 chiến lược và chốt nhãn cuối cùng vào [results/manual_audit.json](../../results/manual_audit.json).

---

## 1. Dữ liệu và pipeline

### Corpus, bản quyền & quy tắc tính trang
- Nguồn & quyền: 4 bài báo arXiv phát hành theo **CC BY 4.0** (kiểm tra qua PDF metadata `/License` ở `2506.16988`, `2506.18027`, `2603.05698` và trang arXiv ở `2404.04302`).
- Quy tắc tính trang ([data/manifest.json](../../data/manifest.json)): chỉ tính trang PDF chứa nội dung chính, bảng, thuật toán hoặc phụ lục; loại các trang chỉ có tài liệu tham khảo (`references_only`) và trang chủ yếu là references (`2506.18027` tr. 11).

| Document ID | Chủ đề (`topic`) | Năm (`year`) | Tổng trang PDF | Trang đưa vào index (`content_pages`) | Trang loại khỏi index |
|---|---|---:|---:|---|---|
| `2404.04302` (CBR-RAG) | `legal_qa` | 2024 | 17 | 1–15 | 16–17 (`references_only`) |
| `2506.16988` (RAGentA) | `attributed_qa` | 2025 | 5 | 1–4 | 5 (`references_only`) |
| `2506.18027` (PIER-QA) | `pdf_qa` | 2026 | 14 | 1–8, 12–14 | 9–10 (`references_only`), 11 (chủ yếu references + 5 dòng đầu Appendix A) |
| `2603.05698` (GraphRAG vs RGB) | `kg_robustness` | 2026 | 6 | 1–6 | Không |

- Tổng cộng: **42 trang vật lý**, còn **36 trang nội dung** (**117,393 ký tự chuẩn hóa**, thời gian parse `0.666s`).

### Trích xuất, chuẩn hóa, embedding & lưu trữ
- **Trích xuất & chuẩn hóa ([starter/src/ingestion.py](../../starter/src/ingestion.py)):**
  - Dùng `pypdf.PdfReader` mặc định (bỏ `extraction_mode="layout"` vì gộp ngang 2 cột IEEE/ACM).
  - `normalize_text` chuẩn hóa Unicode NFKC, nối từ bị gạch nối cuối dòng (`word-\nnext`), và tái tạo ranh giới đoạn `\n\n` dựa trên heading và độ dài dòng.
  - `infer_page_section` gán tất định `section_type` cấp trang (`intro_related`, `methodology`, `evaluation`, `appendix`) theo heading ở đầu/nửa trên trang (`offset <= 0.45`), không dùng thông tin nhãn câu hỏi.
- **Lỗi PDF thực tế còn tồn tại:** dính chữ trong bảng/ký hiệu (`forCUDA`, `SURY A_OCR_DPI`, `ACCREPRESENTS`, `35.35100.0085.85` ở Table II của `2603.05698` tr. 5), running header lặp đầu trang (`2 N. Wiratunga et al.`, dòng hội nghị SIGIR), và một số từ ngắt qua trang giữ gạch nối (`Lan-\n\nguage`).
- **Embedding & Vector Store ([starter/src/embeddings.py](../../starter/src/embeddings.py), [starter/src/store.py](../../starter/src/store.py)):**
  - Model: `gemini-embedding-001` (`3072` chiều, `google-genai==3.3.0`), `task_type="RETRIEVAL_DOCUMENT"` cho chunk và `RETRIEVAL_QUERY` cho câu hỏi (gọi API thật, retry backoff 4 lần, không dùng mock).
  - Store: `qdrant-client==1.20.0`, khởi tạo `QdrantClient(path="starter/results/qdrant_db")` với 3 collection `day6_A`, `day6_B`, `day6_C` (`Distance.COSINE`, exact brute-force scan trên local Python mode) thay cho Docker PostgreSQL để phù hợp máy Apple M1, 8 GB RAM.
  - Môi trường & lệnh chạy: `pypdf==6.21.4`, `python-dotenv==1.2.4`, `pytest==9.2.0`; lần chạy `src.run` xuất `results/` lúc `2026-11-14T10:05+07:00` dùng Python `3.12.16`, `.venv` hiện tại kiểm thử lại trên Python `3.13.5` (đều thuộc dải `3.11–3.14` của starter):
    - Kiểm thử đơn vị (offline): `(cd starter && .venv/bin/python -m pytest -q)`
    - Chạy toàn bộ pipeline: `(cd starter && .venv/bin/python -m src.run)`
- **Idempotency & xử lý tài liệu đổi:** Mỗi chunk dùng UUID5 tất định từ `(strategy, chunk_id)`; trước khi upsert một `document_id`, `QdrantVectorStore.upsert` cuộn và xóa các point cũ không còn trong danh sách chunk mới của `document_id` đó. Reload `2404.04302` trong [results/ingestion_log.json](../../results/ingestion_log.json) giữ nguyên `272 -> 272` points (`no_duplicates: true`); trường hợp tài liệu giảm số chunk được test tự động trong [starter/tests/test_pipeline.py](../../starter/tests/test_pipeline.py).

**Schema Metadata trên mỗi chunk & payload Qdrant (`8` trường):**

| Trường | Kiểu | Nguồn tạo | Lý do chọn |
| :--- | :--- | :--- | :--- |
| `document_id` | `str` | `manifest.json` (`source_id`) | Truy vết bài báo, xóa/cập nhật theo tài liệu và kiểm soát rò rỉ. |
| `source` | `str` | `manifest.json` (`local_path`) | Đường dẫn PDF gốc để đối chiếu trực tiếp trang nguồn. |
| `page` | `int` | Chỉ số trang PDF (1-based) | Xác minh `gold_page` và trích dẫn trang chính xác. |
| `year` | `int` | `manifest.json` (năm arXiv trang 1) | Trường lọc thời gian (`2024`, `2025`, `2026`) ở Chiến lược C. |
| `topic` | `str` | `manifest.json` (chủ đề bài báo) | Nhóm chủ đề (`legal_qa`, `attributed_qa`, `pdf_qa`, `kg_robustness`). |
| `section_type` | `str` | `infer_page_section` (tất định) | Trường lọc phần bài báo (`intro_related`, `methodology`, `evaluation`, `appendix`) ở C. |
| `normalized_chars` | `int` | `len(norm)` sau `normalize_text` | Theo dõi mật độ chữ mỗi trang để phát hiện trang rỗng/lỗi trích xuất. |
| `chunk_index` | `int` | Thứ tự chunk trên trang (`chunking.py`) | Giữ thứ tự gốc các chunk trong cùng trang khi audit. |

---

## 2. Cấu hình và benchmark

### Ba chiến lược so sánh ([artifacts/experiment_log.csv](../../artifacts/experiment_log.csv))
- **A (Baseline):** `fixed_size(size=500, overlap=50)`, không filter (`use_filter=false`) -> **272 chunks** (min `68`, median `500`, mean `474.6`, max `500` ký tự; `129,099` ký tự index).
- **B (Paragraph-aware):** `by_paragraph(max_size=650)`, gom đoạn `\n\n` đến tối đa `650` ký tự và chỉ cắt đoạn dài theo ranh giới câu `(?<=[.!?])\s+`, không filter (`use_filter=false`) -> **245 chunks** (min `68`, median `518`, mean `477.8`, max `648` ký tự; `117,065` ký tự index).
- **C (Paragraph-aware + Pre-retrieval Filter):** Dùng chung hoàn toàn **245 chunks** và vectors với B (`0` chi phí embed thêm); bật Qdrant payload filter trên whitelist (`year`, `section_type`) khi câu hỏi có `filter != null`.

### Bộ câu hỏi benchmark ([benchmark/questions.json](../../benchmark/questions.json))
- **20 câu hỏi** chia theo tài liệu để tránh rò rỉ ngữ cảnh cùng bài:
  - **Dev (`15` câu, `q01`–`q15`):** trên `2404.04302`, `2506.16988`, `2506.18027`; có `4` câu filter (`q05`, `q10`, `q14`, `q15`).
  - **Test giữ kín (`5` câu, `q16`–`q20`):** tách riêng trên `2603.05698`; có `2` câu filter (`q17`: `{year: 2026}`, `q19`: `{year: 2026, section_type: "evaluation"}`). Không chỉnh câu hỏi hay tham số sau khi chạy `test`.
- **Phân loại & kiểm chứng:** Đủ `truc_tiep`, `dien_dat_lai`, `dieu_kien_ngoai_le` (kèm biến thể `_filter`); mọi `evidence_quote` và `required_evidence_clauses` đều được đối chiếu trực tiếp trên text chuẩn hóa của trang nguồn.
- **Quy ước Grounding Quality:** Chấm thủ công `0/1/2` trên 5 câu `test` cho **độ đủ bằng chứng trong top-3 chunk** để trả lời câu hỏi nguồn, không phải đánh giá câu trả lời do LLM sinh ra.
- **Tiêu chí chọn chiến lược chốt trước khi xem test:** (1) `Evidence Hit@3` trên `dev`; (2) đồng hạng xét `Hit@1` trên `dev` và `coherence`; (3) xét tuân thủ filter metadata trên `dev` (`dev filter compliance`) và chi phí embed/lưu trữ.

---

## 3. Kết quả

### Bảng 1: Độ chính xác truy xuất & tài nguyên theo Split
| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian (parse + embed + upsert + query) | Chi phí |
|---|---|---:|---:|---:|---|---|
| **A** (`fixed_size 500/50`) | `dev` (15) | `5/15` (`33.3%`) | `10/15` (`66.7%`) | `272` | parse `0.666s` (chung), embed `10.977s` (`9` calls), upsert `0.242s`, query `0.054s` (20 câu) | `$0` (free tier; `129,099` doc chars) |
| **A** (`fixed_size 500/50`) | `test` (5) | `2/5` (`40.0%`) | `2/5` (`40.0%`) | `272` | *(dùng chung index A)* | *(dùng chung index A)* |
| **B** (`by_paragraph 650`) | `dev` (15) | `6/15` (`40.0%`) | `10/15` (`66.7%`) | `245` | parse `0.666s` (chung), embed `9.261s` (`8` calls), upsert `0.189s`, query `0.037s` (20 câu) | `$0` (free tier; `117,065` doc chars) |
| **B** (`by_paragraph 650`) | `test` (5) | `2/5` (`40.0%`) | `3/5` (`60.0%`) | `245` | *(dùng chung index B)* | *(dùng chung index B)* |
| **C** (`by_paragraph 650` + filter) | `dev` (15) | `6/15` (`40.0%`) | `10/15` (`66.7%`) | `245` | parse `0.666s`, embed `0s` (`0` calls, chung B), upsert `0.201s`, query `0.042s` (20 câu) | `$0` (`0` chi phí embed biên) |
| **C** (`by_paragraph 650` + filter) | `test` (5) | `2/5` (`40.0%`) | `4/5` (`80.0%`) | `245` | *(dùng chung index C)* | *(dùng chung index C)* |

- **Tổng tài nguyên API thật ([results/ingestion_log.json](../../results/ingestion_log.json)):** `37` API calls được bộ đếm ghi nhận (`9` batch A + `8` batch B + `20` query `8.011s`), tổng `248,637` ký tự (`129,099 + 117,065 + 2,473`); chưa tính `1` probe kiểm tra số chiều (`3072`) lúc khởi tạo `MyEmbedder`. Chi phí trực tiếp `$0` trong hạn mức free tier Google AI Studio.
- **Số câu bị tách mảnh (`split_across_chunks`):** `dev = 2` ở cả 3 chiến lược (`A`: `q07`, `q08`; `B/C`: `q02`, `q08`), `test = 0` ở cả 3 chiến lược ([results/traces_all.json](../../results/traces_all.json)).

### Bảng 2: Đánh giá chất lượng thủ công & tính hữu dụng của Metadata
| Strategy | Coherence trung bình /2 (`10` vị trí) | Metadata đủ và đúng (`15` test hits) | Filter tuân thủ theo câu (`dev` / `test` / `all`) | Grounding 0/1/2 (`5` câu test) và trung bình /2 |
|---|---|---|---|---|
| **A** | `0.40/2` (`4/20`; `0:6, 1:4, 2:0`) | `15/15` (`100%`) | `dev 2/4` (`10/12` hits), `test 2/2` (`6/6`), `all 4/6` (`16/18`) | `0:1, 1:2, 2:2` — `6/10` (`1.20/2`) |
| **B** | `1.30/2` (`13/20`; `0:0, 1:7, 2:3`) | `15/15` (`100%`) | `dev 2/4` (`10/12` hits), `test 1/2` (`5/6`), `all 3/6` (`15/18`) | `0:0, 1:2, 2:3` — `8/10` (`1.60/2`) |
| **C** | `1.30/2` (`13/20`; chung chunk B) | `15/15` (`100%`) | `dev 4/4` (`12/12` hits), `test 2/2` (`6/6`), `all 6/6` (`18/18`) | `0:0, 1:1, 2:4` — `9/10` (`1.80/2`) |

- **Lưu ý độ mịn `section_type` trên Bảng 2:** `15/15` (`100%`) đúng theo quy ước gán `section_type` **cấp trang** (`infer_page_section`). Nếu đối chiếu theo **cấp chunk** trên trang chuyển tiếp `2603.05698:p04`, có `3/15` hit ở A (`fixed:p04:0`, `fixed:p04:1`, `fixed:p04:2`) và `2/15` hit ở B/C (`para:p04:1`, `para:p04:2`) chứa nội dung mục `III.C Experimental Settings` trước tiêu đề `IV. EXPERIMENTAL RESULTS` nhưng mang nhãn trang `evaluation`.

### Đường dẫn trace, cách chọn mẫu & tóm tắt nhãn thủ công
- **Trace & mẫu:** [results/traces_all.json](../../results/traces_all.json), [results/coherence_samples.json](../../results/coherence_samples.json), [results/test_audit_samples.json](../../results/test_audit_samples.json), [results/manual_audit.json](../../results/manual_audit.json).
- **Chọn mẫu Coherence trước khi xem điểm ([starter/src/run.py](../../starter/src/run.py)):** Hàm `_select_coherence_positions` chọn tất định `10` vị trí theo hạn mức (`2404.04302`: 3, `2506.16988`: 2, `2506.18027`: 3, `2603.05698`: 2), sắp theo `(document_id, page)`, lấy đoạn `>= 120` ký tự và ưu tiên `paras[1]` ở `page == 1` (nếu có) để tránh block tiêu đề/tác giả đầu trang (lấy vào đoạn Abstract).
- **Chunk Coherence (`0` = vỡ, `1` = tạm, `2` = trọn ý):**
  - `coh_01` (`2404.04302:p01`): **A = 1** (lẫn email tác giả, đứt câu), **B/C = 1** (mở đầu mảnh `"guage Model"` từ `"Lan-\n\nguage"`).
  - `coh_02` (`2404.04302:p02`): **A = 0** (dính header + đứt từ `"context f"`), **B/C = 1** (còn header `2 N. Wiratunga et al.`, cuối câu trọn).
  - `coh_03` (`2404.04302:p03`): **A = 0** (dính header + đứt từ `"in address"`), **B/C = 1** (còn header trang, cuối câu trọn).
  - `coh_04` (`2506.16988:p01`): **A = 1** (dính affiliation, cụt `"architecture that"`), **B/C = 2** (trọn đoạn Abstract).
  - `coh_05` (`2506.16988:p02`): **A = 0** (dính venue SIGIR + đứt từ `"agent specia"`), **B/C = 1** (còn venue line đầu trang, không cắt từ).
  - `coh_06` (`2506.18027:p01`): **A = 1** (dính author header + cụt `"vector diagrams,"`), **B/C = 2** (3 câu Abstract mạch lạc).
  - `coh_07` (`2506.18027:p02`): **A = 1** (cụt `"Integration of RAG with"`), **B/C = 1** (dính tiêu đề mồ côi `2.1 ...` ở cuối).
  - `coh_08` (`2506.18027:p03`): **A = 0** (trộn subsection mới + đứt từ `"Fahim [2022] wh"`), **B/C = 2** (trọn Section 3.2 + Fig. 1 + đoạn PIER-QA).
  - `coh_09` (`2603.05698:p01`): **A = 0** (`fixed:p01:1` mở bằng chữ `"t"` rơi + đứt từ `"Generati"`), **B/C = 1** (cuối câu trọn nhưng câu đầu thiếu chủ ngữ tách dòng trước).
  - `coh_10` (`2603.05698:p02`): **A = 0** (trộn caption + đứt từ `"some level of c"`), **B/C = 1** (trộn caption Fig. 1 với vế từ trang 1, cuối câu trọn).
- **Grounding Quality trên `5` câu `test` (`0` = sai phần, `1` = thiếu/cắt ý, `2` = đủ bằng chứng):**
  - `q16` (4 khả năng của RGB): **A = 2**, **B = 2**, **C = 2** (Rank 1 ở A, Rank 2 ở B/C).
  - `q17` (ngưỡng `0.7`, Leiden `1.0`, `5` docs; `year=2026`): **A = 2**, **B = 2**, **C = 2** (cả 3 đạt Rank 1).
  - `q18` (ngoại lệ `GPT-4o-mini` ở nhiễu `0.2`): **A = 0** (cả 3 hits sang trang 5 Table III), **B = 1**, **C = 1** (Rank 1–2 vào đúng mục Noise Robustness trang 4 nhưng thiếu đoạn kế chứa `0.2`).
  - `q19` (GRcomb vs RGB ở `ACC_doc` & `ED`; `year=2026, section_type=evaluation`): **A = 1** (Rank 3 `fixed:p05:2` cắt `"led t"` ngay trước số `ED`); **B = 1** (Table II Rank 1 bị dính cột `"35.35100.0085.85"`, còn đoạn văn `para:p05:2` bị `para:p03:9` mục `methodology` đẩy ra Rank 4); **C = 2** (filter loại `para:p03:9`, kéo `para:p05:2` vào Rank 3).
  - `q20` (vì sao `GRdef` `12.66%` & giới hạn `< 50%`): **A = 1** (`fixed:p06:1` cắt ở `"(all b"` và `fixed:p06:7` viết paraphrase `"remained below 50%"` tách mảnh); **B = 2**, **C = 2** (Rank 1 `para:p06:1` chứa trọn cả hai vế).

---

## 4. Phân tích và kết luận

### So sánh A -> B và B -> C
- **Fixed-size (A) -> Paragraph-aware (B):**
  - Coherence tăng **4/20 -> 13/20** (hết hẳn 6 ca cắt giữa từ), giảm `~10%` số chunk (`272 -> 245`) và `~15.6%` thời gian embed (`10.977s -> 9.261s`).
  - `dev Hit@1` tăng `5/15 -> 6/15`, `test Hit@3` tăng `2/5 -> 3/5`, `test Grounding` tăng `6/10 -> 8/10`.
  - B chưa đạt coherence `20/20` vì `7/10` mẫu vẫn dính lỗi từ tầng ingestion PDF (header trang lặp, ngắt câu qua trang, hoặc `Lan-\n\nguage` tách đoạn).
- **Paragraph-aware (B) -> Metadata + Pre-filter (C):**
  - Chi phí embed biên `= 0` (dùng lại vectors của B) nhưng loại bỏ hoàn toàn các hit lệch nhãn `section_type`/`year` cấp trang theo bộ lọc: tuân thủ filter tăng từ `3/6` câu (`15/18` hits) lên `6/6` câu (`18/18` hits).
  - Tác động rõ nhất ở **`q19`** (test): `para:p03:9` (`methodology`, score `0.7855`) chiếm Rank 3 của `para:p05:2` (`evaluation`, score `0.7827`) ở B; pre-filter ở C loại `p03:9`, nâng Test Hit@3 từ `3/5` lên `4/5` và Test Grounding lên `9/10`.
  - **Hữu ích & hạn chế của metadata:** `source` + `page` giúp đối chiếu nhanh về đúng trang PDF; `section_type` + `year` hữu ích khoanh vùng ngữ cảnh bài báo, nhưng `section_type` gán cấp trang bằng regex theo corpus (`SECTION_RULES` cần chỉnh lại nếu thêm tài liệu khác format; trang chuyển tiếp `2603.05698:p04` khiến các chunk cuối `III. EXPERIMENTAL SETUP` ở đầu trang cũng mang nhãn `evaluation`), và C phụ thuộc filter truyền đúng trước khi truy xuất.

### Ba ví dụ lỗi cụ thể và nguyên nhân gốc
1. **Lỗi 1 — Cắt ngang chuỗi ở ranh giới cố định (`q19` ở A, `2603.05698:p05`):** `2603.05698:fixed:p05:2` (Rank 3) giữ phần `ACC_doc` (`71 %` vs `35.35 %`), nhưng chạm ngưỡng `500` ký tự nên cắt đúng `"led t"` trước cặp số `ED` (`94.94% / 100%` vs `7% / 77%`). Lỗi tương tự xảy ra ở `q20` (`fixed:p06:1` cắt tại `"(all b"`).
2. **Lỗi 2 — Hồi quy ở bảng cấu hình & văn xuôi mâu thuẫn (`q15` ở B/C, `2506.18027:p12–p13`):** A trúng ở Rank 3 (`fixed:p13:1`), còn B/C trượt vì (i) `para:p12:3` (văn xuôi trang 12) hút Rank 1 (`0.8107`) dù Table 4 ghi `OCR_ALL_PAGES FALSE`; (ii) `by_paragraph` đặt header Table 4 và các tham số đầu vào `para:p13:0` (`0.7494`), đẩy `OCR_ENGINE`/`OCR_ALL_PAGES` sang `para:p13:1`.
3. **Lỗi 3 — Dense embedding bỏ sót điều kiện ngoại lệ hẹp (`q18` ở A/B/C, `2603.05698:p04`):** B/C vào đúng mục Noise Robustness ở Rank 1–2 (`para:p04:4`, `para:p04:3`), nhưng câu chốt chứa ngưỡng `"only for the lowest noise ratio of 0.2"` bị `para:p05:5` (Table III, `0.8218`) đẩy khỏi top-3.

### Kết luận chọn chiến lược, giới hạn & cải tiến
- **Quyết định chọn Chiến lược C (`by_paragraph(650)` + `Qdrant` pre-filter trên `year`, `section_type`):** Theo đúng tiêu chí đã chốt trước khi mở `test`: trên `dev`, 3 chiến lược hòa `Hit@3 = 10/15`; xét `dev Hit@1` (`6/15` vs `5/15`) và `coherence` (`13/20` vs `4/20`) loại A; giữa B và C hòa 2 chỉ số này nhưng C thắng về `dev filter compliance` (`4/4` câu, `12/12` hits vs `2/4` câu, `10/12` hits) với `0` chi phí embed thêm. Kết quả `test` sau đó củng cố lựa chọn này (`Hit@3`: `3/5 -> 4/5`, `Grounding`: `8/10 -> 9/10`).
- **Giới hạn & thiên lệch:** Tập `test` chỉ có `5` câu trên `1` bài giữ kín (`2603.05698`, mỗi câu = `20%` tỷ lệ), nên kết quả chỉ mang tính minh họa trên corpus này; nhãn thủ công (`coherence`, `grounding`) được AI gợi ý nháp và chính tác giả (Nguyễn Bảo Trung) rà soát chốt cuối nên vẫn tồn tại rủi ro thiên lệch nội bộ (`self-scoring bias`).
- **Hướng cải tiến nếu làm tiếp:** (1) Lọc running header/footer lặp ở tầng ingestion trước khi chunk; (2) Tách và giữ nguyên khối bảng (hoặc lặp lại header bảng khi tách); (3) Thử hybrid sparse + dense (BM25 + embedding) cho các truy vấn chứa tên tham số và ngưỡng số hẹp.
- **Công cụ AI hỗ trợ & tự kiểm tra ([RULES.md](../../RULES.md)):** Dùng trợ lý lập trình AI (Antigravity) hỗ trợ viết code pipeline, unit test, gợi ý nháp `20` câu benchmark, gợi ý nháp nhận xét audit và rà soát báo cáo. Học viên đã tự kiểm tra, đối chiếu và xác minh đầy đủ cả `20/20` câu hỏi cùng đáp án trên 4 PDF gốc, tự kiểm mẫu đầu ra `normalize_text` và toàn bộ `10` mẫu coherence + `5` câu test để chốt nhãn `manual_audit.json`; đồng thời toàn bộ bài làm được kiểm chứng lại bằng: (1) `5` unit test offline `pytest`; (2) script kiểm tra tự động SHA-256, số trang, metadata và `evidence_quote`/`required_evidence_clauses` của cả 20 câu trên PDF thật; và (3) chạy pipeline gọi API Gemini thật lưu trace vào `results/`.
