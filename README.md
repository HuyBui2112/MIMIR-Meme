# MIMIR-Meme – Phát hiện Meme Misinformation trên Mạng Xã hội bằng Mô hình Đa phương thức

> Hệ thống đa phương thức phát hiện **meme misinformation** trên mạng xã hội, kết hợp thông tin từ **ảnh**, **chữ trong ảnh (OCR)** và **văn bản đi kèm** (caption/post), hướng tới vừa **phân loại** vừa **giải thích quyết định** dựa trên bằng chứng.

---

## 1. Mô tả đề tài

- Đề tài tập trung vào nhiệm vụ **phát hiện thông tin sai lệch (misinformation) trong meme** – một dạng nội dung rất phổ biến trên mạng xã hội (Facebook, Reddit, X, …).
- Khác với bài toán fact-checking văn bản thuần túy, meme thường:
  - Kết hợp **ảnh + text overlay** (chữ in trên ảnh) + caption/post.
  - Dùng **hình ảnh nổi tiếng** nhưng “tái ngữ cảnh” (image repurposing) để ám chỉ điều khác.
  - Sử dụng **hài hước, ẩn dụ, mỉa mai** nên khó xử lý chỉ với một modality.
- Hệ thống đề xuất sử dụng **mô hình đa phương thức (multimodal)** để trích xuất đặc trưng từ ảnh và văn bản, sau đó:
  - Phân loại meme vào các nhóm liên quan đến misinfo.
  - Cung cấp giải thích: đoạn text nào mâu thuẫn, entity nào nhạy cảm, liên kết fact-check nào liên quan.

---

## 2. Đặt vấn đề & mô phỏng bài toán

### 2.1. Bối cảnh

- Misinformation ngày càng lan rộng trên mạng xã hội dưới dạng meme, vì:
  - Dễ chia sẻ, dễ lan truyền, khó kiểm duyệt tự động.
  - Nội dung “hài hước” khiến người dùng ít phòng thủ, dễ tiếp nhận thông tin sai lệch.
- Nhiều chiến dịch truyền thông/định kiến chính trị sử dụng **meme** để “tái ngữ cảnh” phát ngôn, số liệu, hình ảnh, tạo cảm giác đúng dù sai.

### 2.2. Mô phỏng bài toán

- **Input**: Một meme (ảnh) và các metadata đi kèm (nếu có):
  - File ảnh.
  - Văn bản overlay trong ảnh (thu được qua OCR).
  - Caption/post đi kèm trên mạng xã hội.
- **Output**:
  - **Nhãn** phân loại:
    - `Non-misinfo` / `Misinfo` (nhị phân)  
    - Hoặc chi tiết hơn: `Clean`, `Out-of-Context (OOC)`, `False/Misleading Claim`, `Needs Context`, `Other/Uncertain`, (tùy chọn `Manipulated/AI?`).
  - **Confidence score** cho từng lớp.
  - (Ở giai đoạn nâng cao) **Giải thích**:
    - Highlight đoạn text/claim nghi vấn.
    - Liên kết đến bài fact-check hoặc nguồn tin cậy.
    - Thông tin về entity, bối cảnh.

### 2.3. Mô hình hoá

- Có thể xem bài toán theo hai tầng:
  1. **Tầng 1 – Misinfo Detection (nhị phân)**:
     - Phân biệt meme “sạch” vs meme có dấu hiệu thông tin sai lệch.
  2. **Tầng 2 – Fine-grained Misinfo Type**:
     - Trong nhóm `Misinfo`, phân loại tiếp:
       - `Out-of-Context (OOC)` – ảnh đúng, chú thích/overlay nói chuyện khác.
       - `False/Misleading Claim` – claim sai/lệch cần đối chiếu fact-check.
       - `Needs Context` – thiếu bối cảnh để kết luận đúng/sai.
       - `Other/Uncertain` – trường hợp không rõ, dữ liệu nhiễu.

---

## 3. Mục tiêu & phạm vi đề tài

### 3.1. Mục tiêu chính

- Xây dựng một **hệ thống phân loại meme đa phương thức** có khả năng:
  - Kết hợp thông tin từ **ảnh** và **văn bản** (OCR + caption/post).
  - Phân loại meme vào các nhóm misinfo (ít nhất là nhị phân, ưu tiên mở rộng fine-grained nếu dữ liệu cho phép).
  - Đánh giá bằng các độ đo chuẩn trong lĩnh vực (Macro-F1, ROC-AUC, F1 theo từng lớp).

### 3.2. Mục tiêu nâng cao (nếu đủ thời gian)

- Thử nghiệm:
  - **OOC detection**: sử dụng NLI giữa caption ảnh và overlay/post để phát hiện sai ngữ cảnh.
  - **RAG-Fact**: truy xuất các bài fact-check liên quan và đưa ra gợi ý verdict + evidence.
  - Giải thích kết quả thông qua highlight text và evidence.

### 3.3. Phạm vi

- Tập trung vào **meme tiếng Anh** từ các nguồn dữ liệu công khai (NewsCLIPpings, DisinfoMeme, …).
- Sử dụng **mô hình pretrained** (CLIP, ViT, RoBERTa/DeBERTa, BLIP, TrOCR, sentence-transformers) và fine-tune/adapter mức vừa phải (phù hợp tài nguyên).

---

## 4. Khảo sát & công trình liên quan (Related Work)

### 4.1. Phát hiện misinformation trên meme

- **NewsCLIPpings**:
  - Dataset tập trung vào **Out-of-Context**: ảnh thật nhưng caption sai bối cảnh.
  - Dùng CLIP và các mô hình vision-language để đo độ “phù hợp” giữa ảnh và caption.
  - Ý tưởng chính: **image–text mismatch** là tín hiệu mạnh cho OOC.

- **DisinfoMeme**:
  - Dataset meme từ Reddit, gắn nhãn nhiều loại misinfo, bao gồm false/misleading, propaganda, v.v.
  - Thể hiện **khó khăn thực tế**:
    - Văn bản trong ảnh cần OCR.
    - Meme có layout phức tạp, nhiều vùng text.
    - Cần kiến thức ngoài bối cảnh (world knowledge).

- **Hateful Memes / Fakeddit**:
  - Các tập dữ liệu multimodal cho hateful/rumor detection.
  - Đóng vai trò **bài học thiết kế kiến trúc**: đa số dùng *late fusion* hoặc *co-attention* giữa text và image.

### 4.2. Vision-language models & multimodal fusion

- **CLIP (OpenAI)**:
  - Học liên kết ảnh – text ở scale lớn, cho phép encode ảnh và text trong cùng không gian.
  - Được dùng rộng rãi làm backbone cho các bài toán retrieval, classification đa phương thức.

- **BLIP (Salesforce)**:
  - Mạnh về **image captioning**: sinh câu mô tả nội dung chính của ảnh.
  - Có thể tận dụng caption này như một nguồn text chuẩn hóa để so sánh với overlay/post.

- **TrOCR (Microsoft)**:
  - Mô hình OCR transformer-based cho tiếng Anh.
  - Phù hợp trích xuất chữ in trên meme với chất lượng tương đối tốt.

### 4.3. Fact-checking & RAG (Retrieval-Augmented Generation)

- Các hệ thống **fact-checking** hiện đại thường kết hợp:
  - Bộ tìm kiếm (BM25/bi-encoder) trên **ClaimReview/MediaReview** hoặc tập fact-check.
  - Mô hình NLI/QA để quyết định claim là True/False/Partly False.
- Ý tưởng RAG:
  - Thay vì để mô hình “tự nhớ” hết thế giới, sử dụng module **retriever** để truy vấn tài liệu liên quan.
  - Mô hình downstream (classifier hoặc LLM) dựa vào **context được retrieve** để ra quyết định.

### 4.4. Bài học rút ra

- **Multimodal fusion** luôn cho kết quả tốt hơn text-only hoặc image-only nếu triển khai đúng cách.
- **Xử lý trước (preprocess) tốt** – đặc biệt OCR và caption – giúp giảm tải cho mô hình chính.
- **OOC** thường được xử lý như bài toán **NLI giữa hai câu** (caption ảnh vs caption bài post).
- **RAG-Fact** dễ phức tạp hóa hệ thống, nên nhiều công trình triển khai theo kiểu mô-đun tách rời: classifier trước, RAG sau.

---

## 5. Ý tưởng & giải pháp đề xuất

### 5.1. Thiết kế nhãn (label schema)

- **Tầng 1 – Nhị phân**:
  - `Non-misinfo`: meme không chứa claim sai rõ ràng, hoặc chỉ hài hước, không gây hiểu sai.
  - `Misinfo`: bất kỳ meme nào có dấu hiệu sai lệch, mơ hồ cần bối cảnh, tái ngữ cảnh ảnh, v.v.

- **Tầng 2 – Phân loại chi tiết** (trên subset `Misinfo`):
  - `Clean` – dùng trong báo cáo để tham chiếu lớp nền (hoặc gộp vào `Non-misinfo`).
  - `Out-of-Context (OOC)` – ảnh và caption/overlay nói về hai chuyện khác nhau, hoặc cắt mất bối cảnh.
  - `False/Misleading Claim` – có claim cụ thể bị fact-check đánh giá sai/lệch.
  - `Needs Context` – thông tin đúng một phần nhưng thiếu bối cảnh / điều kiện / mẫu số.
  - `Other/Uncertain` – các trường hợp không rõ ràng hoặc label gốc không khớp hoàn toàn.

### 5.2. Kiến trúc tổng thể

- **Encoder ảnh**: CLIP-ViT hoặc ViT-base để trích xuất embedding ảnh.
- **Encoder văn bản**:
  - Text từ OCR (overlay).
  - Caption/post (nếu có).
  - (Tùy chọn) Caption sinh bởi BLIP.
- **Fusion module**:
  - Kết hợp embedding ảnh và text (concat, attention, hoặc gated fusion).
  - Head phân loại nhị phân + head phân loại chi tiết (multi-task).
- **OOC-head (nâng cao)**:
  - Sử dụng mô hình NLI pretrained để đánh giá quan hệ giữa caption ảnh và overlay/post (entailment/contradiction).
  - Thêm score OOC như một feature cho classifier.
- **RAG-Fact (nâng cao)**:
  - Index các claim từ Fact-Check Insights bằng BM25/sentence-transformers.
  - Cho meme text làm query → lấy top-k evidence → suy ra verdict và evidence links.

---

## 6. Hiện thực phương pháp & kế hoạch triển khai

Kế hoạch chia theo **giai đoạn**, ưu tiên hoàn thành MVP trước, sau đó mở rộng.

### Phase 0 – Thiết kế nhãn & khảo sát dữ liệu

- Xác định **schema nhãn cuối** (nhị phân + chi tiết).
- Đọc mô tả từng dataset (NewsCLIPpings, DisinfoMeme, Fact-Check Insights).
- Viết bảng **mapping nhãn** từ từng dataset về hệ nhãn chung.
- Chọn **kịch bản đánh giá chính**:
  - Train/test trên DisinfoMeme.
  - Cross-dataset: train DisinfoMeme, test NewsCLIPpings (nếu phù hợp).

### Phase 1 – Skeleton hệ thống & baseline đơn giản

- Thiết lập cấu trúc thư mục (như mục 10), môi trường Python, `requirements.txt`.
- Tạo các module chính (chỉ skeleton, chưa cần code chi tiết):
  - `src/data/dataset_loader.py`
  - `src/data/transforms.py`
  - `src/models/{image_model.py,text_model.py,multimodal_model.py}`
  - `src/training/{train.py,eval.py,utils.py}`
  - `src/config.py`, `run_training.py`
- Xây **baseline text-only**:
  - Encoder text (RoBERTa/DeBERTa/CLIP-text) + classifier.
  - Train cho bài toán nhị phân `Non-misinfo` vs `Misinfo`.
- Xây **baseline image-only**:
  - Encoder ảnh (CLIP/Vision Transformer) + classifier.
- Mục tiêu: có benchmark cơ bản để so với multimodal.

### Phase 2 – Multimodal fusion (MVP chính)

- Cài đặt mô hình **fusion**:
  - Lấy embedding ảnh và text, concat hoặc dùng một lớp attention đơn giản.
  - Head chính cho phân loại nhị phân; head phụ cho phân loại chi tiết (nếu dữ liệu đủ).
- Tiền xử lý:
  - Pipeline tạo `data/processed/`:
    - Chuẩn hóa path ảnh, text gốc.
    - (Giai đoạn này có thể chưa cần OCR/BLIP, dùng caption/post trước).
- Train và đánh giá:
  - So sánh Text-only vs Image-only vs Fusion.
  - Tính Macro-F1, F1-Misinfo, ROC-AUC.

### Phase 3 – Bổ sung OCR & Caption (nâng chất multimodal)

- Thêm module **OCR (TrOCR)**:
  - Chạy offline tạo text overlay + bbox → lưu vào `data/processed/`.
- Thêm **image captioning (BLIP)**:
  - Sinh caption mô tả nội dung ảnh, lưu cùng với dữ liệu.
- Cập nhật mô hình:
  - Thử các combination:
    - `overlay_text`
    - `post_text`
    - `overlay_text + post_text`
    - `BLIP_caption + overlay_text`, v.v.
- Thực hiện **ablation study**:
  - Có/không có OCR.
  - Có/không có BLIP caption.

### Phase 4 – OOC detection

- Thiết kế OOC-head đơn giản:
  - Dùng mô hình NLI pretrained (RoBERTa-large-MNLI, DeBERTa-MNLI, …).
  - Input: `BLIP_caption` vs `overlay_text/post_text`.
  - Lấy score `contradiction` / `entailment` làm feature.
- Tích hợp vào pipeline:
  - Cho subset có nhãn OOC → đánh giá F1-OOC.
  - So sánh:
    - Fusion + OOC-head vs Fusion thuần.

### Phase 5 – RAG-Fact & giải thích (nếu đủ thời gian)

- Chuẩn hóa Fact-Check Insights:
  - Trường: `claim_text`, `label`, `url`, `evidence_text`.
  - Lưu vào `data/external/`.
- Xây index:
  - BM25 (pyserini) hoặc embedding (sentence-transformers + faiss).
- Pipeline RAG:
  - Query từ text meme (overlay + post).
  - Retrieve top-k bài fact-check.
  - Rule-based suy luận verdict đơn giản (True/False/Mixed).
- Giải thích:
  - Trả về link bài fact-check, đoạn evidence chính.
  - Log một số case study để đưa vào báo cáo.

### Phase 6 – API & Demo

- Xây dựng API đơn giản bằng FastAPI:
  - Endpoint `/predict`:
    - Input: ảnh + (option) text.
    - Output: nhãn, score, một số trường giải thích (text, evidence).
- Dựng demo bằng Gradio:
  - Cho phép upload meme, hiển thị dự đoán + highlight text & evidence (nếu có).

---

## 7. Bộ dữ liệu dự kiến

- **NewsCLIPpings**:
  - Tập trung vào **Out-of-Context** – ảnh thật, caption sai bối cảnh.
  - Phù hợp cho việc học quan hệ ảnh–text và OOC detection.

- **DisinfoMeme**:
  - Meme misinfo lấy từ Reddit, có nhiều loại nhãn misinfo.
  - Dùng làm dataset chính cho huấn luyện và đánh giá phân loại misinfo.

- **Fact-Check Insights (Duke Reporters’ Lab)**:
  - Kho **ClaimReview/MediaReview** từ các bài fact-check.
  - Dùng cho module RAG-Fact (retrieval & evidence).

- _(Tùy chọn)_ **Hateful Memes / Fakeddit**:
  - Có thể tận dụng để pretrain/regularize phần fusion, nhưng không bắt buộc.

**Tổ chức dữ liệu trong repo**:

- `data/raw/`: dữ liệu ban đầu (download từ nguồn gốc).
- `data/processed/`: dữ liệu đã qua xử lý (OCR, caption, split train/val/test).
- `data/external/`: các resource ngoài (index RAG, lexicon, fact-check).

---

## 8. Pipeline hệ thống

Tổng quan pipeline:

1. **Collector & Ingest**
   - Download và tổ chức dataset vào `data/raw/`.
   - Viết loader thống nhất schema cơ bản:
     - `id, image_path, overlay_text (OCR), post_text, caption (BLIP), label, meta{source, time}`.

2. **Preprocess**
   - OCR với TrOCR → `overlay_text` + (tùy chọn) bbox để giữ layout.
   - Image caption với BLIP → `caption`.
   - Làm sạch text (lowercase, remove URL, normalize emoji nếu cần).
   - (Tùy chọn) NER + Entity Linking cho analysis.

3. **Modeling**
   - `image_model.py`: CLIP/ViT encoder → embedding ảnh.
   - `text_model.py`: Transformer encoder → embedding text (overlay/post/caption).
   - `multimodal_model.py`: fusion (late/hybrid) + classifier:
     - Head nhị phân `Non-misinfo` vs `Misinfo`.
     - Head chi tiết (OOC, False, NeedsContext, …).
   - (Nâng cao) OOC-head dựa trên NLI.
   - (Nâng cao) RAG-Fact module cho retrieval & verdict.

4. **Training & Evaluation**
   - `train.py`: vòng lặp huấn luyện, logging, lưu checkpoint vào `experiments/runs/`.
   - `eval.py`: tính các độ đo chính, cross-dataset, ablation.
   - `utils.py`: early stopping, scheduler, gradient clipping, load/save model.

5. **Serving & Explainability**
   - FastAPI `/predict` nhận input, trả về kết quả.
   - UI Gradio demo cho phép tương tác trực quan.
   - Xuất thêm thông tin giải thích: text, entity, evidence link.

---

## 9. Độ đo & cách đánh giá

- **Độ đo chính**:
  - `Macro-F1`: ưu tiên vì nhãn có thể mất cân bằng.
  - `F1-Misinfo` (hoặc F1 cho từng lớp chính: OOC, False/Misleading).
  - `ROC-AUC` cho bài toán nhị phân (Non-misinfo vs Misinfo).

- **Phân tích chi tiết**:
  - **Confusion matrix** giữa các lớp fine-grained (`Clean`, `OOC`, `False`, `NeedsContext`).
  - **Ablation study**:
    - ± OCR.
    - ± BLIP caption.
    - ± OOC-head.
    - ± RAG-Fact (nếu tích hợp).

- **Cross-dataset evaluation**:
  - Train trên DisinfoMeme, test trên NewsCLIPpings (và ngược lại nếu hợp lý).
  - Mục tiêu: kiểm tra khả năng **tổng quát hóa** (generalization).

- **Đánh giá định tính (qualitative)**:
  - Case study: một số meme tiêu biểu, xem mô hình dự đoán gì, evidence nào được retrieve, độ hợp lý của giải thích.

---

## 10. Cấu trúc thư mục & môi trường

### 10.1. Cấu trúc thư mục dự kiến

```text
MIMIR-Meme/
├─ .venv/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ external/
├─ notebooks/
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ dataset_loader.py
│  │  └─ transforms.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ text_model.py
│  │  ├─ image_model.py
│  │  └─ multimodal_model.py
│  ├─ training/
│  │  ├─ __init__.py
│  │  ├─ train.py
│  │  ├─ eval.py
│  │  └─ utils.py
│  └─ utils/
│     ├─ __init__.py
│     ├─ logging.py
│     └─ seed.py
├─ api/
│  └─ main.py
├─ experiments/
│  ├─ logs/
│  ├─ runs/
│  └─ configs/
├─ reports/
│  ├─ figures/
│  └─ paper/
├─ tests/
├─ .gitignore
├─ requirements.txt
├─ README.md
└─ run_training.py
```

### 10.2. Thiết lập môi trường

- Khuyến nghị Python **3.10.11** trở lên.

Kích hoạt môi trường ảo:

```bash
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

---

## 11. requirements.txt (gợi ý)

> Nếu dùng GPU NVIDIA, cần cài `torch/torchvision/torchaudio` theo hướng dẫn chính thức của PyTorch (tùy phiên bản CUDA).

```txt
# PyTorch core (chỉnh theo CUDA nếu có)
torch==2.3.1
torchvision==0.18.1
torchaudio==2.3.1

# Transformers & vision-language
transformers==4.44.2
accelerate==0.34.2
timm==1.0.3
einops==0.8.0
sentencepiece==0.2.0

# Image / OCR / Vision utils
Pillow==10.4.0
opencv-python==4.10.0.84

# Captioning (BLIP via HF; LAVIS optional)
lavis==1.0.2

# Retrieval / RAG
pyserini==0.43.0
faiss-cpu==1.8.0.post1
sentence-transformers==3.2.0
rank-bm25==0.2.2

# NLP utils
spacy==3.7.5
spacy-transformers==1.3.4
nltk==3.9.1

# Data & training utils
pandas==2.2.3
pyarrow==17.0.0
scikit-learn==1.5.2
mlflow==2.16.2
tensorboard==2.18.0
rich==13.9.2
loguru==0.7.2
pyyaml==6.0.2
python-dotenv==1.0.1

# Serving & Demo
fastapi==0.115.5
uvicorn[standard]==0.32.0
gradio==5.0.2

# Dev
ipykernel==6.29.5
```

---

## 12. .gitignore (gợi ý)

```gitignore
.venv/
__pycache__/
*.pyc
experiments/runs/
experiments/logs/
data/raw/
data/processed/
data/external/
reports/figures/
reports/paper/build/
*.ipynb_checkpoints
```

---

## 13. Kế hoạch thực hiện (tóm tắt)

1. **Tuần 1–2**: Khảo sát tài liệu, chốt nhãn & mapping dataset, dựng skeleton code, cài môi trường.
2. **Tuần 3–4**: Xây baseline text-only & image-only, chuẩn hóa pipeline train/eval, có kết quả đầu tiên.
3. **Tuần 5–6**: Hoàn thiện multimodal fusion, chạy ablation, tối ưu hyper-parameters.
4. **Tuần 7–8**: Thêm OCR + BLIP caption, OOC-head prototype; thực hiện cross-dataset evaluation.
5. **Tuần 9–10**: (Nếu kịp) Triển khai RAG-Fact, xây API + demo Gradio, hoàn thiện báo cáo & slide.


---

## Cập nhật requirements (thực tế đã dùng)

- Dùng CUDA 12.1, thêm dòng index PyTorch: `--extra-index-url https://download.pytorch.org/whl/cu121`.
- Pin bộ PyTorch: `torch==2.3.1+cu121`, `torchvision==0.18.1+cu121`, `torchaudio==2.3.1+cu121`.
- Pin `numpy==1.26.4`, `Pillow==10.4.0` (tương thích faiss/thinc/gradio).
- `pyserini` (0.43.0) tạm không cài vì đòi torch>=2.4; `spacy-transformers` tạm không cài vì xung đột transformers>=4.41; `lavis` comment (khó cài Windows, BLIP dùng qua `transformers`).

```txt
--extra-index-url https://download.pytorch.org/whl/cu121

# PyTorch core (CUDA 12.1)
torch==2.3.1+cu121
torchvision==0.18.1+cu121
torchaudio==2.3.1+cu121

# Transformers & vision-language
transformers==4.44.2
accelerate==0.34.2
timm==1.0.3
einops==0.8.0
sentencepiece==0.2.0

# Base numeric stack
numpy==1.26.4
Pillow==10.4.0
opencv-python==4.10.0.84

# Captioning (BLIP via HF; LAVIS optional)
# lavis==1.0.2

# Retrieval / RAG
# pyserini==0.43.0
faiss-cpu==1.8.0.post1
sentence-transformers==3.2.0
rank-bm25==0.2.2

# NLP utils
spacy==3.7.5
# spacy-transformers==1.3.4
nltk==3.9.1

# Data & training utils
pandas==2.2.3
pyarrow==17.0.0
scikit-learn==1.5.2
mlflow==2.16.2
tensorboard==2.18.0
rich==13.9.2
loguru==0.7.2
pyyaml==6.0.2
python-dotenv==1.0.1

# Serving & Demo
fastapi==0.115.5
uvicorn[standard]==0.32.0
gradio==5.0.2

# Dev
ipykernel==6.29.5
```

---

## Thực hiện setup môi trường (đã làm)

- Máy: Windows, Python 3.10.11, GPU NVIDIA RTX 4050 (driver CUDA 12.9), virtualenv `.venv`.
- Đã cài PyTorch CUDA 12.1 bằng wheel cu121 và pin version như trên.
- Đã pin `numpy==1.26.4`, `Pillow==10.4.0`.
- `pyserini` và `spacy-transformers` chưa cài để tránh xung đột; sẽ bật lại nếu nâng torch hoặc tách môi trường/WSL.
- Kiểm tra GPU & phiên bản đã chạy:
  ```bash
  nvidia-smi
  python -c "import torch, torchvision, torchaudio, numpy, PIL; \
  print(torch.__version__, torchvision.__version__, torchaudio.__version__); \
  print('cuda?', torch.cuda.is_available(), 'build', torch.version.cuda); \
  print('numpy', numpy.__version__, 'Pillow', PIL.__version__); \
  print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
  ```
  Kết quả: torch/vision/audio 2.3.1+cu121, CUDA build 12.1, `cuda? True`, numpy 1.26.4, Pillow 10.4.0, GPU nhận RTX 4050.
