# Demo chatbot hỏi đáp tài liệu nội bộ

Phiên bản tối giản để trình diễn luồng **RAG** (Retrieval-Augmented Generation): tải tài liệu, chia nhỏ nội dung, lập chỉ mục, tìm đoạn liên quan và trả lời có nguồn.

## Định dạng hỗ trợ

- PDF: `.pdf`
- Word: `.docx`
- PowerPoint: `.pptx`
- Excel và CSV: `.xlsx`, `.csv`
- Văn bản và Markdown: `.txt`, `.md`
- Trang HTML lưu thành tệp: `.html`, `.htm`

## Chạy nhanh

### Cách 1 — dùng `uv` (gọn nhất)

1. Mở Terminal/PowerShell trong thư mục này.
2. Tạo tệp `.env` từ `.env.example`, sau đó thay `sk-your-key-here` bằng OpenAI API key.
3. Chạy:

```powershell
uv run streamlit run app.py
```

4. Trình duyệt sẽ mở `http://localhost:8501`.

### Cách 2 — dùng Python có sẵn

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Cách demo

1. Kéo/thả tối đa 5 tài liệu vào thanh bên trái.
2. Chọn **Đọc và lập chỉ mục**.
3. Nhập câu hỏi ở ô chat.
4. Mở **Xem các đoạn nguồn đã dùng** để kiểm tra tệp, trang, slide, sheet/dòng hoặc trích đoạn.
5. Hỏi một câu ngoài phạm vi để thấy chatbot trả lời rõ ràng rằng không tìm thấy.

> Lưu ý: bản demo trích xuất phần chữ có sẵn trong PDF. PDF scan chỉ có hình ảnh sẽ cần thêm OCR.

> Dữ liệu: nội dung tài liệu được gửi tới OpenAI API để tạo embedding; các đoạn liên quan được gửi tiếp để tạo câu trả lời. Không dùng tài liệu mật khi chưa có chính sách phù hợp.

## Luồng hoạt động

```text
Tài liệu → bộ đọc theo định dạng → chia đoạn có metadata nguồn
    → OpenAI embeddings → chỉ mục trong bộ nhớ phiên

Câu hỏi → embedding câu hỏi → cosine similarity → 4 đoạn gần nhất
    → kiểm tra ngưỡng liên quan → GPT trả lời chỉ từ ngữ cảnh
    → hiển thị câu trả lời + tệp + vị trí + trích đoạn
```

Mỗi lần khởi động lại app, chỉ mục sẽ được tạo lại. Cách này cố ý giữ demo đơn giản và không cần database.

## Khác gì so với repo tham chiếu?

Repo [`mayooear/ai-pdf-chatbot-langchain`](https://github.com/mayooear/ai-pdf-chatbot-langchain) tách frontend Next.js và backend LangGraph, lưu vector trong Supabase, hỗ trợ streaming và nhiều cấu hình mở rộng. Bản demo này giữ lại những phần cốt lõi:

- upload/kiểm tra nhiều định dạng tài liệu;
- ingest thành các đoạn có metadata nguồn;
- embedding + truy xuất theo độ tương đồng;
- prompt buộc trả lời theo ngữ cảnh;
- hiển thị nguồn đã dùng.

Và bỏ đi Supabase, LangGraph, LangSmith, Turborepo, API routes riêng, streaming và lưu lịch sử lâu dài. Toàn bộ logic nằm trong `rag.py`; giao diện nằm trong `app.py`.

## Cấu hình tùy chọn

```dotenv
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Hai giá trị này có thể thay trong `.env`. Chỉ mục và nội dung tài liệu không được ghi xuống ổ đĩa trong bản demo.

