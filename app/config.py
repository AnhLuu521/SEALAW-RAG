"""Cấu hình: đọc biến môi trường (model, URL Qdrant, kích thước chunk, top_k...)."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# =============================================================
# LLM (Gemini qua API tương thích OpenAI)
# =============================================================
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # KHÔNG để key thật ở đây, chỉ để trong .env
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.8-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2")

# Số chiều vector của model embedding (gemini-embedding-2 = 3072).
# Đổi EMBED_MODEL thì phải đổi số này và nạp lại toàn bộ tài liệu.
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))

# =============================================================
# Qdrant (vector DB)
# =============================================================
# Có QDRANT_URL  -> kết nối Qdrant server (mặc định: Docker ở cổng 6333)
# Để trống       -> chạy local, lưu dữ liệu vào thư mục QDRANT_PATH (không cần Docker)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", BASE_DIR / "qdrant_storage"))
COLLECTION = os.getenv("COLLECTION", "docs")

# =============================================================
# Ingest (nạp tài liệu)
# =============================================================
DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "data" / "docs"))

# Độ dài mỗi đoạn (tính bằng ký tự).
#   Nhỏ (300-500)  : tìm chính xác hơn, nhưng mỗi đoạn ít ngữ cảnh
#   Lớn (1000-1500): nhiều ngữ cảnh hơn, nhưng dễ lẫn nội dung không liên quan
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))

# Số ký tự 2 đoạn liền nhau dùng chung, để câu không bị cắt mất ý.
# Thường để khoảng 10-20% của CHUNK_SIZE. Phải nhỏ hơn CHUNK_SIZE.
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Cách chia đoạn:
#   sentence : gom nguyên câu (hết câu = sau dấu . ? ! …), không cắt giữa câu
#   char     : cắt theo số ký tự, ưu tiên chỗ hết đoạn/dòng/câu
CHUNK_MODE = os.getenv("CHUNK_MODE", "sentence")

# =============================================================
# Retrieval (truy vấn)
# =============================================================
# Số đoạn lấy ra để gửi cho LLM mỗi lần hỏi.
#   Ít (3)       : rẻ, nhanh, nhưng dễ thiếu thông tin
#   Nhiều (8-10) : đủ thông tin hơn, nhưng tốn token và dễ bị nhiễu
TOP_K = int(os.getenv("TOP_K", "8"))

# Số lần thử lại khi Gemini báo lỗi 429 (quá giới hạn) hoặc 503 (quá tải)
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
