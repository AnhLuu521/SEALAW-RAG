"""Khởi tạo client dùng chung: LLM/embedding (API tương thích OpenAI) và Qdrant.

Các file khác chỉ cần:
    from app.clients import chat, embed, get_qdrant

Chạy thử:
    python -m app.clients
"""
from functools import lru_cache

from openai import OpenAI
from qdrant_client import QdrantClient

from app import config

# Số đoạn text gửi trong 1 lần gọi embedding.
# Gemini nhận tối đa 100 đoạn / lần, để 50 cho an toàn.
EMBED_BATCH_SIZE = 100


# =============================================================
# LLM (Gemini)
# =============================================================
@lru_cache(maxsize=1)
def get_llm() -> OpenAI:
    """Tạo client gọi Gemini. Nhờ @lru_cache, client chỉ tạo 1 lần rồi dùng lại."""
    if not config.LLM_API_KEY:
        raise RuntimeError("Chưa có LLM_API_KEY. Kiểm tra file .env")

    return OpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        # Thư viện openai tự thử lại khi gặp lỗi 429 (quá giới hạn), 5xx (quá tải) hoặc mất mạng.
        # Mỗi lần thử lại sẽ chờ lâu hơn lần trước.
        max_retries=config.MAX_RETRIES,
        timeout=60,  # giây
    )


def chat(messages: list[dict]) -> str:
    """Gửi danh sách tin nhắn cho LLM, trả về câu trả lời (chuỗi).

    messages có dạng:
        [{"role": "system", "content": "..."},
         {"role": "user",   "content": "..."}]
    """
    resp = get_llm().chat.completions.create(
        model=config.CHAT_MODEL,
        messages=messages,
    )
    return resp.choices[0].message.content or ""


def embed(texts: list[str]) -> list[list[float]]:
    """Đổi danh sách text thành danh sách vector (mỗi vector có EMBED_DIM số).

    Thứ tự vector trả về khớp với thứ tự text đưa vào.
    """
    if not texts:
        return []

    # Gemini báo lỗi cả batch nếu có 1 đoạn rỗng -> chặn sớm để biết đoạn nào bị lỗi
    for i, text in enumerate(texts):
        if not text or not text.strip():
            raise ValueError(f"Đoạn thứ {i} bị rỗng, không embed được")

    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        resp = get_llm().embeddings.create(model=config.EMBED_MODEL, input=batch)
        # Gemini trả vector theo đúng thứ tự input (trường index có thể là None nên không sort theo nó)
        if len(resp.data) != len(batch):
            raise RuntimeError(f"Gửi {len(batch)} đoạn nhưng nhận về {len(resp.data)} vector")

        # Số chiều sai thì Qdrant sẽ từ chối lúc lưu, báo ở đây cho dễ hiểu hơn
        dim = len(resp.data[0].embedding)
        if dim != config.EMBED_DIM:
            raise RuntimeError(f"Vector có {dim} chiều nhưng EMBED_DIM = {config.EMBED_DIM}. Sửa EMBED_DIM trong .env")

        vectors.extend(item.embedding for item in resp.data)
    return vectors


# =============================================================
# Qdrant
# =============================================================
@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    """Tạo client Qdrant: server nếu có QDRANT_URL, ngược lại chạy local trong thư mục."""
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL)
    return QdrantClient(path=str(config.QDRANT_PATH))


# =============================================================
# Chạy thử: python -m app.clients
# =============================================================
if __name__ == "__main__":
    import sys

    # Windows mặc định in bằng cp1252 khi output bị chuyển hướng -> lỗi với tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8")

    print("1) Chat với", config.CHAT_MODEL)
    answer = chat([{"role": "user", "content": "Chào bạn, trả lời 1 câu ngắn bằng tiếng Việt."}])
    print("   ->", answer.strip())

    print("\n2) Embedding với", config.EMBED_MODEL)
    vecs = embed(["Quy trình xin nghỉ phép", "Cách đăng ký nghỉ phép năm"])
    print(f"   -> {len(vecs)} vector, mỗi vector {len(vecs[0])} chiều")

    print("\n3) Qdrant:", config.QDRANT_URL or f"local ({config.QDRANT_PATH})")
    print("   -> collections:", [c.name for c in get_qdrant().get_collections().collections])

    print("\nOK, clients chạy tốt.")
