"""Nạp tài liệu vào vector DB.

Bước 3 (đã làm):
1. load_file(path): đọc pdf/docx/md/txt -> text
2. chunk(text): chia đoạn có overlap, theo câu (chunk_by_sentences) hoặc theo ký tự (chunk_text), xem CHUNK_MODE
   load_chunks(): đọc cả thư mục DOCS_DIR -> danh sách chunk kèm metadata

Bước 4 (TODO):
3. embed các chunk
4. upsert vào Qdrant kèm metadata (tên file, trang)

Chạy thử:
    python -m app.ingest
"""
import re
import sys
import unicodedata
from pathlib import Path

from docx import Document
from docx.table import Table
from pypdf import PdfReader

from app import config

SUPPORTED = {".pdf", ".docx", ".md", ".txt"}

# Bỏ qua file/thư mục có tên bắt đầu bằng:
#   "_" : tự đặt để tạm tắt 1 file, vd _README.txt
#   "." : file ẩn
#   "~" : file tạm Word tạo ra khi đang mở (~$abc.docx)
SKIP_PREFIXES = ("_", ".", "~")

# Thứ tự ưu tiên chỗ cắt: hết đoạn văn > hết dòng > hết câu > khoảng trắng
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " "]
_WHITESPACE = re.compile(r"\s")

_SENTENCE_END = re.compile(r"(?<=[.?!…])\s+|(?<=[.?!…][\"'”’)\]])\s+|\n\s*\n")


# =============================================================
# 1. Đọc file
# =============================================================
def load_file(path: Path) -> list[dict]:
    """Đọc 1 file, trả về danh sách trang: [{"text": "...", "page": 1}, ...]

    Chỉ PDF có số trang. md/txt/docx coi cả file là 1 trang (page=None).
    """
    ext = path.suffix.lower()

    if ext == ".pdf":
        reader = PdfReader(path)
        return [{"text": page.extract_text() or "", "page": i}
                for i, page in enumerate(reader.pages, start=1)]

    if ext == ".docx":
        return [{"text": _read_docx(path), "page": None}]

    if ext in (".md", ".txt"):
        # utf-8-sig: tự bỏ ký tự BOM mà Notepad hay chèn vào đầu file
        return [{"text": path.read_text(encoding="utf-8-sig"), "page": None}]

    raise ValueError(f"Không hỗ trợ định dạng {ext}")


def _read_docx(path: Path) -> str:
    """Đọc docx theo đúng thứ tự đoạn văn và bảng (tài liệu nội bộ hay để thông tin trong bảng)."""
    parts = []
    for block in Document(path).iter_inner_content():
        if isinstance(block, Table):
            for row in block.rows:
                cells: list[str] = []
                for cell in row.cells:
                    text = cell.text.strip()
                    # Ô gộp (merge) bị python-docx trả về lặp lại -> bỏ trùng
                    if not cells or cells[-1] != text:
                        cells.append(text)
                parts.append(" | ".join(cells))
        else:
            parts.append(block.text)
    return "\n".join(parts)


def clean_text(text: str) -> str:
    """Chuẩn hoá text trước khi chunk."""
    # Chữ có dấu tiếng Việt có 2 cách mã hoá (dựng sẵn / tổ hợp): nhìn giống nhau nhưng khác byte.
    # PDF/Word hay dùng kiểu tổ hợp -> đưa hết về dựng sẵn (NFC) để tìm kiếm nhất quán.
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text)  # nhiều dấu cách/tab -> 1 dấu cách
    text = re.sub(r" *\n *", "\n", text)        # bỏ dấu cách ở đầu/cuối dòng
    text = re.sub(r"\n{3,}", "\n\n", text)      # nhiều dòng trống -> 1 dòng trống
    return text.strip()


# =============================================================
# 2. Chia đoạn (chunk)
# =============================================================
def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Chia text thành các đoạn dài tối đa `size` ký tự, 2 đoạn liền nhau dùng chung khoảng `overlap` ký tự.

    Ưu tiên cắt ở chỗ hết đoạn văn / hết dòng / hết câu để không cắt ngang ý.
    """
    if overlap >= size:
        raise ValueError(f"CHUNK_OVERLAP ({overlap}) phải nhỏ hơn CHUNK_SIZE ({size})")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))

        # Chưa tới cuối text -> lùi điểm cắt về ranh giới gần nhất,
        # nhưng không lùi quá nửa đoạn (tránh đoạn quá ngắn)
        if end < len(text):
            for sep in SEPARATORS:
                cut = text.rfind(sep, start + size // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        # Đoạn sau lùi lại `overlap` ký tự. Nếu rơi vào giữa 1 chữ thì nhích tới đầu chữ kế tiếp.
        next_start = end - overlap
        if not text[next_start - 1].isspace():
            match = _WHITESPACE.search(text, next_start, end)
            if match:
                next_start = match.end()
        # Phòng trường hợp overlap quá lớn làm vòng lặp đứng yên
        start = next_start if next_start > start else end

    return chunks


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Vị trí (bắt đầu, kết thúc) của từng câu trong text."""
    spans = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        if match.start() > start:
            spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def split_sentences(text: str) -> list[str]:
    """Tách text thành từng câu: từ sau dấu chấm trước đến hết dấu chấm kế tiếp."""
    return [text[a:b] for a, b in _sentence_spans(text)]


def chunk_by_sentences(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Chia theo câu: gom các câu liền nhau vào 1 chunk cho tới khi sắp vượt `size` ký tự.

    - Không bao giờ cắt giữa câu (trừ khi 1 câu dài hơn cả `size`, lúc đó cắt theo ký tự).
    - Overlap tính bằng câu nguyên: chunk sau lặp lại vài câu cuối của chunk trước,
      tổng độ dài phần lặp không quá `overlap` ký tự.
    """
    if overlap >= size:
        raise ValueError(f"CHUNK_OVERLAP ({overlap}) phải nhỏ hơn CHUNK_SIZE ({size})")

    spans = _sentence_spans(text)
    chunks = []
    i = 0  # câu đầu tiên của chunk đang gom
    while i < len(spans):
        start = spans[i][0]

        # Câu dài hơn cả 1 chunk -> đành cắt theo ký tự
        if spans[i][1] - start > size:
            chunks.extend(chunk_text(text[start:spans[i][1]], size, overlap))
            i += 1
            continue

        # Gom thêm câu cho tới khi sắp vượt size. Sau vòng lặp: chunk = câu i .. câu j-1
        j = i + 1
        while j < len(spans) and spans[j][1] - start <= size:
            j += 1
        end = spans[j - 1][1]
        chunks.append(text[start:end])
        if j >= len(spans):
            break

        # Chunk sau bắt đầu lùi lại vài câu cuối (overlap), nhưng phải tiến ít nhất 1 câu so với i
        k = j
        while k - 1 > i and end - spans[k - 1][0] <= overlap:
            k -= 1
        i = k

    return chunks


def chunk(text: str) -> list[str]:
    """Chia text theo cách đã chọn trong config.CHUNK_MODE."""
    if config.CHUNK_MODE == "sentence":
        return chunk_by_sentences(text)
    if config.CHUNK_MODE == "char":
        return chunk_text(text)
    raise ValueError(f"CHUNK_MODE phải là 'sentence' hoặc 'char', đang là '{config.CHUNK_MODE}'")


def load_chunks(docs_dir: Path = config.DOCS_DIR) -> list[dict]:
    """Đọc mọi tài liệu trong thư mục (kể cả thư mục con) -> danh sách chunk.

    Mỗi chunk: {"text": "...", "source": "thu-muc/ten-file.pdf", "page": 3, "chunk_index": 0}
    PDF được chunk theo từng trang để giữ đúng số trang khi trích nguồn.
    """
    chunks: list[dict] = []
    for path in sorted(docs_dir.rglob("*")):
        rel = path.relative_to(docs_dir)
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        if any(part.startswith(SKIP_PREFIXES) for part in rel.parts):
            continue
        source = rel.as_posix()  # dùng "/" cho mọi hệ điều hành

        try:
            pages = load_file(path)
        except Exception as e:  # 1 file lỗi không làm hỏng cả lần nạp
            print(f"  !! Bỏ qua {source}: {e}")
            continue

        file_chunks = []
        for page in pages:
            for text in chunk(clean_text(page["text"])):
                file_chunks.append({
                    "text": text,
                    "source": source,
                    "page": page["page"],
                    "chunk_index": len(file_chunks),
                })

        if file_chunks:
            print(f"  {source}: {len(pages)} trang -> {len(file_chunks)} chunk")
        else:
            print(f"  !! {source}: không có chữ nào (file rỗng, hoặc PDF dạng ảnh scan)")
        chunks.extend(file_chunks)

    return chunks


# =============================================================
# Chạy thử: python -m app.ingest
# =============================================================
if __name__ == "__main__":
    # Windows mặc định in bằng cp1252 khi output bị chuyển hướng -> lỗi với tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8")

    print(f"Đọc tài liệu trong {config.DOCS_DIR}")
    print(f"CHUNK_MODE={config.CHUNK_MODE}, CHUNK_SIZE={config.CHUNK_SIZE}, CHUNK_OVERLAP={config.CHUNK_OVERLAP}\n")
    chunks = load_chunks()
    if not chunks:
        print("Chưa có chunk nào. Bỏ tài liệu pdf/docx/md/txt vào thư mục trên rồi chạy lại.")
        sys.exit()

    lengths = [len(c["text"]) for c in chunks]
    print(f"\nTổng: {len(chunks)} chunk | độ dài (ký tự): "
          f"ngắn nhất {min(lengths)}, trung bình {sum(lengths) // len(lengths)}, dài nhất {max(lengths)}")

    print("\n2 chunk đầu tiên (để xem chỗ cắt và phần overlap):")
    for c in chunks[:2]:
        where = c["source"] + (f", trang {c['page']}" if c["page"] else "")
        print(f"\n--- {where} | chunk #{c['chunk_index']} | {len(c['text'])} ký tự ---")
        print(c["text"])
