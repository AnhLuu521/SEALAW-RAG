"""Lõi RAG: truy vấn -> ngữ cảnh -> LLM.

TODO:
1. retrieve(question): embed câu hỏi, tìm top-k trong Qdrant
2. build_prompt(question, hits): ghép ngữ cảnh + yêu cầu trích nguồn
3. answer(question): gọi LLM, trả về câu trả lời + nguồn
"""
