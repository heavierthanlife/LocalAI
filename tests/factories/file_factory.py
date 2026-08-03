"""Test file builders for upload tests."""
import io


def make_txt(content="测试内容", name="test.txt"):
    return name, io.BytesIO(content.encode("utf-8"))


def make_pdf(content=b"%PDF-1.4 mock content", name="test.pdf"):
    return name, io.BytesIO(content)
