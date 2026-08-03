# -*- coding: utf-8 -*-
"""Integration tests for the unified document parser pipeline.

Tests schema, classifier, parser, and both consumer adapters (wiki + template).
Covers 3 fixture types: law, bid, general.
"""
import io
import os
import sys
import pytest

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.document_schema import make_section, make_document, DOC_LAW, DOC_BID, DOC_GENERAL
from app.services.document_classifier import classify
from app.services.document_parser import (
    build_document, to_wiki_markdown, to_template_sections,
)


# ── Fixtures ──

LAW_PARAS = [
    {"text": "\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u653f\u5e9c\u91c7\u8d2d\u6cd5", "level": 1, "style": "heading"},
    {"text": "\u7b2c\u4e00\u7ae0 \u603b\u5219", "level": 1, "style": "heading"},
    {"text": "\u7b2c\u4e00\u6761 \u6839\u636e\u672c\u6cd5\u5236\u5b9a\u6761\u4f8b", "level": 2, "style": "heading"},
    {"text": "\u653f\u5e9c\u91c7\u8d2d\u5e94\u5f53\u9075\u5faa\u516c\u5f00\u900f\u660e\u539f\u5219", "level": 0, "style": "paragraph"},
    {"text": "\u7b2c\u4e8c\u6761 \u91c7\u8d2d\u5b9a\u4e49", "level": 2, "style": "heading"},
    {"text": "\u672c\u6cd5\u6240\u79f0\u91c7\u8d2d\u662f\u6307\u8d27\u7269\u3001\u5de5\u7a0b\u3001\u670d\u52a1", "level": 0, "style": "paragraph"},
    {"text": "\u7b2c\u4e8c\u7ae0 \u91c7\u8d2d\u5f53\u4e8b\u4eba", "level": 1, "style": "heading"},
    {"text": "\u7b2c\u4e09\u6761 \u91c7\u8d2d\u4eba\u5e94\u5f53\u7ef4\u62a4\u56fd\u5bb6\u5229\u76ca", "level": 2, "style": "heading"},
]


BID_PARAS = [
    {"text": "\u67d0\u9879\u76ee\u62db\u6807\u516c\u544a", "level": 1, "style": "heading"},
    {"text": "\u7b2c\u4e00\u7ae0 \u6295\u6807\u4eba\u987b\u77e5", "level": 1, "style": "heading"},
    {"text": "1. \u8d44\u8d28\u8981\u6c42", "level": 2, "style": "heading"},
    {"text": "\u6295\u6807\u4eba\u5e94\u5177\u5907\u65bd\u5de5\u603b\u627f\u5305\u4e00\u7ea7\u8d44\u8d28", "level": 0, "style": "paragraph"},
    {"text": "2. \u6280\u672f\u89c4\u683c", "level": 2, "style": "heading"},
    {"text": "\u9879\u76ee\u5e94\u7b26\u5408\u56fd\u5bb6\u6807\u51c6", "level": 0, "style": "paragraph"},
    {"text": "\u7b2c\u4e8c\u7ae0 \u8bc4\u6807\u529e\u6cd5", "level": 1, "style": "heading"},
    {"text": "\u91c7\u7528\u7efc\u5408\u8bc4\u5206\u6cd5", "level": 0, "style": "paragraph"},
]


GENERAL_PARAS = [
    {"text": "\u666e\u901a\u6587\u6863\u6807\u9898", "level": 0, "style": "paragraph"},
    {"text": "\u8fd9\u662f\u4e00\u6bb5\u666e\u901a\u7684\u6587\u672c\u5185\u5bb9\uff0c\u6ca1\u6709\u4efb\u4f55\u7ed3\u6784\u5316\u6807\u8bb0", "level": 0, "style": "paragraph"},
    {"text": "\u7b2c\u4e8c\u6bb5\u6587\u672c\u5185\u5bb9", "level": 0, "style": "paragraph"},
]


# ── Schema tests ──

def test_schema_make_section():
    s = make_section("s1", title="\u7b2c\u4e00\u7ae0", content="\u5185\u5bb9", level=1)
    assert s["id"] == "s1"
    assert s["title"] == "\u7b2c\u4e00\u7ae0"
    assert s["content"] == "\u5185\u5bb9"
    assert s["level"] == 1
    assert s["parent_id"] is None


def test_schema_make_document():
    s = make_section("s1", title="Test")
    doc = make_document(title="TestDoc", doc_type=DOC_LAW, sections=[s])
    assert doc["title"] == "TestDoc"
    assert doc["type"] == "law"
    assert len(doc["sections"]) == 1
    assert doc["sections"][0]["id"] == "s1"


# ── Classifier tests ──

def test_classifier_law_only():
    text = "\u7b2c\u4e00\u6761 \u6839\u636e\u672c\u6cd5\u5236\u5b9a"
    assert classify(text) == "law"


def test_classifier_bid_only():
    text = "\u62db\u6807\u516c\u544a \u6295\u6807\u4eba\u987b\u77e5"
    assert classify(text) == "bid"


def test_classifier_bid_priority():
    text = "\u7b2c\u4e00\u7ae0 \u603b\u5219 \u62db\u6807\u6587\u4ef6 \u6295\u6807\u6761\u4ef6"
    assert classify(text) == "bid"


def test_classifier_general():
    text = "\u666e\u901a\u6587\u672c\u5185\u5bb9"
    assert classify(text) == "general"


# ── Parser tests ──

def test_parser_law_builds_chapters_and_articles():
    doc = build_document(LAW_PARAS, doc_type="law")
    assert doc["type"] == "law"
    sections = doc["sections"]
    titles = [s["title"] for s in sections]
    assert "\u7b2c\u4e00\u7ae0 \u603b\u5219" in titles
    assert "\u7b2c\u4e00\u6761" in titles[:2][0] or titles[:2][1]
    # Verify parent_id links
    article_section = next((s for s in sections if s["level"] == 2 and s["parent_id"]), None)
    assert article_section is not None


def test_parser_bid_uses_heading_levels():
    doc = build_document(BID_PARAS, doc_type="bid")
    assert doc["type"] == "bid"
    sections = doc["sections"]
    assert len(sections) >= 3
    levels = [s["level"] for s in sections]
    assert 1 in levels
    assert 2 in levels


def test_parser_general_flat_sections():
    doc = build_document(GENERAL_PARAS, doc_type="general")
    assert doc["type"] == "general"
    sections = doc["sections"]
    assert len(sections) == len(GENERAL_PARAS)
    assert all(s["level"] == 1 for s in sections)


# ── Wiki adapter tests ──

def test_wiki_adapter_law():
    doc = build_document(LAW_PARAS, doc_type="law")
    md = to_wiki_markdown(doc)
    assert "## \u7b2c\u4e00\u6761" in md
    assert "# \u7b2c\u4e00\u7ae0" in md
    assert "\u516c\u5f00\u900f\u660e\u539f\u5219" in md


def test_wiki_adapter_bid():
    doc = build_document(BID_PARAS, doc_type="bid")
    md = to_wiki_markdown(doc)
    assert "\u62db\u6807\u516c\u544a" in md
    assert "\u8d44\u8d28\u8981\u6c42" in md


def test_wiki_adapter_general():
    doc = build_document(GENERAL_PARAS, doc_type="general")
    md = to_wiki_markdown(doc)
    assert "\u666e\u901a\u6587\u6863" in md
    assert "\u7ed3\u6784\u5316\u6807\u8bb0" in md


# ── Template adapter tests ──

def test_template_adapter_law():
    doc = build_document(LAW_PARAS, doc_type="law")
    secs = to_template_sections(doc)
    assert len(secs) >= 4
    assert all("id" in s and "title" in s and "content" in s for s in secs)
    assert any(s["level"] == 1 for s in secs)
    assert any(s["level"] == 2 for s in secs)


def test_template_adapter_bid():
    doc = build_document(BID_PARAS, doc_type="bid")
    secs = to_template_sections(doc)
    assert len(secs) >= 3
    assert all("order" in s for s in secs)


def test_template_adapter_general():
    doc = build_document(GENERAL_PARAS, doc_type="general")
    secs = to_template_sections(doc)
    assert len(secs) == len(GENERAL_PARAS)
    assert all(s["order"] == i + 1 for i, s in enumerate(secs))
