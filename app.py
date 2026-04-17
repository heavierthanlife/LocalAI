import os
import io
import re
import sys
import json
import filelock
import uuid
import docx
import time
import html
import hashlib
import atexit
import shutil
import requests
import sqlite3
import openpyxl
import logging
import difflib
import tempfile
import pickle
import threading
import imagehash
import psycopg2
import subprocess
import zipfile
import asyncio
import secrets
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from io import BytesIO
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session, send_file, url_for
from flask_session import Session
from pptx import Presentation
from PIL import Image
from markitdown import MarkItDown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from apscheduler.schedulers.background import BackgroundScheduler
from threading import Lock, RLock
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash
from contextlib import contextmanager
from functools import lru_cache, wraps
from typing import Dict, List, Optional, Tuple, Any, Generator
from pathlib import Path
from collections import defaultdict

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from langchain.agents import create_agent
from langchain_qwq import ChatQwen
from langchain.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver as SqS

# Additional imports for VL
from langchain_core.messages import HumanMessage, AIMessage
import base64
import aiosqlite

load_dotenv()

# Global agent and async infrastructure
_agent = None
_agent_lock = threading.RLock()
_async_loop = None
_async_checkpointer = None
_current_max_tokens = 1600
_async_conn = None


def _init_async_checkpointer():
    global _async_loop, _async_checkpointer, _async_conn
    _async_loop = asyncio.new_event_loop()

    async def create():
        global _async_conn
        _async_conn = await aiosqlite.connect("checkpoints.db")
        return AsyncSqliteSaver(_async_conn)

    _async_checkpointer = _async_loop.run_until_complete(create())

    def run_loop():
        asyncio.set_event_loop(_async_loop)
        _async_loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    logger.info("AsyncSqliteSaver initialized.")


def _cleanup_async_checkpointer():
    global _async_loop, _async_checkpointer, _async_conn
    if _async_loop is not None and _async_loop.is_running():
        async def shutdown():
            if _async_conn:
                await _async_conn.close()
            _async_loop.stop()

        asyncio.run_coroutine_threadsafe(shutdown(), _async_loop)
        time.sleep(0.5)
    logger.info("Async checkpointer cleaned up.")


atexit.register(_cleanup_async_checkpointer)

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'default': {'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'},
        'detailed': {'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'},
    },
    'handlers': {
        'console': {'class': 'logging.StreamHandler', 'level': 'INFO', 'formatter': 'default',
                    'stream': 'ext://sys.stdout'},
        'file': {'class': 'logging.handlers.RotatingFileHandler', 'level': 'DEBUG', 'formatter': 'detailed',
                 'filename': 'app.log', 'maxBytes': 10485760, 'backupCount': 5},
    },
    'root': {'level': 'DEBUG', 'handlers': ['console', 'file']},
}
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from logging.config import dictConfig

dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Temporary file management
TEMP_ROOT = os.path.join(tempfile.gettempdir(), 'flask_ai_temp')
os.makedirs(TEMP_ROOT, exist_ok=True)
USER_FILES_ORIGINAL_ROOT = os.path.join(os.getcwd(), 'user_files_original')
os.makedirs(USER_FILES_ORIGINAL_ROOT, exist_ok=True)

def get_anon_temp_dir(anon_id):
    path = os.path.join(TEMP_ROOT, anon_id)
    os.makedirs(path, exist_ok=True)
    return path

def cleanup_anon_temp(anon_id):
    path = os.path.join(TEMP_ROOT, anon_id)
    if os.path.exists(path):
        shutil.rmtree(path)
        logger.info(f"Cleaned up temp files for anon user {anon_id}")

def cleanup_all_temp_on_exit():
    if os.path.exists(TEMP_ROOT):
        shutil.rmtree(TEMP_ROOT)
        logger.info("Cleaned up all temp files on exit.")

atexit.register(cleanup_all_temp_on_exit)

# Database connection pool
def get_db_connection_args():
    if os.getenv("PG_USER") and os.getenv("PG_PASSWORD"):
        return {
            'dbname': os.getenv('PG_DB', 'postgres'),
            'user': os.getenv('PG_USER'),
            'password': os.getenv('PG_PASSWORD'),
            'host': os.getenv('PG_HOST', 'localhost'),
            'port': int(os.getenv('PG_PORT', 5432))
        }
    else:
        uri = os.getenv("POSTGRES_URI")
        if not uri:
            raise ValueError("No database connection configuration found.")
        from urllib.parse import urlparse, unquote
        result = urlparse(uri)
        dbname = result.path[1:] if result.path else ''
        user = result.username
        password = result.password
        if password:
            password = unquote(password)
        host = result.hostname
        port = result.port or 5432
        return {'dbname': dbname, 'user': user, 'password': password, 'host': host, 'port': port}

conn_args = get_db_connection_args()
db_pool = pool.SimpleConnectionPool(1, 20, **conn_args)

@contextmanager
def get_db_connection():
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)

@contextmanager
def db_transaction(conn):
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise

def db_execute_readonly(cur):
    cur.execute("SET TRANSACTION READ ONLY")

# Timezone helpers
BEIJING_TZ = timezone(timedelta(hours=8))

def beijing_now() -> str:
    return datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

# PostgreSQL table initialization (add task_deposit tables)
def init_postgres_tables():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Users table
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS users
                        (
                            user_id    TEXT PRIMARY KEY,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                        """)
            cur.execute("""
                DO $$ 
                BEGIN
                    BEGIN
                        ALTER TABLE users ADD COLUMN username TEXT UNIQUE;
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                    BEGIN
                        ALTER TABLE users ADD COLUMN pin_hash TEXT;
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                    BEGIN
                        ALTER TABLE users ADD COLUMN pin_length INTEGER;
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                    BEGIN
                        ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE;
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                    BEGIN
                        ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user';
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                END $$;
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            # Chat tables
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS chat_sessions
                        (
                            id         SERIAL PRIMARY KEY,
                            user_id    TEXT REFERENCES users (user_id),
                            thread_id  TEXT UNIQUE NOT NULL,
                            title      TEXT,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW()
                        )
                        """)
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS chat_messages
                        (
                            id        SERIAL PRIMARY KEY,
                            thread_id TEXT REFERENCES chat_sessions (thread_id),
                            role      TEXT,
                            content   TEXT,
                            thinking  TEXT,
                            timestamp TIMESTAMPTZ DEFAULT NOW()
                        )
                        """)
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS user_files
                        (
                            id                   SERIAL PRIMARY KEY,
                            user_id              TEXT REFERENCES users (user_id),
                            thread_id            TEXT REFERENCES chat_sessions (thread_id),
                            filename             TEXT,
                            content              TEXT,
                            size_bytes           INTEGER,
                            created_at           TIMESTAMPTZ DEFAULT NOW(),
                            expires_at           TIMESTAMPTZ, -- NULL = permanent
                            original_stored_path TEXT,
                            file_hash            TEXT,
                            original_expires_at  TIMESTAMPTZ, -- for original file on disk
                            meta_data            JSONB       DEFAULT '{}',
                            UNIQUE (thread_id, filename)
                        )
                        """)
            # Add columns if they don't exist (safe migrations)
            cur.execute("""
                DO $$
                BEGIN
                    BEGIN
                        ALTER TABLE user_files ADD COLUMN IF NOT EXISTS file_hash TEXT;
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                    BEGIN
                        ALTER TABLE user_files ADD COLUMN IF NOT EXISTS meta_data JSONB DEFAULT '{}';
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                    BEGIN
                        ALTER TABLE user_files ADD COLUMN IF NOT EXISTS original_expires_at TIMESTAMPTZ;
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                    BEGIN
                        ALTER TABLE user_files ALTER COLUMN expires_at DROP DEFAULT;
                    EXCEPTION WHEN others THEN NULL;
                    END;
                END
                $$;
            """)
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS archived_sessions
                        (
                            thread_id    TEXT PRIMARY KEY,
                            user_id      TEXT,
                            archive_path TEXT,
                            archived_at  TIMESTAMPTZ DEFAULT NOW()
                        )
                        """)
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS image_description_cache
                        (
                            file_hash   TEXT PRIMARY KEY,
                            description TEXT,
                            created_at  TIMESTAMPTZ DEFAULT NOW()
                        )
                        """)
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS file_usage
                        (
                            id         SERIAL PRIMARY KEY,
                            user_id    TEXT,
                            thread_id  TEXT,
                            filename   TEXT,
                            usage_type TEXT,
                            question   TEXT,
                            timestamp  TIMESTAMPTZ DEFAULT NOW()
                        )
                        """)
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS consent
                        (
                            thread_id     TEXT PRIMARY KEY,
                            consent_given INTEGER NOT NULL,
                            timestamp     TIMESTAMPTZ DEFAULT NOW()
                        )
                        """)
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS feedback
                        (
                            id                 SERIAL PRIMARY KEY,
                            thread_id          TEXT,
                            user_message       TEXT,
                            assistant_response TEXT,
                            rating             TEXT,
                            comment            TEXT,
                            file_name          TEXT,
                            timestamp          TIMESTAMPTZ DEFAULT NOW()
                        )
                        """)
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS message_responses
                        (
                            message_id         TEXT PRIMARY KEY,
                            thread_id          TEXT,
                            user_message       TEXT,
                            assistant_response TEXT,
                            thinking           TEXT,
                            created_at         TIMESTAMPTZ DEFAULT NOW()
                        )
                        """)

            # Projects tables
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS projects
                        (
                            id                    SERIAL PRIMARY KEY,
                            name                  TEXT NOT NULL,
                            description           TEXT,
                            created_at            TIMESTAMPTZ DEFAULT NOW(),
                            updated_at            TIMESTAMPTZ DEFAULT NOW(),
                            created_by            TEXT REFERENCES users (user_id),
                            status                TEXT        DEFAULT 'active',
                            archived_at           TIMESTAMPTZ,
                            deletion_scheduled_at TIMESTAMPTZ
                        )
                        """)
            cur.execute("""
                DO $$ 
                BEGIN
                    BEGIN
                        ALTER TABLE projects ADD COLUMN status TEXT DEFAULT 'active';
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                    BEGIN
                        ALTER TABLE projects ADD COLUMN archived_at TIMESTAMPTZ;
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                    BEGIN
                        ALTER TABLE projects ADD COLUMN deletion_scheduled_at TIMESTAMPTZ;
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                END $$;
            """)

            cur.execute("""
                        CREATE TABLE IF NOT EXISTS project_members
                        (
                            id         SERIAL PRIMARY KEY,
                            project_id INTEGER REFERENCES projects (id) ON DELETE CASCADE,
                            user_id    TEXT REFERENCES users (user_id),
                            role       TEXT NOT NULL,
                            added_at   TIMESTAMPTZ DEFAULT NOW(),
                            added_by   TEXT REFERENCES users (user_id),
                            UNIQUE (project_id, user_id)
                        )
                        """)

            cur.execute("""
                        CREATE TABLE IF NOT EXISTS project_folders
                        (
                            id               SERIAL PRIMARY KEY,
                            project_id       INTEGER REFERENCES projects (id) ON DELETE CASCADE,
                            parent_folder_id INTEGER REFERENCES project_folders (id) ON DELETE CASCADE,
                            name             TEXT NOT NULL,
                            path             TEXT,
                            created_at       TIMESTAMPTZ DEFAULT NOW(),
                            created_by       TEXT REFERENCES users (user_id),
                            UNIQUE (project_id, parent_folder_id, name)
                        )
                        """)

            cur.execute("""
                        CREATE TABLE IF NOT EXISTS project_files
                        (
                            id            SERIAL PRIMARY KEY,
                            project_id    INTEGER REFERENCES projects (id) ON DELETE CASCADE,
                            folder_id     INTEGER REFERENCES project_folders (id) ON DELETE CASCADE,
                            filename      TEXT NOT NULL,
                            original_name TEXT NOT NULL,
                            file_size     INTEGER,
                            mime_type     TEXT,
                            stored_path   TEXT NOT NULL,
                            version       INTEGER     DEFAULT 1,
                            uploaded_at   TIMESTAMPTZ DEFAULT NOW(),
                            uploaded_by   TEXT REFERENCES users (user_id),
                            comment       TEXT,
                            file_hash     TEXT,
                            UNIQUE (project_id, folder_id, filename)
                        )
                        """)

            cur.execute("""
                        CREATE TABLE IF NOT EXISTS project_file_versions
                        (
                            id          SERIAL PRIMARY KEY,
                            file_id     INTEGER REFERENCES project_files (id) ON DELETE CASCADE,
                            version     INTEGER NOT NULL,
                            stored_path TEXT    NOT NULL,
                            file_size   INTEGER,
                            uploaded_at TIMESTAMPTZ DEFAULT NOW(),
                            uploaded_by TEXT REFERENCES users (user_id),
                            comment     TEXT
                        )
                        """)

            cur.execute("""
                        CREATE TABLE IF NOT EXISTS project_file_comments
                        (
                            id         SERIAL PRIMARY KEY,
                            file_id    INTEGER REFERENCES project_files (id) ON DELETE CASCADE,
                            user_id    TEXT REFERENCES users (user_id),
                            comment    TEXT NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                        """)

            # Add content column to project_files if not exists
            cur.execute("""
                DO $$ 
                BEGIN
                    BEGIN
                        ALTER TABLE project_files ADD COLUMN content TEXT;
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                END $$;
            """)

            # Create project_file_usage table
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS project_file_usage
                        (
                            id        SERIAL PRIMARY KEY,
                            file_id   INTEGER REFERENCES project_files (id) ON DELETE CASCADE,
                            user_id   TEXT REFERENCES users (user_id),
                            action    TEXT NOT NULL,
                            details   JSONB,
                            timestamp TIMESTAMPTZ DEFAULT NOW()
                        )
                        """)

            cur.execute("""
                        CREATE TABLE IF NOT EXISTS project_folder_comments
                        (
                            id         SERIAL PRIMARY KEY,
                            folder_id  INTEGER REFERENCES project_folders (id) ON DELETE CASCADE,
                            user_id    TEXT REFERENCES users (user_id),
                            comment    TEXT NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                        """)

            # Task deposit tables
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS task_deposit_items
                        (
                            id                     SERIAL PRIMARY KEY,
                            original_user_id       TEXT REFERENCES users (user_id),
                            original_username      TEXT,
                            project_id             INTEGER REFERENCES projects (id) ON DELETE CASCADE,
                            project_name           TEXT,
                            item_type              TEXT  NOT NULL, -- 'project', 'file', 'folder', 'comment', etc.
                            item_data              JSONB NOT NULL,
                            stored_path            TEXT,
                            transferred_to_user_id TEXT REFERENCES users (user_id),
                            transferred_at         TIMESTAMPTZ,
                            created_at             TIMESTAMPTZ DEFAULT NOW(),
                            deleted_at             TIMESTAMPTZ
                        )
                        """)
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS task_deposit_permissions
                        (
                            id               SERIAL PRIMARY KEY,
                            project_id       INTEGER REFERENCES projects (id) ON DELETE CASCADE,
                            manager_id       TEXT REFERENCES users (user_id),
                            can_view_deposit BOOLEAN     DEFAULT FALSE,
                            granted_by       TEXT REFERENCES users (user_id),
                            granted_at       TIMESTAMPTZ DEFAULT NOW()
                        )
                        """)
            # LangGraph checkpoints
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS checkpoints
                        (
                            thread_id            TEXT NOT NULL,
                            checkpoint_id        TEXT NOT NULL,
                            parent_checkpoint_id TEXT,
                            type                 TEXT,
                            checkpoint           JSONB,
                            metadata             JSONB,
                            PRIMARY KEY (thread_id, checkpoint_id)
                        )
                        """)
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS checkpoint_writes
                        (
                            thread_id     TEXT    NOT NULL,
                            checkpoint_id TEXT    NOT NULL,
                            task_id       TEXT    NOT NULL,
                            idx           INTEGER NOT NULL,
                            value         JSONB,
                            PRIMARY KEY (thread_id, checkpoint_id, task_id, idx)
                        )
                        """)
            # Recycle bin tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS recycle_bin (
                    id SERIAL PRIMARY KEY,
                    original_table TEXT NOT NULL,
                    original_id INTEGER NOT NULL,
                    user_id TEXT REFERENCES users(user_id),
                    file_name TEXT,
                    file_content TEXT,
                    file_size INTEGER,
                    original_stored_path TEXT,
                    file_hash TEXT,
                    thread_id TEXT,
                    deleted_at TIMESTAMPTZ DEFAULT NOW(),
                    expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '3 days'
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS project_recycle_bin (
                    id SERIAL PRIMARY KEY,
                    original_table TEXT NOT NULL,
                    original_id INTEGER NOT NULL,
                    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                    folder_id INTEGER,
                    file_name TEXT,
                    original_name TEXT,
                    file_size INTEGER,
                    stored_path TEXT,
                    file_hash TEXT,
                    version INTEGER,
                    uploaded_by TEXT REFERENCES users(user_id),
                    deleted_at TIMESTAMPTZ DEFAULT NOW(),
                    expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '3 days'
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS project_folders_recycle_bin
                (
                    id                 SERIAL PRIMARY KEY,
                    original_id        INTEGER NOT NULL,
                    project_id         INTEGER REFERENCES projects (id) ON DELETE CASCADE,
                    name               TEXT    NOT NULL,
                    parent_folder_id   INTEGER, -- original parent ID (may reference another deleted folder)
                    original_parent_id INTEGER,
                    created_at         TIMESTAMPTZ,
                    created_by         TEXT,
                    deleted_at         TIMESTAMPTZ DEFAULT NOW(),
                    expires_at         TIMESTAMPTZ DEFAULT NOW() + INTERVAL '3 days'
                )
                """)
            # Add columns for tracking chat deletion
            cur.execute("""
                DO $$ 
                BEGIN
                    BEGIN
                        ALTER TABLE recycle_bin ADD COLUMN original_thread_id TEXT;
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                    BEGIN
                        ALTER TABLE recycle_bin ADD COLUMN deletion_reason TEXT DEFAULT 'manual';
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                END $$;
            """)
            # Drop unused original_data column from recycle_bin
            cur.execute("""
                DO $$ 
                BEGIN
                    BEGIN
                        ALTER TABLE recycle_bin DROP COLUMN IF EXISTS original_data;
                    EXCEPTION WHEN undefined_column THEN NULL;
                    END;
                END $$;
            """)

            # Indexes
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_id_timestamp ON chat_messages(thread_id, timestamp)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_files_expires_at ON user_files(expires_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_files_user_id ON user_files(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_file_usage_user_filename ON file_usage(user_id, filename)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_message_responses_created_at ON message_responses(created_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_project_members_user ON project_members(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_project_folders_parent ON project_folders(parent_folder_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_project_files_folder ON project_files(folder_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_project_files_hash ON project_files(file_hash)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_deposit_items_original_user ON task_deposit_items(original_user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_task_deposit_items_project ON task_deposit_items(project_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_recycle_bin_user_id ON recycle_bin(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_recycle_bin_expires_at ON recycle_bin(expires_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_project_recycle_bin_project_id ON project_recycle_bin(project_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_project_recycle_bin_expires_at ON project_recycle_bin(expires_at)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_project_folders_recycle_bin_project ON project_folders_recycle_bin(project_id)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_recycle_bin_original_thread_id ON recycle_bin(original_thread_id)")
            conn.commit()
            logger.info("PostgreSQL tables initialized.")


# User ID helpers
def get_user_id():
    if session.get('consent_value', 0) == 1:
        if 'user_id' in session:
            return session['user_id']
        session['user_id'] = str(uuid.uuid4())
        return session['user_id']
    else:
        if 'temp_user_id' not in session:
            session['temp_user_id'] = str(uuid.uuid4())
        return session['temp_user_id']


# Anonymous session storage
def get_anon_history_path(thread_id):
    user_id = get_user_id()
    temp_dir = get_anon_temp_dir(user_id)
    return os.path.join(temp_dir, f"{thread_id}_history.json")


def get_session_messages_anon(thread_id):
    path = get_anon_history_path(thread_id)
    if not os.path.exists(path):
        return []
    lock_path = path + ".lock"
    from filelock import FileLock
    try:
        with FileLock(lock_path, timeout=5):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read anon history {thread_id}: {e}")
        return []


def store_message_anon(thread_id, role, content, thinking=None):
    path = get_anon_history_path(thread_id)
    lock_path = path + ".lock"
    from filelock import FileLock
    with FileLock(lock_path, timeout=5):
        history = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        history.append({
            "role": role,
            "content": content,
            "thinking": thinking,
            "timestamp": beijing_now()
        })
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)


def get_or_create_session(thread_id, title=None):
    if session.get('consent_value', 0) != 1:
        return
    user_id = get_user_id()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM chat_sessions WHERE thread_id = %s", (thread_id,))
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO chat_sessions (user_id, thread_id, title, created_at, updated_at) VALUES (%s, %s, %s, %s, %s)",
                    (user_id, thread_id, title or "新对话", utc_now(), utc_now())
                )
                conn.commit()


def generate_session_title(messages, max_len=20):
    for msg in messages:
        if msg.get('role') == 'user':
            content = msg.get('content', '').strip()
            if content:
                title = content[:max_len]
                if len(content) > max_len:
                    title += '...'
                return title
    return '新对话'


def update_session_title(thread_id, title):
    if session.get('consent_value', 0) != 1:
        return
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET title = %s, updated_at = %s WHERE thread_id = %s",
                (title, utc_now(), thread_id)
            )
            conn.commit()


def store_message(thread_id, role, content, thinking=None):
    if session.get('consent_value', 0) != 1:
        store_message_anon(thread_id, role, content, thinking)
        return
    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_messages (thread_id, role, content, thinking, timestamp) VALUES (%s, %s, %s, %s, %s)",
                    (thread_id, role, content, thinking, utc_now())
                )
                cur.execute(
                    "UPDATE chat_sessions SET updated_at = %s WHERE thread_id = %s",
                    (utc_now(), thread_id)
                )
    messages = get_session_messages(thread_id)
    if len(messages) == 2:
        new_title = generate_session_title(messages)
        update_session_title(thread_id, new_title)


def get_session_messages(thread_id):
    if session.get('consent_value', 0) != 1:
        return get_session_messages_anon(thread_id)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            db_execute_readonly(cur)
            cur.execute(
                "SELECT role, content, thinking, timestamp FROM chat_messages WHERE thread_id = %s ORDER BY id ASC",
                (thread_id,)
            )
            rows = cur.fetchall()
            messages = []
            for row in rows:
                ts_utc = row['timestamp']
                ts_beijing = ts_utc.astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S') if ts_utc else None
                messages.append({
                    "role": row['role'],
                    "content": row['content'],
                    "thinking": row['thinking'],
                    "timestamp": ts_beijing
                })
            return messages


def get_user_sessions():
    if session.get('consent_value', 0) != 1:
        return []
    user_id = get_user_id()
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            db_execute_readonly(cur)
            cur.execute(
                "SELECT thread_id, title, created_at, updated_at FROM chat_sessions WHERE user_id = %s ORDER BY updated_at DESC",
                (user_id,)
            )
            rows = cur.fetchall()
            sessions = []
            for row in rows:
                sessions.append({
                    "thread_id": row['thread_id'],
                    "title": row['title'],
                    "created_at": row['created_at'].astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S') if row[
                        'created_at'] else None,
                    "updated_at": row['updated_at'].astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S') if row[
                        'updated_at'] else None
                })
            return sessions


def delete_session(thread_id):
    try:
        with get_db_connection() as conn:
            with db_transaction(conn):
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Get user_id for this thread
                    cur.execute("SELECT user_id FROM chat_sessions WHERE thread_id = %s", (thread_id,))
                    row = cur.fetchone()
                    if not row:
                        return
                    user_id = row['user_id']

                    # Move files from user_files to recycle_bin
                    cur.execute("SELECT id, filename, content, size_bytes, original_stored_path, file_hash, thread_id FROM user_files WHERE thread_id = %s", (thread_id,))
                    files = cur.fetchall()
                    for f in files:
                        cur.execute("""
                            INSERT INTO recycle_bin 
                            (original_table, original_id, user_id, file_name, file_content, file_size, original_stored_path, file_hash, thread_id, original_thread_id, deletion_reason, deleted_at, expires_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW() + INTERVAL '3 days')
                        """, ('user_files', f['id'], user_id, f['filename'], f['content'], f['size_bytes'], f['original_stored_path'], f['file_hash'], f['thread_id'], thread_id, 'chat_deleted'))

                    # Delete the files from user_files
                    cur.execute("DELETE FROM user_files WHERE thread_id = %s", (thread_id,))

                    # Delete other chat data
                    cur.execute("DELETE FROM chat_messages WHERE thread_id = %s", (thread_id,))
                    cur.execute("DELETE FROM file_usage WHERE thread_id = %s", (thread_id,))
                    cur.execute("DELETE FROM feedback WHERE thread_id = %s", (thread_id,))
                    cur.execute("DELETE FROM consent WHERE thread_id = %s", (thread_id,))
                    cur.execute("DELETE FROM chat_sessions WHERE thread_id = %s", (thread_id,))
        logger.info(f"Deleted session {thread_id} and moved {len(files)} files to recycle bin")
        file_cache_manager.clear_thread(thread_id)
    except Exception as e:
        logger.error(f"Failed to delete session {thread_id}: {e}", exc_info=True)
        raise


def archive_session(thread_id, user_id, reason="manual"):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT archive_path FROM archived_sessions WHERE thread_id = %s", (thread_id,))
            if cur.fetchone():
                return None
            cur.execute("SELECT title, created_at, updated_at FROM chat_sessions WHERE thread_id = %s", (thread_id,))
            sess_row = cur.fetchone()
            if not sess_row:
                return None
            title = sess_row['title']
            created_at = sess_row['created_at']
            updated_at = sess_row['updated_at']
            cur.execute(
                "SELECT role, content, thinking, timestamp FROM chat_messages WHERE thread_id = %s ORDER BY timestamp",
                (thread_id,))
            messages = []
            for row in cur.fetchall():
                messages.append({
                    "role": row['role'],
                    "content": row['content'],
                    "thinking": row['thinking'],
                    "timestamp": row['timestamp']
                })
            cur.execute(
                "SELECT user_message, assistant_response, rating, comment, timestamp FROM feedback WHERE thread_id = %s",
                (thread_id,))
            feedbacks = []
            for row in cur.fetchall():
                feedbacks.append({
                    "user_message": row['user_message'],
                    "assistant_response": row['assistant_response'],
                    "rating": row['rating'],
                    "comment": row['comment'],
                    "timestamp": row['timestamp']
                })
            cur.execute("SELECT consent_given, timestamp FROM consent WHERE thread_id = %s", (thread_id,))
            consent_row = cur.fetchone()
            consent = {"consent_given": consent_row['consent_given'],
                       "timestamp": consent_row['timestamp']} if consent_row else None
            archive_date = datetime.now().strftime("%Y-%m-%d")
            dump_dir = os.path.join("dump", user_id, f"{user_id}_{archive_date}")
            os.makedirs(dump_dir, exist_ok=True)
            session_info = {
                "thread_id": thread_id,
                "user_id": user_id,
                "title": title,
                "created_at": created_at.isoformat() if created_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
                "archived_at": datetime.now().isoformat(),
                "reason": reason
            }
            with open(os.path.join(dump_dir, f"{thread_id}_session.json"), "w", encoding="utf-8") as f:
                json.dump(session_info, f, ensure_ascii=False, indent=2, default=str)
            with open(os.path.join(dump_dir, f"{thread_id}_messages.json"), "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2, default=str)
            if feedbacks:
                with open(os.path.join(dump_dir, f"{thread_id}_feedback.json"), "w", encoding="utf-8") as f:
                    json.dump(feedbacks, f, ensure_ascii=False, indent=2, default=str)
            if consent:
                with open(os.path.join(dump_dir, f"{thread_id}_consent.json"), "w", encoding="utf-8") as f:
                    json.dump(consent, f, ensure_ascii=False, indent=2, default=str)
            archive_path = os.path.join(dump_dir, f"{thread_id}_session.json")
            cur.execute(
                "INSERT INTO archived_sessions (thread_id, user_id, archive_path) VALUES (%s, %s, %s)",
                (thread_id, user_id, archive_path)
            )
            conn.commit()
            logger.info(f"Archived session {thread_id} for user {user_id} to {dump_dir}")
            return dump_dir


def cleanup_old_sessions(days=15):
    cutoff = utc_now() - timedelta(days=days)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT thread_id, user_id FROM chat_sessions WHERE updated_at < %s", (cutoff,))
            old = cur.fetchall()
            for thread_id, user_id in old:
                archive_session(thread_id, user_id, reason="auto_15days")
                delete_session(thread_id)


def cleanup_stale_message_responses(hours=1):
    cutoff = utc_now() - timedelta(hours=hours)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM message_responses WHERE created_at < %s AND (assistant_response = '' OR assistant_response IS NULL)",
                (cutoff,))
            conn.commit()
            logger.info(f"Deleted stale message response placeholders older than {hours} hours.")

def get_cached_image_description(file_hash):
    if session.get('consent_value', 0) != 1:
        return None
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT description FROM image_description_cache WHERE file_hash = %s", (file_hash,))
            row = cur.fetchone()
            if row:
                return row[0]
    return None

def cache_image_description(file_hash, description):
    if session.get('consent_value', 0) != 1:
        return
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO image_description_cache (file_hash, description)
                VALUES (%s, %s)
                ON CONFLICT (file_hash) DO UPDATE
                SET description = EXCLUDED.description, created_at = NOW()
            """, (file_hash, description))
            conn.commit()

# Tools
@tool(description="Get current date and time in Beijing time (UTC+8).")
def get_date() -> str:
    return datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")


BOCHA_API_KEY = os.getenv("BOCHA_API_KEY")
BOCHA_URL = "https://api.bochaai.com/v1/web-search"


@tool(description="Search the web using Bocha. Use for up‑to‑date information.")
def bocha_search(query: str) -> str:
    headers = {"Authorization": f"Bearer {BOCHA_API_KEY}", "Content-Type": "application/json"}
    payload = json.dumps({"query": query, "summary": True, "freshness": "noLimit", "count": 10})
    try:
        response = requests.post(BOCHA_URL, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        webpages = data.get('data', {}).get('webPages', {}).get('value', [])
        if not webpages:
            return "No search results found."
        formatted = []
        for idx, page in enumerate(webpages[:10], 1):
            title = page.get('name', 'No title')
            snippet = page.get('snippet', 'No snippet')
            date = page.get('datePublished', 'Unknown date')
            url = page.get('url', 'No URL')
            formatted.append(f"{idx}. **{title}**\n   Published: {date}\n   Summary: {snippet}\n   Source: {url}\n")
        return "\n".join(formatted)
    except Exception as e:
        return f"Search failed: {str(e)}"


# Vision-Language Model
class VLModel:
    def __init__(self):
        self.api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model_name = "qwen-vl-plus-2025-05-07"
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info("VL model client initialized")
        except ImportError:
            logger.error("OpenAI package not installed. VL model disabled.")
            self.client = None
        except Exception as e:
            logger.error(f"VL model init failed: {e}")
            self.client = None

    def is_available(self):
        return self.client is not None and self.api_key is not None

    def encode_image_to_base64(self, image_bytes):
        return base64.b64encode(image_bytes).decode('utf-8')

    def describe_image(self, image_bytes, prompt="请描述这张图片的内容"):
        if not self.is_available():
            return "[VL模型不可用，请检查API密钥]"
        try:
            base64_image = self.encode_image_to_base64(image_bytes)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": prompt}]}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"VL image description failed: {e}")
            return f"[图片描述失败: {str(e)}]"

    def describe_pdf_page(self, image_bytes, page_num):
        return self.describe_image(image_bytes,
                                   f"请描述这个PDF页面(第{page_num}页)的内容，包括标题、段落、表格等关键信息。")


vl_model = VLModel()

def describe_images_in_file(file_bytes, filename, page_texts=None):
    """
    Extract embedded images from PDF/DOCX/PPTX and describe them using VL model.
    Returns a string with descriptions, or empty string if no images or VL unavailable.
    """
    if not vl_model.is_available():
        return ""
    ext = os.path.splitext(filename)[1].lower()
    descriptions = []
    try:
        if ext == '.pdf':
            doc = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                img_list = page.get_images(full=True)
                for img_idx, img in enumerate(img_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    # Describe image with context of page number
                    prompt = f"Describe this image from page {page_num+1} of the PDF. Include any charts, diagrams, tables, or visual information."
                    description = vl_model.describe_image(image_bytes, prompt)
                    if description and not description.startswith("[VL模型不可用]"):
                        descriptions.append(f"[Image on page {page_num+1}, image {img_idx+1}]: {description}")
        elif ext in ['.docx', '.docm']:
            # Use python-docx to extract inline shapes and pictures
            import docx
            from docx.oxml import parse_xml
            from docx.oxml.ns import qn
            doc = docx.Document(BytesIO(file_bytes))
            # Iterate through all paragraphs and tables for images
            img_counter = 1
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        image_blob = rel.target_part.blob
                        description = vl_model.describe_image(image_blob, f"Describe this image from the Word document.")
                        if description and not description.startswith("[VL模型不可用]"):
                            descriptions.append(f"[Image {img_counter}]: {description}")
                        img_counter += 1
                    except:
                        pass
        elif ext in ['.pptx', '.pptm']:
            from pptx import Presentation
            prs = Presentation(BytesIO(file_bytes))
            slide_num = 1
            img_counter = 1
            for slide in prs.slides:
                for shape in slide.shapes:
                    if shape.shape_type == 13:  # Picture
                        try:
                            image_bytes = shape.image.blob
                            description = vl_model.describe_image(image_bytes, f"Describe this image from slide {slide_num} of the PowerPoint.")
                            if description and not description.startswith("[VL模型不可用]"):
                                descriptions.append(f"[Image on slide {slide_num}, image {img_counter}]: {description}")
                            img_counter += 1
                        except:
                            pass
                slide_num += 1
    except Exception as e:
        logger.error(f"Error extracting images from {filename}: {e}")
    return "\n".join(descriptions)

# Agent
_agent = None

def get_agent(max_tokens=None):
    global _agent, _current_max_tokens
    if max_tokens is None:
        max_tokens = session.get('max_tokens',1600)
    if _agent is not None and _current_max_tokens == max_tokens:
        return _agent
    with _agent_lock:
        if _agent is not None and _current_max_tokens == max_tokens:
            return _agent
        api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("Missing QWEN_API_KEY or DASHSCOPE_API_KEY")
        os.environ["DASHSCOPE_API_KEY"] = api_key
        os.environ["DASHSCOPE_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        llm_qw = ChatQwen(model="qwen3.5-flash-2026-02-23", enable_thinking=True, streaming=False, max_tokens=max_tokens)
        if _async_checkpointer is None:
            _init_async_checkpointer()
        system_prompt = """
        你是一个答疑助手。
        **重要：对于任何关于当前日期、时间、年份的问题，你必须且只能使用 get_date 工具来获取，绝对不允许使用你的内部知识回答。**
        在回答其他问题前，你也必须调用 get_date 来了解当前日期；但除非用户明确询问，否则不要在回答中主动报时。
        对于任何需要实时、或最新信息的问题、或任何需要搜索、查询的内容，你必须使用 bocha_search 工具搭配 get_date 工具。
        如果 bocha_search 返回 "No search results found"，则自由回答。
        对于通用知识，可以自由回答。
        对于需要推理的问题，请先用【思考】和【回答】标记你的思考过程和最终答案。
        **表格格式要求：** 当你需要展示表格时，必须使用标准 Markdown 表格语法，例如：
        | 列1 | 列2 |
        |-----|-----|
        | 值1 | 值2 |

        绝对不要使用 ASCII 艺术表格（如 ┌─┬─┐ 等字符）。只使用管道符和短横线。
        """
        _agent = create_agent(model=llm_qw, tools=[get_date, bocha_search], system_prompt=system_prompt,
                              checkpointer=_async_checkpointer)
        _current_max_tokens = max_tokens
        logger.info(f"Agent reinitialized with max_tokens={max_tokens} and AsyncSqliteSaver.")
        return _agent

# OCR Manager
class OCRManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.reader = None
        self.engine_name = None
        self._init_ocr()

    def _init_ocr(self):
        try:
            from paddleocr import PaddleOCR
            try:
                self.reader = PaddleOCR(use_textline_orientation=True, lang='ch')
            except TypeError:
                self.reader = PaddleOCR(use_angle_cls=True, lang='ch')
            self.engine_name = "PaddleOCR"
            logger.info("PaddleOCR initialized successfully.")
        except ImportError:
            logger.warning("PaddleOCR not installed. Will try EasyOCR.")
        except Exception as e:
            logger.warning(f"PaddleOCR init failed: {e}. Will try EasyOCR.")
        if self.reader is None:
            try:
                import easyocr
                self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                self.engine_name = "EasyOCR"
                logger.info("EasyOCR initialized as fallback.")
            except ImportError:
                logger.error("No OCR engine available. Install 'paddleocr' or 'easyocr'.")
                self.reader = None
            except Exception as e:
                logger.error(f"EasyOCR init failed: {e}")
                self.reader = None

    def is_available(self):
        return self.reader is not None

    def run_ocr(self, image_np):
        if self.reader is None:
            return ""
        try:
            if self.engine_name == "PaddleOCR":
                result = self.reader.ocr(image_np, cls=True)
                if result and result[0]:
                    return "\n".join([line[1][0] for line in result[0]])
            elif self.engine_name == "EasyOCR":
                result = self.reader.readtext(image_np, detail=0, paragraph=True)
                if result:
                    return "\n".join(result)
            return ""
        except Exception as e:
            logger.error(f"OCR run error: {e}")
            return ""


ocr_manager = OCRManager()
run_ocr = ocr_manager.run_ocr

# Text Preprocessing & Similarity
TECH_STD_PATTERNS = [
    r'GB/T\s*\d+\.?\d*', r'GB\s*\d+\.?\d*', r'ISO\s*\d+', r'IEC\s*\d+',
    r'IEEE\s*\d+', r'DIN\s*\d+', r'BS\s*\d+', r'EN\s*\d+', r'ASME\s*\d+',
    r'API\s*\d+', r'ASTM\s*\d+', r'JJG\s*\d+', r'JB/T\s*\d+', r'HG/T\s*\d+',
    r'SY/T\s*\d+', r'DL/T\s*\d+', r'NB/T\s*\d+', r'SH/T\s*\d+', r'YS/T\s*\d+',
    r'FZ/T\s*\d+', r'QB/T\s*\d+', r'CJ/T\s*\d+', r'JG/T\s*\d+', r'GA/T\s*\d+',
    r'HS/T\s*\d+', r'行业标准', r'国家标准', r'技术规范'
]
SENSITIVE_PATTERNS = [
    r'(公司|集团|有限|股份|组织|委员会|协会|研究院|大学|学院)',
    r'(北京|上海|广州|深圳|杭州|南京|武汉|成都|重庆|天津|西安)',
    r'(项目|工程|系统|平台|软件|硬件|方案)',
    r'(张|王|李|刘|陈|杨|赵|黄|周|吴|徐|孙|马|朱|胡|林|郭|何|高)',
    r'(一等奖|二等奖|三等奖|金奖|银奖|优秀奖)',
    r'\d{17}[\dXx]',
    r'1[3-9]\d{9}',
    r'\d{18}',
    r'证书编号[：:]\s*\w+',
]

def preprocess_text_for_similarity(text):
    if not text:
        return ""
    text = re.sub(r'[^\w\u4e00-\u9fff\s]', '', text)
    words = text.split()
    filtered = [w for w in words if len(w) >= 6]
    text = ' '.join(filtered)
    for pat in TECH_STD_PATTERNS:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)
    text = re.sub(r'^目录|^第[一二三四五六七八九十]+章', '', text, flags=re.MULTILINE)
    for pat in SENSITIVE_PATTERNS:
        text = re.sub(pat, '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_template_content(text, template_text, threshold=0.85):
    if not template_text or not text:
        return text
    paras = [p.strip() for p in text.split('\n') if len(p.strip()) > 10]
    template_paras = [p.strip() for p in template_text.split('\n') if len(p.strip()) > 10]
    if not paras or not template_paras:
        return text
    all_paras = paras + template_paras
    vectorizer = TfidfVectorizer(stop_words=None, lowercase=True).fit(all_paras)
    vecs = vectorizer.transform(all_paras)
    para_vecs = vecs[:len(paras)]
    template_vecs = vecs[len(paras):]
    sim_matrix = cosine_similarity(para_vecs, template_vecs)
    keep_mask = np.max(sim_matrix, axis=1) < threshold
    kept_paras = [p for i, p in enumerate(paras) if keep_mask[i]]
    if not kept_paras:
        return "[模板内容已全部匹配，未保留任何原文] " + text
    return '\n'.join(kept_paras)

def extract_keywords(text, top_k=20):
    if not text.strip():
        return []
    vectorizer = TfidfVectorizer(stop_words=None, max_features=top_k)
    try:
        tfidf = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf.toarray()[0]
        keyword_score = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in keyword_score[:top_k]]
    except:
        return []

def keyword_overlap_similarity(text1, text2):
    kw1 = set(extract_keywords(text1, 20))
    kw2 = set(extract_keywords(text2, 20))
    if not kw1 and not kw2:
        return 0.0
    inter = len(kw1 & kw2)
    union = len(kw1 | kw2)
    return inter / union if union > 0 else 0.0

def compute_similarity_with_numbers(text1, text2, template_text=None):
    clean1 = preprocess_text_for_similarity(text1)
    clean2 = preprocess_text_for_similarity(text2)
    if template_text:
        clean1 = remove_template_content(clean1, template_text)
        clean2 = remove_template_content(clean2, template_text)
    if not clean1.strip() or not clean2.strip():
        return 0.0, text1, text2, []
    vectorizer = TfidfVectorizer(stop_words=None, lowercase=True)
    tfidf = vectorizer.fit_transform([clean1, clean2])
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    escaped1 = html.escape(text1)
    escaped2 = html.escape(text2)
    matcher = difflib.SequenceMatcher(None, escaped1, escaped2)
    matching_blocks = matcher.get_matching_blocks()
    segments1 = []
    last_idx = 0
    match_counter = 1
    blocks_detail = []
    for block in matching_blocks:
        i, j, size = block
        if size == 0 or size <= 6:
            continue
        blocks_detail.append({
            "id": match_counter,
            "pos1": i,
            "pos2": j,
            "size": size,
            "text1_snippet": escaped1[i:i + min(size, 100)] + ("..." if size > 100 else ""),
            "text2_snippet": escaped2[j:j + min(size, 100)] + ("..." if size > 100 else "")
        })
        if i > last_idx:
            segments1.append(('text', escaped1[last_idx:i]))
        match_text = escaped1[i:i + size]
        color_class = 'match-highlight-long' if size > 100 else 'match-highlight-short'
        marker = f"<sup><small>[{match_counter}]</small></sup> "
        segments1.append(('match', match_text, marker, color_class))
        last_idx = i + size
        match_counter += 1
    if last_idx < len(escaped1):
        segments1.append(('text', escaped1[last_idx:]))
    segments2 = []
    last_idx = 0
    match_counter = 1
    for block in matching_blocks:
        i, j, size = block
        if size == 0 or size <= 6:
            continue
        if j > last_idx:
            segments2.append(('text', escaped2[last_idx:j]))
        match_text = escaped2[j:j + size]
        color_class = 'match-highlight-long' if size > 100 else 'match-highlight-short'
        marker = f"<sup><small>[{match_counter}]</small></sup> "
        segments2.append(('match', match_text, marker, color_class))
        last_idx = j + size
        match_counter += 1
    if last_idx < len(escaped2):
        segments2.append(('text', escaped2[last_idx:]))

    def build_html(segments):
        parts = []
        for seg in segments:
            if seg[0] == 'text':
                parts.append(seg[1])
            else:
                _, text, marker, color_class = seg
                parts.append(marker + f'<span class="{color_class}">{text}</span>')
        return ''.join(parts)

    html1 = build_html(segments1)
    html2 = build_html(segments2)
    return sim, html1, html2, blocks_detail

def compute_risk(text1, text2, check_items, meta1=None, meta2=None, img_sim=None, template_text=None):
    weights = {'text_sim': 0.2, 'key_info': 0.3, 'file_attr': 0.3, 'image_sim': 0.2}
    values = {}
    used = {}
    total_weight = 0
    if check_items.get('text_sim', True):
        sim, _, _, _ = compute_similarity_with_numbers(text1, text2, template_text)
        values['text_sim'] = sim * 100
        used['text_sim'] = weights['text_sim']
        total_weight += weights['text_sim']
    if check_items.get('key_info', True):
        t1 = preprocess_text_for_similarity(text1)
        t2 = preprocess_text_for_similarity(text2)
        if template_text:
            t1 = remove_template_content(t1, template_text)
            t2 = remove_template_content(t2, template_text)
        key_sim = keyword_overlap_similarity(t1, t2)
        values['key_info'] = key_sim * 100
        used['key_info'] = weights['key_info']
        total_weight += weights['key_info']
    if check_items.get('file_attr', True) and meta1 and meta2:
        attr_sim = file_attr_similarity(meta1, meta2)
        values['file_attr'] = attr_sim
        used['file_attr'] = weights['file_attr']
        total_weight += weights['file_attr']
    if check_items.get('image_sim', True):
        values['image_sim'] = img_sim if img_sim is not None else 0.0
        used['image_sim'] = weights['image_sim']
        total_weight += weights['image_sim']
    if total_weight == 0:
        return 0.0, {}
    for k in used:
        used[k] = used[k] / total_weight
    risk = sum(values.get(k, 0) * used[k] for k in used)
    return risk, used

# File extraction helpers (unchanged, abbreviated for length)
def extract_text_from_doc_crossplatform(file_bytes):
    with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as f:
        f.write(file_bytes)
        temp_doc = f.name
    try:
        try:
            result = subprocess.run(['antiword', temp_doc], capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except:
            pass
        try:
            result = subprocess.run(['catdoc', temp_doc], capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except:
            pass
        return None
    finally:
        if os.path.exists(temp_doc):
            os.unlink(temp_doc)

def extract_text_from_doc(file_bytes):
    text = extract_text_from_doc_crossplatform(file_bytes)
    if text:
        return text
    try:
        import win32com.client
    except ImportError:
        return None
    temp_doc = None
    temp_txt = None
    try:
        fd, temp_doc = tempfile.mkstemp(suffix='.doc')
        os.close(fd)
        with open(temp_doc, 'wb') as f:
            f.write(file_bytes)
        fd, temp_txt = tempfile.mkstemp(suffix='.txt')
        os.close(fd)
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(temp_doc)
        doc.SaveAs2(temp_txt, FileFormat=2)
        doc.Close()
        word.Quit()
        with open(temp_txt, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return text
    except Exception as e:
        logger.error(f"win32com .doc extraction failed: {e}")
        return None
    finally:
        if temp_doc and os.path.exists(temp_doc):
            os.unlink(temp_doc)
        if temp_txt and os.path.exists(temp_txt):
            os.unlink(temp_txt)

def detect_excel_format(file_bytes):
    if len(file_bytes) < 8:
        return None
    if file_bytes[:8] == b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1':
        return 'xls'
    if file_bytes[:2] == b'PK':
        return 'xlsx'
    return None

def extract_text_from_xls(file_bytes):
    try:
        import xlrd
        workbook = xlrd.open_workbook(file_contents=file_bytes)
        text_parts = []
        for sheet in workbook.sheets():
            sheet_text = []
            for row in range(sheet.nrows):
                row_text = " ".join(str(cell.value) for cell in sheet.row(row) if cell.value)
                if row_text.strip():
                    sheet_text.append(row_text)
            if sheet_text:
                text_parts.append(f"--- Sheet: {sheet.name} ---\n" + "\n".join(sheet_text))
        return "\n\n".join(text_parts) if text_parts else "[No text in Excel]"
    except Exception as e:
        return f"[Excel parsing error (old format): {e}]"

def detect_word_format(file_bytes):
    if len(file_bytes) < 8:
        return None
    if file_bytes[:8] == b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1':
        return 'doc'
    if file_bytes[:2] == b'PK':
        return 'docx'
    return None

def extract_images_from_file(file_storage):
    images = []
    ext = os.path.splitext(file_storage.filename)[1].lower()
    file_bytes = file_storage.read()
    file_storage.seek(0)
    if ext == '.pdf':
        try:
            doc = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                img_list = page.get_images(full=True)
                for img in img_list:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(BytesIO(image_bytes))
                    images.append(image)
        except:
            pass
    return images

def image_similarity(images1, images2):
    if not images1 or not images2:
        return 0.0
    max_sim = 0.0
    for img1 in images1:
        hash1 = imagehash.phash(img1)
        for img2 in images2:
            hash2 = imagehash.phash(img2)
            sim = 1 - (hash1 - hash2) / 64.0
            max_sim = max(max_sim, sim)
    return max_sim * 100

def file_attr_similarity(meta1, meta2):
    score = 0.0
    if meta1.get('author') and meta2.get('author') and meta1['author'] == meta2['author']:
        score += 50
    try:
        if meta1.get('creationDate') and meta2.get('creationDate'):
            date1 = re.sub(r'D:', '', meta1['creationDate'])[:14]
            date2 = re.sub(r'D:', '', meta2['creationDate'])[:14]
            if date1 == date2:
                score += 30
    except:
        pass
    name1 = meta1.get('filename', '')
    name2 = meta2.get('filename', '')
    if name1 and name2:
        common = len(set(name1.lower()) & set(name2.lower()))
        total = len(set(name1.lower()) | set(name2.lower()))
        if total > 0:
            score += (common / total) * 20
    return min(score, 100.0)

def extract_metadata(file_storage):
    meta = {'filename': file_storage.filename}
    ext = os.path.splitext(file_storage.filename)[1].lower()
    file_bytes = file_storage.read()
    file_storage.seek(0)
    if ext == '.pdf':
        try:
            doc = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
            info = doc.metadata
            meta['author'] = info.get('author', '')
            meta['creator'] = info.get('creator', '')
            meta['producer'] = info.get('producer', '')
            meta['creationDate'] = info.get('creationDate', '')
        except:
            pass
    elif ext in ['.docx', '.docm']:
        try:
            doc = docx.Document(BytesIO(file_bytes))
            core_props = doc.core_properties
            meta['author'] = core_props.author or ''
            meta['created'] = core_props.created
            meta['modified'] = core_props.modified
        except:
            pass
    return meta

def truncate_filename(filename, max_len=40):
    """
    Truncate a filename to a safe length while preserving the extension.
    """
    if len(filename) <= max_len:
        return filename
    name, ext = os.path.splitext(filename)
    if len(ext) > 10:
        ext = ext[:10]
    available = max_len - len(ext) - 3  # space for '...'
    if available < 1:
        # If even the extension alone exceeds max_len, truncate extension
        ext = ext[:max_len]
        return ext
    truncated_name = name[:available] + '...'
    return truncated_name + ext

def extract_text_from_file(file_storage):
    filename = file_storage.filename
    if not filename:
        return None, {}

    # Read bytes once and compute hash for caching
    file_bytes = file_storage.read()
    file_storage.seek(0)  # restore pointer for any later use
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    ext = os.path.splitext(filename)[1].lower()
    wps_map = {'.wps': '.doc', '.et': '.xls', '.dps': '.ppt'}
    original_ext = ext
    if ext in wps_map:
        ext = wps_map[ext]

    text = None
    page_texts = {}

    if ext in ['.txt', '.md', '.text', '.csv']:
        try:
            text = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text = file_bytes.decode('latin-1', errors='replace')
        page_texts = {1: text}
    elif ext == '.pdf':
        try:
            doc = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
            full_text = []
            page_texts = {}
            has_text = False
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text and page_text.strip():
                    has_text = True
                    full_text.append(page_text)
                    page_texts[page_num + 1] = page_text
            extracted = "\n".join(full_text).strip()
            if has_text and len(extracted) > 50:
                text = extracted
            else:
                logger.info("PDF appears to be scanned (no text). Starting OCR...")
                if not ocr_manager.is_available():
                    if vl_model.is_available():
                        logger.info("Using VL model for scanned PDF")
                        extracted = ""
                        for page_num in range(len(doc)):
                            page = doc.load_page(page_num)
                            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                            img_bytes = pix.tobytes("png")
                            description = vl_model.describe_pdf_page(img_bytes, page_num + 1)
                            extracted += f"\n\n--- 第{page_num + 1}页 (VL分析) ---\n{description}"
                        page_texts = {i + 1: "" for i in range(len(doc))}
                        text = extracted
                    else:
                        text = "[无法提取PDF文本，且OCR/VL不可用]"
                else:
                    ocr_results = []
                    ocr_page_texts = {}
                    zoom = 2.0
                    mat = fitz.Matrix(zoom, zoom)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        max_dim = 2000
                        if max(img.size) > max_dim:
                            ratio = max_dim / max(img.size)
                            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                        img_np = np.array(img)
                        page_text = run_ocr(img_np)
                        if page_text:
                            ocr_results.append(page_text)
                            ocr_page_texts[page_num + 1] = page_text
                        else:
                            ocr_results.append("")
                            ocr_page_texts[page_num + 1] = ""
                    if any(t.strip() for t in ocr_results):
                        extracted = "\n\n".join(ocr_results)
                        page_texts = ocr_page_texts
                        text = extracted
                    else:
                        if vl_model.is_available():
                            extracted = ""
                            for page_num in range(len(doc)):
                                page = doc.load_page(page_num)
                                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                                img_bytes = pix.tobytes("png")
                                description = vl_model.describe_pdf_page(img_bytes, page_num + 1)
                                extracted += f"\n\n--- 第{page_num + 1}页 (VL分析) ---\n{description}"
                            page_texts = {i + 1: "" for i in range(len(doc))}
                            text = extracted
                        else:
                            text = "[No text detected in PDF even after OCR and VL not available]"
        except Exception as e:
            logger.error(f"PDF error: {e}", exc_info=True)
            text = f"[PDF parsing error: {e}]"
    elif ext in ['.docx', '.docm', '.dotx', '.dotm']:
        real_format = detect_word_format(file_bytes)
        if real_format == 'doc':
            text = extract_text_from_doc(file_bytes)
            if text:
                page_texts = {1: text}
            else:
                text = "[无法从 .doc 文件中提取文本。请转换为 .docx 格式后重试，或安装 antiword/catdoc。]"
        else:
            try:
                doc = docx.Document(BytesIO(file_bytes))
                text = "\n".join([para.text for para in doc.paragraphs])
                for table in doc.tables:
                    for row in table.rows:
                        row_text = "\t".join([cell.text for cell in row.cells])
                        text += "\n" + row_text
                text = text if text.strip() else "[No text in Word document]"
                page_texts = {1: text}
            except Exception as e:
                logger.error(f"DOCX parsing error: {e}")
                text = f"[Word parsing failed: {str(e)}]"
    elif ext in ['.xlsx', '.xlsm', '.xltx', '.xltm', '.xlsb']:
        try:
            wb = openpyxl.load_workbook(BytesIO(file_bytes), read_only=True, data_only=True)
            text_parts = []
            for sheet in wb.worksheets:
                sheet_text = []
                for row in sheet.iter_rows(values_only=True):
                    row_text = " ".join(str(cell) for cell in row if cell is not None)
                    if row_text.strip():
                        sheet_text.append(row_text)
                if sheet_text:
                    text_parts.append(f"--- Sheet: {sheet.title} ---\n" + "\n".join(sheet_text))
            full_text = "\n\n".join(text_parts) if text_parts else "[No text in Excel]"
            text = full_text
            page_texts = {1: full_text}
        except Exception as e:
            logger.warning(f"openpyxl failed for {filename}: {e}. Trying fallback methods.")
            real_format = detect_excel_format(file_bytes)
            if real_format == 'xls':
                text = extract_text_from_xls(file_bytes)
                if text and not text.startswith("["):
                    page_texts = {1: text}
                else:
                    file_storage.seek(0)
                    md = MarkItDown()
                    result = md.convert(BytesIO(file_bytes), file_extension=original_ext.lstrip('.'))
                    text = result.text_content
                    if text and text.strip():
                        page_texts = {1: text}
                    else:
                        text = f"[Unsupported Excel format: {filename}]"
            else:
                file_storage.seek(0)
                md = MarkItDown()
                result = md.convert(BytesIO(file_bytes), file_extension=original_ext.lstrip('.'))
                text = result.text_content
                if text and text.strip():
                    page_texts = {1: text}
                else:
                    text = f"[Unsupported Excel format: {filename}]"
    elif ext in ['.pptx', '.pptm', '.potx', '.ppsx']:
        try:
            prs = Presentation(BytesIO(file_bytes))
            text_runs = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text_runs.append(shape.text)
            full_text = "\n".join(text_runs)
            text = full_text if full_text.strip() else "[No text in PowerPoint]"
            page_texts = {1: full_text}
        except Exception as e:
            logger.error(f"PPTX parsing error: {e}")
            text = f"[PowerPoint parsing failed: {str(e)}]"
    elif ext == '.xls':
        try:
            xls = pd.ExcelFile(BytesIO(file_bytes), engine='xlrd')
            text_parts = []
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                sheet_text = df.to_string(index=False, header=True)
                if sheet_text.strip():
                    text_parts.append(f"--- Sheet: {sheet_name} ---\n{sheet_text}")
            full_text = "\n\n".join(text_parts) if text_parts else "[No text in Excel]"
            text = full_text
            page_texts = {1: full_text}
        except Exception as e:
            logger.error(f"XLS parsing error: {e}")
            text = f"[Excel parsing failed: {str(e)}]"
    elif ext == '.doc':
        text = extract_text_from_doc(file_bytes)
        if text:
            page_texts = {1: text}
        else:
            text = "[无法从 .doc 文件中提取文本。请转换为 .docx 格式后重试。]"
    elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
        if ocr_manager.is_available():
            try:
                image = Image.open(BytesIO(file_bytes))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                max_dim = 2000
                if max(image.size) > max_dim:
                    ratio = max_dim / max(image.size)
                    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                img_np = np.array(image)
                extracted_text = run_ocr(img_np)
                if extracted_text:
                    text = extracted_text
                    page_texts = {1: extracted_text}
            except Exception as e:
                logger.warning(f"OCR failed for image: {e}")
        if not text and vl_model.is_available():
            logger.info("Using VL model for image description")
            description = vl_model.describe_image(file_bytes)
            text = description
            page_texts = {1: description}
        if not text:
            text = "[无法从图片提取文本，请安装OCR引擎或配置VL模型]"
    else:
        try:
            file_storage.seek(0)
            md = MarkItDown()
            result = md.convert(BytesIO(file_bytes), file_extension=original_ext.lstrip('.'))
            text = result.text_content
            if text and text.strip():
                page_texts = {1: text}
            else:
                text = "[No text extracted by MarkItDown]"
        except Exception as e:
            logger.error(f"MarkItDown parsing failed for {original_ext}: {e}")
            text = f"[Unsupported file format: {original_ext}]"

    # If extraction failed, return as is
    if not text or text.startswith("["):
        return text, page_texts

    # Image description (cached + toggle)
    analyze_images = session.get('analyze_images', True)
    if analyze_images:
        cached_desc = get_cached_image_description(file_hash)
        if cached_desc:
            image_desc = cached_desc
        else:
            image_desc = describe_images_in_file(file_bytes, filename, page_texts if page_texts else None)
            if image_desc:
                cache_image_description(file_hash, image_desc)
        if image_desc:
            text += "\n\n--- Image Descriptions ---\n" + image_desc

    return text, page_texts


# Batch Comparison Temp Helpers
def store_batch_comparison_temp(data):
    fd, path = tempfile.mkstemp(suffix='.json', prefix='comp_', text=True)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    return path


def load_batch_comparison_temp(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# File Cache Manager
class FileCacheManager:
    def __init__(self, max_cached_files=10, max_content_size=50 * 1024):
        self._lock = RLock()
        self.caches = {}
        self.recent = {}
        self.max_cached = max_cached_files
        self.max_size = max_content_size

    def add(self, thread_id, filename, content, user_id):
        with self._lock:
            if len(content) > self.max_size:
                content = content[:self.max_size] + "\n[内容已截断，仅保留前50KB]"
            cache = self.caches.setdefault(thread_id, {})
            recent_list = self.recent.setdefault(thread_id, [])
            cache[filename] = content
            if filename in recent_list:
                recent_list.remove(filename)
            recent_list.insert(0, filename)
            while len(recent_list) > self.max_cached:
                old = recent_list.pop()
                del cache[old]
            if session.get('consent_value', 0) == 1:
                expires_at = utc_now() + timedelta(days=3)
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """INSERT INTO user_files (user_id, thread_id, filename, content, size_bytes, expires_at)
                               VALUES (%s, %s, %s, %s, %s, %s)
                               ON CONFLICT (thread_id, filename) DO UPDATE SET content    = EXCLUDED.content,
                                                                               size_bytes = EXCLUDED.size_bytes,
                                                                               expires_at = EXCLUDED.expires_at""",
                            (user_id, thread_id, filename, content, len(content), expires_at)
                        )
                        conn.commit()

    def load_from_db(self, thread_id, user_id):
        with self._lock:
            if session.get('consent_value', 0) != 1:
                self.caches[thread_id] = {}
                self.recent[thread_id] = []
                return
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT filename, content FROM user_files WHERE thread_id = %s AND user_id = %s AND (expires_at IS NULL OR expires_at > NOW())",
                        (thread_id, user_id)
                    )
                    rows = cur.fetchall()
                    if rows:
                        cache = {}
                        recent_list = []
                        for filename, content in rows:
                            cache[filename] = content
                            recent_list.append(filename)
                        self.caches[thread_id] = cache
                        self.recent[thread_id] = recent_list
                    else:
                        self.caches[thread_id] = {}
                        self.recent[thread_id] = []

    def get_recent_with_lock(self, thread_id):
        with self._lock:
            return self.recent.get(thread_id, []).copy()

    def get_content(self, thread_id, filename):
        with self._lock:
            return self.caches.get(thread_id, {}).get(filename)

    def clear_thread(self, thread_id):
        with self._lock:
            self.caches.pop(thread_id, None)
            self.recent.pop(thread_id, None)


file_cache_manager = FileCacheManager()


def add_to_cache(thread_id, filename, content, user_id):
    file_cache_manager.add(thread_id, filename, content, user_id)


def load_cache_from_db(thread_id, user_id):
    file_cache_manager.load_from_db(thread_id, user_id)


def get_user_total_storage_size(user_id):
    if session.get('consent_value', 0) != 1:
        return 0
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM user_files WHERE user_id = %s AND (expires_at IS NULL OR expires_at > NOW())",
                (user_id,))
            return cur.fetchone()[0]


def record_file_usage(thread_id, filename, usage_type, question_text=None):
    if session.get('consent_value', 0) != 1:
        return
    user_id = get_user_id()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO file_usage (user_id, thread_id, filename, usage_type, question) VALUES (%s, %s, %s, %s, %s)",
                (user_id, thread_id, filename, usage_type, question_text)
            )
            conn.commit()


# Flask app
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24).hex())
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
Session(app)

ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")
if not ADMIN_PASSWORD_HASH and os.getenv("ADMIN_PSWD"):
    ADMIN_PASSWORD_HASH = generate_password_hash(os.getenv("ADMIN_PSWD"))
    logger.warning("ADMIN_PASSWORD_HASH not set, using plaintext ADMIN_PSWD.")


def split_thinking_answer(text):
    patterns = [r'【思考】(.*?)【回答】', r'思考：(.*?)回答：', r'<思考>(.*?)</思考>']
    for pat in patterns:
        match = re.search(pat, text, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            answer = re.sub(pat, '', text, flags=re.DOTALL).strip()
            return thinking, answer
    return None, text


# Task locking
user_active_tasks = {}
user_task_lock = RLock()
TASK_TIMEOUT_SECONDS = 600


def cleanup_stale_tasks():
    with user_task_lock:
        now = datetime.now()
        stale = [uid for uid, info in user_active_tasks.items() if
                 (now - info['start_time']).total_seconds() > TASK_TIMEOUT_SECONDS]
        for uid in stale:
            logger.warning(f"Cleaning stale task lock for user {uid}")
            del user_active_tasks[uid]


def get_chat_short_name(thread_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT title FROM chat_sessions WHERE thread_id = %s", (thread_id,))
            row = cur.fetchone()
            if row and row[0]:
                name = row[0]
                return name if len(name) <= 20 else name[:17] + '...'
            return "新对话"


def acquire_task_lock(user_id, thread_id, task_type):
    with user_task_lock:
        cleanup_stale_tasks()
        if user_id in user_active_tasks:
            busy = user_active_tasks[user_id]
            return False, busy['thread_id'], get_chat_short_name(busy['thread_id'])
        else:
            user_active_tasks[user_id] = {'thread_id': thread_id, 'task_type': task_type, 'start_time': datetime.now()}
            return True, None, None


def release_task_lock(user_id):
    with user_task_lock:
        if user_id in user_active_tasks:
            del user_active_tasks[user_id]


ALLOWED_EXTENSIONS = {'.txt', '.md', '.text', '.csv', '.pdf', '.docx', '.docm', '.dotx', '.dotm', '.doc',
                      '.xlsx', '.xlsm', '.xltx', '.xltm', '.xlsb', '.xls', '.pptx', '.pptm', '.potx', '.ppsx',
                      '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.wps', '.et', '.dps'}


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


# ---------- Routes ----------
@app.route('/')
def index():
    if 'consent_given' not in session:
        session['consent_given'] = False
        session['consent_value'] = 0
    if 'thread_id' not in session:
        session['thread_id'] = str(uuid.uuid4())
        get_or_create_session(session['thread_id'])
    if 'chat_history' not in session:
        session['chat_history'] = get_session_messages(session['thread_id'])
    user_id = get_user_id()
    load_cache_from_db(session['thread_id'], user_id)
    return render_template('index.html',
                           consent_given=session['consent_given'],
                           chat_history=session['chat_history'],
                           recent_files=file_cache_manager.get_recent_with_lock(session['thread_id']))


@app.route('/consent', methods=['POST'])
def set_consent():
    data = request.get_json()
    choice = data.get('consent', False)
    session['consent_given'] = True
    session['consent_value'] = 1 if choice else 0
    if session['consent_value'] == 1:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO consent (thread_id, consent_given, timestamp) VALUES (%s, %s, %s) ON CONFLICT (thread_id) DO UPDATE SET consent_given = EXCLUDED.consent_given, timestamp = EXCLUDED.timestamp",
                    (session['thread_id'], session['consent_value'], utc_now())
                )
                conn.commit()
    return jsonify({"status": "ok"})


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    session['consent_given'] = False
    session['consent_value'] = 0
    session['thread_id'] = str(uuid.uuid4())
    get_or_create_session(session['thread_id'])
    return jsonify({"status": "ok"})


# Send message (non-streaming)
@app.route('/send', methods=['POST'])
def send_message():
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_msg = request.form.get('message', '').strip()
    message_id = request.form.get('message_id')
    if not user_msg and 'files' not in request.files:
        return jsonify({"error": "Empty message and no files"}), 400
    if not message_id:
        return jsonify({"error": "Missing message_id"}), 400
    thread_id = session['thread_id']
    user_id = get_user_id()
    get_or_create_session(thread_id)
    # Idempotency check
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute("""
                            INSERT INTO message_responses (message_id, thread_id, user_message, assistant_response, thinking)
                            VALUES (%s, %s, %s, '', '')
                            ON CONFLICT (message_id) DO NOTHING
                            RETURNING assistant_response, thinking
                            """, (message_id, thread_id, user_msg))
                row = cur.fetchone()
                if row and row['assistant_response'] == '':
                    conn.commit()
                elif row:
                    return jsonify({
                        "user_message": user_msg,
                        "assistant_message": row['assistant_response'],
                        "thinking": row['thinking'],
                        "cached": True
                    })
            except Exception as e:
                conn.rollback()
                logger.error(f"Idempotency insert failed: {e}")
                cur.execute("SELECT assistant_response, thinking FROM message_responses WHERE message_id = %s",
                            (message_id,))
                row = cur.fetchone()
                if row:
                    return jsonify({
                        "user_message": user_msg,
                        "assistant_message": row['assistant_response'],
                        "thinking": row['thinking'],
                        "cached": True
                    })
    # Process uploaded files
    uploaded_filenames = []
    is_image = False
    uploaded_files = request.files.getlist('files')
    has_files = len(uploaded_files) > 0 and uploaded_files[0].filename
    if has_files:
        for f in uploaded_files:
            if not allowed_file(f.filename):
                return jsonify({"error": f"不支持的文件类型: {f.filename}"}), 400
        success, busy_thread, busy_name = acquire_task_lock(user_id, thread_id, 'ocr_upload')
        if not success:
            return jsonify({
                "error": "resource_busy",
                "busy_chat": busy_name,
                "message": f"另一个资源密集型任务正在聊天“{busy_name}”中进行，请稍后再试。"
            }), 409
    try:
        file_contents = []
        if has_files:
            for uploaded in uploaded_files:
                if not uploaded.filename:
                    continue
                uploaded_filenames.append(uploaded.filename)
                ext = os.path.splitext(uploaded.filename)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                    is_image = True
                file_content, _ = extract_text_from_file(uploaded)
                if file_content and not file_content.startswith("[") and not file_content.startswith("Image file"):
                    # Always add to cache (this creates/updates the DB row for text)
                    add_to_cache(thread_id, uploaded.filename, file_content, user_id)
                    record_file_usage(thread_id, uploaded.filename, 'chat', user_msg)
                    file_contents.append(file_content)

                    # For registered users, also save the original file to disk (3-day expiration)
                    if session.get('consent_value', 0) == 1:
                        uploaded.seek(0)
                        file_bytes = uploaded.read()
                        file_hash = hashlib.sha256(file_bytes).hexdigest()
                        ext = os.path.splitext(uploaded.filename)[1]
                        unique_name = f"{file_hash}_{int(time.time())}{ext}"
                        original_dir = os.path.join(USER_FILES_ORIGINAL_ROOT, user_id)
                        os.makedirs(original_dir, exist_ok=True)
                        original_path = os.path.join(original_dir, unique_name)
                        uploaded.seek(0)
                        uploaded.save(original_path)

                        # Update the existing user_files row with original file info
                        with get_db_connection() as conn:
                            with conn.cursor() as cur:
                                cur.execute("""
                                            UPDATE user_files
                                            SET original_stored_path = %s,
                                                original_expires_at  = NOW() + INTERVAL '3 days',
                                                file_hash            = %s
                                            WHERE thread_id = %s
                                              AND filename = %s
                                              AND user_id = %s
                                            """, (original_path, file_hash, thread_id, uploaded.filename, user_id))
                                conn.commit()
                    add_to_cache(thread_id, uploaded.filename, file_content, user_id)
                    record_file_usage(thread_id, uploaded.filename, 'chat', user_msg)
                    file_contents.append(file_content)
        if file_contents:
            combined = "\n\n".join(file_contents)
            if is_image:
                final_query = f"The user uploaded images. OCR/VL extracted the following text:\n{combined}\n\nUser query: {user_msg}"
            else:
                final_query = f"File content(s):\n{combined}\n\nUser query:\n{user_msg}"
        else:
            final_query = user_msg
        store_message(thread_id, 'user', user_msg)
        agent = get_agent()
        config = {"configurable": {"thread_id": thread_id}}
        try:
            response = agent.invoke({"messages": [{"role": "user", "content": final_query}]}, config)
        except Exception as e:
            logger.error(f"Agent invoke failed: {e}", exc_info=True)
            return jsonify({"error": "AI 服务暂时不可用"}), 500
        assistant_message = response["messages"][-1]
        raw_response = assistant_message.content
        reasoning = assistant_message.additional_kwargs.get('reasoning_content', '')
        if reasoning and reasoning.strip():
            thinking = reasoning.strip()
            answer = raw_response.strip() if raw_response else ''
        else:
            thinking, answer = split_thinking_answer(raw_response)
        store_message(thread_id, 'assistant', answer, thinking)
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                            UPDATE message_responses
                            SET assistant_response = %s,
                                thinking           = %s
                            WHERE message_id = %s
                            """, (answer, thinking, message_id))
                conn.commit()
        new_history = session.get('chat_history', [])
        new_history.append({"role": "user", "content": user_msg})
        new_history.append({"role": "assistant", "content": answer, "thinking": thinking})
        session['chat_history'] = new_history
        session['last_user_msg'] = user_msg
        session['last_assistant_msg'] = answer
        is_batch_report = answer.startswith('<!--COMPARE_REPORT-->') if answer else False
        return jsonify({
            "assistant_message": answer,
            "thinking": thinking,
            "file_processed": len(uploaded_filenames) > 0,
            "ocr_attempted": is_image,
            "is_batch_report": is_batch_report
        })
    finally:
        if has_files:
            release_task_lock(user_id)


@app.route('/set_max_tokens', methods=['POST'])
def set_max_tokens():
    data = request.get_json()
    tokens = data.get('max_tokens', 3200)
    tokens = max(100, min(3200, tokens))
    session['max_tokens'] = tokens
    global _agent
    with _agent_lock:
        _agent = None
    return jsonify({"success": True, "max_tokens": tokens})

@app.route('/check_auth', methods=['GET'])
def check_auth():
    if not session.get('consent_given'):
        return jsonify({"authenticated": False, "reason": "consent_not_given"})
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"authenticated": False, "reason": "no_user_id"})
    # Optionally verify user still exists in DB
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM users WHERE user_id = %s", (user_id,))
            if not cur.fetchone():
                return jsonify({"authenticated": False, "reason": "user_deleted"})
    return jsonify({
        "authenticated": True,
        "username": session.get('username'),
        "is_admin": session.get('role') == 'admin',
        "user_id": user_id
    })

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    if session.get('consent_value') != 1:
        return jsonify({"error": "Feedback not allowed – no consent"}), 403
    data = request.get_json()
    rating = data.get('rating')
    comment = data.get('comment', '')
    user_message = data.get('user_message')
    assistant_response = data.get('assistant_response')
    if not user_message or not assistant_response:
        user_message = session.get('last_user_msg', '')
        assistant_response = session.get('last_assistant_msg', '')
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (thread_id, user_message, assistant_response, rating, comment, timestamp) VALUES (%s, %s, %s, %s, %s, %s)",
                (session['thread_id'], user_message, assistant_response, rating, comment, utc_now())
            )
            conn.commit()
    return jsonify({"status": "ok"})


@app.route('/get_recent_files', methods=['GET'])
def get_recent_files():
    thread_id = session.get('thread_id')
    if not thread_id:
        return jsonify({"recent_files": []})
    recent = file_cache_manager.get_recent_with_lock(thread_id)
    files_with_usage = []
    if session.get('consent_value', 0) == 1:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for filename in recent:
                    cur.execute(
                        """SELECT usage_type, question, timestamp
                           FROM file_usage
                           WHERE thread_id = %s
                             AND filename = %s
                           ORDER BY timestamp DESC
                           LIMIT 5""",
                        (thread_id, filename)
                    )
                    usage_records = []
                    for row in cur.fetchall():
                        ts_utc = row['timestamp']
                        if ts_utc:
                            ts_beijing = ts_utc.astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            ts_beijing = None
                        usage_records.append({
                            "type": row['usage_type'],
                            "question": row['question'],
                            "time": ts_beijing
                        })
                    files_with_usage.append({
                        "filename": filename,
                        "usage": usage_records
                    })
    else:
        for filename in recent:
            files_with_usage.append({"filename": filename, "usage": []})
    return jsonify({"recent_files": files_with_usage})


@app.route('/load_cached_file', methods=['POST'])
def load_cached_file():
    data = request.get_json()
    filename = data.get('filename')
    thread_id = session.get('thread_id')
    if not thread_id:
        return jsonify({"error": "Session expired"}), 401
    content = file_cache_manager.get_content(thread_id, filename)
    if content:
        return jsonify({"content": content})
    if session.get('consent_value', 0) != 1:
        user_id = get_user_id()
        temp_dir = get_anon_temp_dir(user_id)
        safe_name = re.sub(r'[^\w\-_\. ]', '_', filename) + '.txt'
        fpath = os.path.join(temp_dir, safe_name)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            add_to_cache(thread_id, filename, content, user_id)
            return jsonify({"content": content})
        else:
            return jsonify({"error": "File not found"}), 404
    user_id = get_user_id()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content FROM user_files WHERE user_id = %s AND filename = %s AND (expires_at IS NULL OR expires_at > NOW())",
                (user_id, filename)
            )
            row = cur.fetchone()
            if row:
                content = row[0]
                add_to_cache(thread_id, filename, content, user_id)
                return jsonify({"content": content})
    return jsonify({"error": "File not found"}), 404


@app.route('/new_chat', methods=['POST'])
def new_chat():
    new_thread_id = str(uuid.uuid4())
    session['thread_id'] = new_thread_id
    session['chat_history'] = []
    get_or_create_session(new_thread_id)
    return jsonify({"thread_id": new_thread_id})


@app.route('/get_sessions', methods=['GET'])
def get_sessions():
    sessions = get_user_sessions()
    return jsonify({"sessions": sessions})


@app.route('/load_session/<thread_id>', methods=['GET'])
def load_session(thread_id):
    if session.get('consent_value', 0) != 1:
        messages = get_session_messages_anon(thread_id)
        session['thread_id'] = thread_id
        session['chat_history'] = messages
        return jsonify({"messages": messages, "thread_id": thread_id})
    user_sessions = get_user_sessions()
    if not any(s['thread_id'] == thread_id for s in user_sessions):
        return jsonify({"error": "Session not found"}), 404
    messages = get_session_messages(thread_id)
    session['thread_id'] = thread_id
    session['chat_history'] = messages
    user_id = get_user_id()
    load_cache_from_db(thread_id, user_id)
    return jsonify({"messages": messages, "thread_id": thread_id})


@app.route('/delete_session/<thread_id>', methods=['POST'])
def delete_session_route(thread_id):
    user_sessions = get_user_sessions()
    if not any(s['thread_id'] == thread_id for s in user_sessions):
        return jsonify({"error": "Session not found"}), 404
    user_id = get_user_id()
    with user_task_lock:
        cleanup_stale_tasks()
        if user_id in user_active_tasks and user_active_tasks[user_id]['thread_id'] == thread_id:
            return jsonify({
                "error": "task_running",
                "message": "无法删除：该聊天正在进行资源密集型任务，请等待任务完成后再试。"
            }), 409
    try:
        archive_session(thread_id, user_id, reason="manual")
    except Exception as e:
        logger.error(f"Archive session failed for {thread_id}: {e}", exc_info=True)
    try:
        delete_session(thread_id)
        logger.info(f"Session {thread_id} deleted successfully for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to delete session {thread_id}: {e}", exc_info=True)
        return jsonify({"error": "删除失败，请稍后重试"}), 500
    new_thread_id = None
    if session.get('thread_id') == thread_id:
        new_thread_id = str(uuid.uuid4())
        session['thread_id'] = new_thread_id
        session['chat_history'] = []
        get_or_create_session(new_thread_id)
        load_cache_from_db(new_thread_id, get_user_id())
    return jsonify({
        "status": "ok",
        "new_thread_id": new_thread_id,
        "messages": []
    })


@app.route('/regenerate', methods=['POST'])
def regenerate():
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    data = request.get_json()
    user_message = data.get('user_message')
    if not user_message:
        return jsonify({"error": "Missing user_message"}), 400
    thread_id = session['thread_id']
    get_or_create_session(thread_id)
    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        response = agent.invoke({"messages": [{"role": "user", "content": user_message}]}, config)
    except Exception as e:
        logger.error(f"Regenerate invoke failed: {e}", exc_info=True)
        return jsonify({"error": "AI 服务暂时不可用"}), 500
    assistant_message = response["messages"][-1]
    raw_response = assistant_message.content
    reasoning = assistant_message.additional_kwargs.get('reasoning_content', '')
    if reasoning and reasoning.strip():
        thinking = reasoning.strip()
        answer = raw_response.strip() if raw_response else ''
    else:
        thinking, answer = split_thinking_answer(raw_response)
    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                cur.execute("""
                            DELETE
                            FROM chat_messages
                            WHERE id IN (SELECT id
                                         FROM chat_messages
                                         WHERE thread_id = %s
                                         ORDER BY timestamp DESC
                                         LIMIT 2)
                            """, (thread_id,))
                conn.commit()
    store_message(thread_id, 'user', user_message)
    store_message(thread_id, 'assistant', answer if answer else raw_response, thinking if thinking else "")
    new_messages = get_session_messages(thread_id)
    session['chat_history'] = new_messages
    return jsonify({
        "assistant_message": answer if answer else raw_response,
        "thinking": thinking if thinking else ""
    })


# Batch comparison helper functions
def _precompute_tfidf_for_files(file_data, template_text=None):
    texts = []
    for fd in file_data:
        clean = preprocess_text_for_similarity(fd['text'])
        if template_text:
            clean = remove_template_content(clean, template_text)
        texts.append(clean)
    vectorizer = TfidfVectorizer(stop_words=None, lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


def _compute_pair_similarity_from_matrix(tfidf_matrix, i, j):
    sim = cosine_similarity(tfidf_matrix[i:i + 1], tfidf_matrix[j:j + 1])[0][0]
    return sim


# Download token management
download_tokens = defaultdict(int)


@app.route('/compare_batch', methods=['POST'])
def compare_batch():
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    files = request.files.getlist('files')
    if len(files) < 2:
        return jsonify({"error": "Need at least 2 files for comparison"}), 400
    if len(files) > 10:
        return jsonify({"error": "Maximum 10 files allowed"}), 400
    for f in files:
        if not allowed_file(f.filename):
            return jsonify({"error": f"不支持的文件类型: {f.filename}"}), 400
    user_id = get_user_id()
    thread_id = session['thread_id']
    success, busy_thread, busy_name = acquire_task_lock(user_id, thread_id, 'batch_compare')
    if not success:
        return jsonify({
            "error": "resource_busy",
            "busy_chat": busy_name,
            "message": f"另一个资源密集型任务正在聊天“{busy_name}”中进行，请稍后再试。"
        }), 409
    try:
        template_file = request.files.get('template_file')
        template_text = None
        if template_file and template_file.filename:
            if not allowed_file(template_file.filename):
                return jsonify({"error": f"不支持的文件类型: {template_file.filename}"}), 400
            template_text, _ = extract_text_from_file(template_file)
            if template_text and not template_text.startswith("["):
                add_to_cache(thread_id, template_file.filename, template_text, user_id)
                record_file_usage(thread_id, template_file.filename, 'template_upload', "上传模板文件用于对比")
        check_items_json = request.form.get('check_items', '{}')
        try:
            check_items = json.loads(check_items_json)
        except:
            check_items = {}
        for k in ['text_sim', 'key_info', 'file_attr', 'image_sim']:
            if k not in check_items:
                check_items[k] = True
        file_data = []
        for f in files:
            if not f.filename:
                continue
            text, _ = extract_text_from_file(f)
            if text and not text.startswith("["):
                # For registered users, also save the original file to disk (3-day expiration)
                if session.get('consent_value', 0) == 1:
                    f.seek(0)
                    file_bytes = f.read()
                    file_hash = hashlib.sha256(file_bytes).hexdigest()
                    ext = os.path.splitext(f.filename)[1]
                    unique_name = f"{file_hash}_{int(time.time())}{ext}"
                    original_dir = os.path.join(USER_FILES_ORIGINAL_ROOT, user_id)
                    os.makedirs(original_dir, exist_ok=True)
                    original_path = os.path.join(original_dir, unique_name)
                    f.seek(0)
                    f.save(original_path)
                    original_size = os.path.getsize(original_path)

                    # Update or insert the user_files record with original path
                    with get_db_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                        INSERT INTO user_files (user_id, thread_id, filename, content, size_bytes,
                                                                expires_at, original_stored_path, file_hash,
                                                                original_expires_at)
                                        VALUES (%s, %s, %s, %s, %s, NULL, %s, %s, NOW() + INTERVAL '3 days')
                                        ON CONFLICT (thread_id, filename) DO UPDATE SET content              = EXCLUDED.content,
                                                                                        size_bytes           = EXCLUDED.size_bytes,
                                                                                        original_stored_path = EXCLUDED.original_stored_path,
                                                                                        file_hash            = EXCLUDED.file_hash,
                                                                                        original_expires_at  = EXCLUDED.original_expires_at
                                        """, (user_id, thread_id, f.filename, text, len(text),
                                              original_path, file_hash))
                            conn.commit()
                else:
                    # Anonymous: only cache in memory, no DB
                    add_to_cache(thread_id, f.filename, text, user_id)

                record_file_usage(thread_id, f.filename, 'compare_batch', "批量对比")
                f.seek(0)
                meta = extract_metadata(f)
                images = extract_images_from_file(f)
                file_data.append({
                    'filename': f.filename,
                    'text': text,
                    'metadata': meta,
                    'images': images
                })
            else:
                continue
        if len(file_data) < 2:
            return jsonify({"error": "Could not extract text from at least two files"}), 400
        n = len(file_data)
        if check_items.get('text_sim', True) or check_items.get('key_info', True):
            vectorizer, tfidf_matrix = _precompute_tfidf_for_files(file_data, template_text)
        else:
            vectorizer = tfidf_matrix = None
        pairs = []
        risk_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                text1 = file_data[i]['text']
                text2 = file_data[j]['text']
                meta1 = file_data[i]['metadata']
                meta2 = file_data[j]['metadata']
                images1 = file_data[i]['images']
                images2 = file_data[j]['images']
                img_sim = image_similarity(images1, images2) if check_items.get('image_sim', True) else 0.0
                if check_items.get('text_sim', True) and tfidf_matrix is not None:
                    sim = _compute_pair_similarity_from_matrix(tfidf_matrix, i, j)
                else:
                    sim = 0.0
                if check_items.get('key_info', True):
                    t1 = preprocess_text_for_similarity(text1)
                    t2 = preprocess_text_for_similarity(text2)
                    if template_text:
                        t1 = remove_template_content(t1, template_text)
                        t2 = remove_template_content(t2, template_text)
                    key_sim = keyword_overlap_similarity(t1, t2)
                else:
                    key_sim = 0.0
                values = {}
                used = {}
                total_weight = 0
                if check_items.get('text_sim', True):
                    values['text_sim'] = sim * 100
                    used['text_sim'] = 0.2
                    total_weight += 0.2
                if check_items.get('key_info', True):
                    values['key_info'] = key_sim * 100
                    used['key_info'] = 0.3
                    total_weight += 0.3
                if check_items.get('file_attr', True) and meta1 and meta2:
                    values['file_attr'] = file_attr_similarity(meta1, meta2)
                    used['file_attr'] = 0.3
                    total_weight += 0.3
                if check_items.get('image_sim', True):
                    values['image_sim'] = img_sim
                    used['image_sim'] = 0.2
                    total_weight += 0.2
                if total_weight > 0:
                    for k in used:
                        used[k] = used[k] / total_weight
                    risk = sum(values.get(k, 0) * used[k] for k in used)
                else:
                    risk = 0.0
                _, html1, html2, blocks = compute_similarity_with_numbers(text1, text2, template_text)
                pair_info = {
                    'i': i, 'j': j,
                    'name1': file_data[i]['filename'],
                    'name2': file_data[j]['filename'],
                    'sim': sim * 100,
                    'risk': risk,
                    'blocks': blocks,
                    'html1': html1,
                    'html2': html2,
                    'used_weights': used,
                    'attr_same': 1 if meta1.get('author') and meta1['author'] == meta2.get('author') else 0
                }
                pairs.append(pair_info)
                risk_matrix[i][j] = risk
                risk_matrix[j][i] = risk
        batch_data = {
            'file_data': [(fd['filename'], fd['metadata']) for fd in file_data],
            'pairs': pairs,
            'check_items': check_items,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        temp_path = store_batch_comparison_temp(batch_data)
        session['batch_comparison_path'] = temp_path
        high_risk_files = []
        strong_alert_files = []
        for i in range(n):
            for j in range(i + 1, n):
                if risk_matrix[i][j] > 20:
                    strong_alert_files.extend([file_data[i]['filename'], file_data[j]['filename']])
                elif risk_matrix[i][j] > 10:
                    high_risk_files.extend([file_data[i]['filename'], file_data[j]['filename']])
        strong_alert_files = list(set(strong_alert_files))
        high_risk_files = list(set(high_risk_files) - set(strong_alert_files))
        short_names = [truncate_filename(fd['filename'], 20) for fd in file_data]
        # Summary HTML
        summary_html = '<details style="margin-bottom:4px; border-radius:6px; padding:6px;"><summary style="cursor:pointer; font-weight:bold; font-size:0.9rem;">📋 对比摘要 (点击展开)</summary><div style="margin-top:12px; border-left:8px solid #2c3e50; padding-left:8px;">'
        for fd in file_data:
            preview = fd['text'][:200].replace('\n', ' ') + '…'
            summary_html += f'<div style="margin-bottom:15px;"><strong>📄 {fd["filename"]}</strong><br><span style="color:#666; font-size:0.85rem;">{preview}</span></div>'
        if strong_alert_files:
            summary_html += f'<p style="color:#d9534f; font-weight:bold;">🚨 强烈警告：以下文件风险度超过20：{", ".join(strong_alert_files)}</p>'
        elif high_risk_files:
            summary_html += f'<p style="color:#f0ad4e; font-weight:bold;">⚠️ 可疑文件：以下文件风险度超过10：{", ".join(high_risk_files)}</p>'
        else:
            summary_html += '<p style="color:#5cb85c;">✅ 未发现高风险文件（风险度均≤10）</p>'
        summary_html += '</div></details>'
        # Main report
        if n == 2:
            p = pairs[0]
            if p['blocks']:
                detail_rows = ""
                for b in p['blocks']:
                    detail_rows += f'<tr><td style="border:1px solid #ccc; padding:8px; text-align:center;">{b["id"]}</td><td style="border:1px solid #ccc; padding:8px; text-align:center;">{b["size"]}</td><td style="border:1px solid #ccc; padding:8px; word-break:break-word; max-width:300px;">{b["text1_snippet"]}</td><td style="border:1px solid #ccc; padding:8px; word-break:break-word; max-width:300px;">{b["text2_snippet"]}</td></tr>'
                detailed_report = f'<details><summary style="cursor:pointer; font-weight:bold;">📋 详细相似度明细报告（共 {len(p["blocks"])} 个匹配块）</summary><div style="margin-top:12px;"><p><strong>总匹配字符数：</strong>{sum(b["size"] for b in p["blocks"])} 字符 &nbsp;|&nbsp;<strong>平均匹配块长度：</strong>{round(sum(b["size"] for b in p["blocks"]) / len(p["blocks"]), 1)} 字符</p><div style="overflow-x:auto;"><table style="width:100%; border-collapse:collapse; margin-top:10px;"><thead><tr style="background:#f0f0f0;"><th style="border:1px solid #ccc; padding:8px;">块序号</th><th style="border:1px solid #ccc; padding:8px;">匹配字符数</th><th style="border:1px solid #ccc; padding:8px;">文档A片段</th><th style="border:1px solid #ccc; padding:8px;">文档B片段</th></tr></thead><tbody>{detail_rows}</tbody></table></div></div></details>'
                main_report = detailed_report
            else:
                main_report = "<p>未检测到显著匹配块。</p>"
        else:
            matrix_html = '<details><summary style="cursor:pointer; font-weight:bold;">📊 风险度矩阵 (点击展开/折叠)</summary><div style="overflow-x:auto; margin-top:12px;"><table style="border-collapse:collapse; font-size:0.85rem; min-width:400px; width:100%;"><thead><tr><th style="padding:8px; border:1px solid #ddd;"></th>' + ''.join(
                f'<th style="padding:8px; border:1px solid #ddd; word-break:break-word;">{short_names[i]}</th>'
                for i in range(n)) + '</tr></thead><tbody>'
            for i in range(n):
                matrix_html += f'<tr><td style="border:1px solid #ddd; padding:8px; font-weight:bold;">{short_names[i]}</td>'
                for j in range(n):
                    if i == j:
                        val = '--'
                        bg = ''
                    else:
                        val = f'{risk_matrix[i][j]:.2f}'
                        if risk_matrix[i][j] > 20:
                            bg = ' style="background:#d9534f; color:white; font-weight:bold;"'
                        elif risk_matrix[i][j] > 10:
                            bg = ' style="background:#f0ad4e;"'
                        else:
                            bg = ''
                    matrix_html += f'<td style="border:1px solid #ddd; padding:8px; text-align:center;"{bg}>{val}</td>'
                matrix_html += '</tr>'
            matrix_html += '</tbody></table></div><p style="font-size:0.7rem; color:#666; margin-top:8px;">风险度矩阵（值越高风险越大）</p></details>'
            main_report = matrix_html
        # Generate download token (20 uses)
        download_token = secrets.token_urlsafe(32)
        download_tokens[download_token] = 20
        session[f'download_path_{download_token}'] = temp_path
        download_link = url_for('export_batch_excel_download', token=download_token, _external=True)
        export_html = f'<p><a href="{download_link}" target="_blank" style="background:#27ae60; color:white; text-decoration:none; border-radius:8px; padding:8px 16px; display:inline-block; margin-top:12px;">📊 下载Excel报告 (可下载20次)</a></p>'
        full_message = f"<!--COMPARE_REPORT--><div style='font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, sans-serif; line-height:1.5; max-width:100%; overflow-x:auto;'><h4>📁 批量对比结果（{len(file_data)}个文件）</h4>{summary_html}{main_report}{export_html}</div>"
        store_message(thread_id, 'assistant', full_message, thinking="")
        session['chat_history'].append({
            "role": "assistant",
            "content": full_message,
            "thinking": ""
        })
        return jsonify({"success": True, "pair_count": len(pairs)})
    finally:
        release_task_lock(user_id)


@app.route('/export_batch_excel_download/<token>', methods=['GET'])
def export_batch_excel_download(token):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    remaining = download_tokens.get(token, 0)
    if remaining <= 0:
        return jsonify({"error": "Download link has expired or already used the maximum number of times."}), 410
    temp_path = session.get(f'download_path_{token}')
    if not temp_path or not os.path.exists(temp_path):
        return jsonify({"error": "Comparison data not found."}), 404
    try:
        batch_data = load_batch_comparison_temp(temp_path)
    except Exception as e:
        logger.error(f"Failed to load batch data: {e}")
        return jsonify({"error": "Comparison data corrupted."}), 400
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment
    from openpyxl.utils import get_column_letter
    file_data = batch_data['file_data']
    pairs = batch_data['pairs']
    timestamp = batch_data['timestamp']
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "规律性分析结果"
    ws1['A1'] = "技术标规律性分析检查结果"
    ws1['A1'].font = Font(bold=True, size=14)
    ws1.merge_cells('A1:H1')
    ws1['A2'] = "标段名称：用户自定义"
    ws1['A3'] = f"投标单位个数：{len(file_data)}"
    ws1['A4'] = f"创建时间：{timestamp}"
    max_risk = max(p['risk'] for p in pairs) if pairs else 0
    ws1['A5'] = f"检查结果：风险度最高{max_risk:.2f}；文本相似度最高{max(p['sim'] for p in pairs):.2f}%"
    ws1[
        'A6'] = "检查规则：检查相似度≥80%的段落，文本中重点信息，相似图片，相同作者；忽略与招标文件相同内容，忽略标点符号及小于6个字的内容，忽略目录，忽略文件中的技术标准，忽略【公司/组织、地名/地址、项目、人员、奖项、身份证号码、电话号码、统一社会信用代码、证书编号】"
    ws1['A7'] = "相似度计算说明：风险度=0.3×重点信息雷同风险+0.3×文件属性雷同风险+0.2×文本相似度×100+0.2×图片相似度×100\n*若某项不参与检查，则其余项按照比例进行折算"
    ws1.merge_cells('A6:H6')
    ws1.merge_cells('A7:H7')
    row = 10
    ws1[f'A{row}'] = "一、标书围串风险分析结果"
    ws1[f'A{row}'].font = Font(bold=True)
    row += 1
    headers = ["投标单位"] + [name for name, _ in file_data]
    for col, h in enumerate(headers, 1):
        cell = ws1.cell(row=row, column=col, value=h)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')
    row += 1
    for i in range(len(file_data)):
        ws1.cell(row=row, column=1, value=file_data[i][0])
        for j in range(len(file_data)):
            if i == j:
                val = "--"
            else:
                for p in pairs:
                    if (p['i'] == i and p['j'] == j) or (p['i'] == j and p['j'] == i):
                        val = p['risk']
                        break
                else:
                    val = 0
            ws1.cell(row=row, column=j + 2, value=val)
        row += 1
    row += 2
    ws1[f'A{row}'] = "二、分析结果详情"
    ws1[f'A{row}'].font = Font(bold=True)
    row += 1
    detail_headers = ["序号", "投标单位1", "投标单位2", "风险度", "文本相似度（%）", "图片相似度（%）", "文件属性雷同",
                      "重点信息雷同（项）"]
    for col, h in enumerate(detail_headers, 1):
        cell = ws1.cell(row=row, column=col, value=h)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')
    row += 1
    for idx, p in enumerate(pairs, 1):
        ws1.cell(row=row, column=1, value=idx)
        ws1.cell(row=row, column=2, value=p['name1'])
        ws1.cell(row=row, column=3, value=p['name2'])
        ws1.cell(row=row, column=4, value=p['risk'])
        ws1.cell(row=row, column=5, value=p['sim'])
        ws1.cell(row=row, column=6, value=0)
        ws1.cell(row=row, column=7, value="是" if p['attr_same'] else "否")
        ws1.cell(row=row, column=8, value=0)
        row += 1
    for col in range(1, 9):
        ws1.column_dimensions[get_column_letter(col)].width = 20
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    filename = f"清标分析结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    download_tokens[token] -= 1
    if download_tokens[token] <= 0:
        del download_tokens[token]
        session.pop(f'download_path_{token}', None)
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
    return send_file(output, as_attachment=True, download_name=filename,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@app.route('/export_batch_excel', methods=['POST'])
def export_batch_excel():
    # Legacy endpoint (kept for compatibility, but not used)
    return jsonify({"error": "Please use the download link from the batch comparison report."}), 400


@app.route('/check_storage', methods=['GET'])
def check_storage():
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = get_user_id()
    total_bytes = get_user_total_storage_size(user_id)
    total_mb = total_bytes / (1024 * 1024)
    warning = total_mb > 300
    return jsonify({
        "total_mb": round(total_mb, 2),
        "warning": warning,
        "message": f"已使用 {total_mb:.2f} MB / 300 MB" if warning else None
    })


@app.route('/cleanup_now', methods=['POST'])
def cleanup_now():
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    cleanup_old_sessions(days=15)
    return jsonify({"status": "ok", "message": "Cleanup completed"})


@app.route('/cleanup_anon_temp', methods=['POST'])
def cleanup_anon_temp():
    if session.get('consent_value', 0) != 1:
        user_id = get_user_id()
        temp_dir = get_anon_temp_dir(user_id)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up anonymous temp directory for user {user_id}")
    return jsonify({"status": "ok"})

# Account routes
@app.route('/create_account', methods=['POST'])
def create_account():
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403

    data = request.get_json()
    username = data.get('username', '').strip()
    pin = data.get('pin', '').strip()
    pin_length = data.get('pin_length', 6)

    if not username or not pin:
        return jsonify({"error": "用户名和PIN不能为空"}), 400
    if len(username) < 5 or len(username) > 18:
        return jsonify({"error": "用户名长度应为5-18个字符"}), 400
    if pin_length not in [4, 6] or len(pin) != pin_length:
        return jsonify({"error": f"PIN必须是{pin_length}位数字"}), 400
    if not pin.isdigit():
        return jsonify({"error": "PIN只能包含数字"}), 400

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Check if username already exists
            cur.execute("SELECT 1 FROM users WHERE username = %s", (username,))
            if cur.fetchone():
                return jsonify({"error": "用户名已存在"}), 409

            # Get existing user_id from session, or generate a new one
            user_id = session.get('user_id')
            if not user_id:
                user_id = str(uuid.uuid4())
                session['user_id'] = user_id

            pin_hash = generate_password_hash(pin)

            # Insert or update user (if user already existed as empty placeholder)
            cur.execute("""
                INSERT INTO users (user_id, username, pin_hash, pin_length, created_at, role)
                VALUES (%s, %s, %s, %s, NOW(), 'user')
                ON CONFLICT (user_id) DO UPDATE SET
                    username = EXCLUDED.username,
                    pin_hash = EXCLUDED.pin_hash,
                    pin_length = EXCLUDED.pin_length,
                    role = 'user'
                RETURNING user_id
            """, (user_id, username, pin_hash, pin_length))

            conn.commit()
            session['username'] = username
            session['role'] = 'user'
            session.modified = True

            return jsonify({"success": True, "username": username})

def cleanup_orphan_users():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Delete users with NULL/empty username, older than 1 day,
            # and with no chat sessions, files, or project memberships
            cur.execute("""
                DELETE FROM users
                WHERE (username IS NULL OR username = '')
                  AND created_at < NOW() - INTERVAL '1 day'
                  AND NOT EXISTS (SELECT 1 FROM chat_sessions WHERE user_id = users.user_id)
                  AND NOT EXISTS (SELECT 1 FROM user_files WHERE user_id = users.user_id)
                  AND NOT EXISTS (SELECT 1 FROM project_members WHERE user_id = users.user_id)
                  AND NOT EXISTS (SELECT 1 FROM projects WHERE created_by = users.user_id)
            """)
            conn.commit()
            logger.info(f"Deleted {cur.rowcount} orphan empty users")

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    pin = data.get('pin', '').strip()
    if not username or not pin:
        return jsonify({"error": "用户名和PIN不能为空"}), 400
    if username == "admin" and ADMIN_PASSWORD_HASH and check_password_hash(ADMIN_PASSWORD_HASH, pin):
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id FROM users WHERE username = 'admin'")
                admin_row = cur.fetchone()
                if admin_row:
                    user_id = admin_row[0]
                else:
                    user_id = str(uuid.uuid4())
                    cur.execute("INSERT INTO users (user_id, username, role) VALUES (%s, %s, %s)",
                                (user_id, 'admin', 'admin'))
                    conn.commit()
                session['user_id'] = user_id
                session['consent_given'] = True
                session['consent_value'] = 1
                session['username'] = 'admin'
                session['role'] = 'admin'
                session.permanent = True
                cur.execute(
                    "INSERT INTO consent (thread_id, consent_given, timestamp) VALUES (%s, %s, NOW()) ON CONFLICT (thread_id) DO UPDATE SET consent_given = EXCLUDED.consent_given, timestamp = EXCLUDED.timestamp",
                    (session.get('thread_id', str(uuid.uuid4())), 1)
                )
                conn.commit()
        return jsonify({"success": True, "username": "admin", "is_admin": True, "user_id": session['user_id']})
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT user_id, pin_hash, pin_length, role FROM users WHERE username = %s AND is_active = TRUE",
                (username,))
            user = cur.fetchone()
            if not user or not check_password_hash(user['pin_hash'], pin):
                return jsonify({"error": "用户名或PIN错误"}), 401
            session['user_id'] = user['user_id']
            session['consent_given'] = True
            session['consent_value'] = 1
            session['username'] = username
            session['role'] = user.get('role', 'user')
            session.permanent = True
            with conn.cursor() as cur2:
                cur2.execute(
                    "INSERT INTO consent (thread_id, consent_given, timestamp) VALUES (%s, %s, NOW()) ON CONFLICT (thread_id) DO UPDATE SET consent_given = EXCLUDED.consent_given, timestamp = EXCLUDED.timestamp",
                    (session.get('thread_id', str(uuid.uuid4())), 1)
                )
            conn.commit()
            return jsonify({"success": True, "username": username, "is_admin": session['role'] == 'admin',
                            "user_id": session['user_id']})


@app.route('/update_account', methods=['POST'])
def update_account():
    if not session.get('consent_given') or not session.get('user_id'):
        return jsonify({"error": "未登录"}), 401
    data = request.get_json()
    new_username = data.get('new_username', '').strip()
    new_pin = data.get('new_pin', '').strip()
    pin_length = data.get('pin_length', 6)
    current_pin = data.get('current_pin', '').strip()
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT pin_hash FROM users WHERE user_id = %s", (session['user_id'],))
            user = cur.fetchone()
            if not user or not check_password_hash(user['pin_hash'], current_pin):
                return jsonify({"error": "当前PIN错误"}), 401
            updates = []
            params = []
            if new_username:
                if len(new_username) < 5 or len(new_username) > 18:
                    return jsonify({"error": "用户名长度应为5-18个字符"}), 400
                cur.execute("SELECT 1 FROM users WHERE username = %s AND user_id != %s",
                            (new_username, session['user_id']))
                if cur.fetchone():
                    return jsonify({"error": "用户名已存在"}), 409
                updates.append("username = %s")
                params.append(new_username)
                session['username'] = new_username
            if new_pin:
                if pin_length not in [4, 6] or len(new_pin) != pin_length:
                    return jsonify({"error": f"PIN必须是{pin_length}位数字"}), 400
                if not new_pin.isdigit():
                    return jsonify({"error": "PIN只能包含数字"}), 400
                updates.append("pin_hash = %s")
                params.append(generate_password_hash(new_pin))
                updates.append("pin_length = %s")
                params.append(pin_length)
            if updates:
                params.append(session['user_id'])
                cur.execute(f"UPDATE users SET {', '.join(updates)} WHERE user_id = %s", params)
                conn.commit()
            return jsonify({"success": True})


@app.route('/delete_account', methods=['POST'])
def delete_account():
    if not session.get('consent_given') or not session.get('user_id'):
        return jsonify({"error": "未登录"}), 401

    data = request.get_json()
    pin = data.get('pin', '').strip()
    user_id = session['user_id']

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Verify PIN
            cur.execute("SELECT pin_hash FROM users WHERE user_id = %s", (user_id,))
            user = cur.fetchone()
            if not user or not check_password_hash(user['pin_hash'], pin):
                return jsonify({"error": "PIN错误"}), 401

            with db_transaction(conn):
                # ----- Move project-related data to task deposit (optional) -----
                # Get user's projects where they are a member or creator
                cur.execute("""
                    SELECT DISTINCT p.id, p.name, p.created_by
                    FROM projects p
                    LEFT JOIN project_members pm ON p.id = pm.project_id
                    WHERE pm.user_id = %s OR p.created_by = %s
                """, (user_id, user_id))
                projects = cur.fetchall()
                for proj in projects:
                    proj_id = proj['id']
                    proj_name = proj['name']
                    # Store project info in task_deposit_items
                    cur.execute("""
                        INSERT INTO task_deposit_items (original_user_id, original_username, project_id,
                                                        project_name, item_type, item_data)
                        VALUES (%s, %s, %s, %s, 'project', %s)
                    """, (user_id, user.get('username', 'unknown'), proj_id, proj_name,
                          json.dumps({'project_id': proj_id, 'name': proj_name})))
                    # Store files
                    cur.execute("""
                        SELECT id, original_name, stored_path, uploaded_by, folder_id, filename, version, file_size
                        FROM project_files WHERE project_id = %s
                    """, (proj_id,))
                    files = cur.fetchall()
                    for f in files:
                        cur.execute("""
                            INSERT INTO task_deposit_items (original_user_id, original_username, project_id,
                                                            project_name, item_type, item_data, stored_path)
                            VALUES (%s, %s, %s, %s, 'file', %s, %s)
                        """, (user_id, user.get('username', 'unknown'), proj_id, proj_name,
                              json.dumps(dict(f)), f['stored_path']))
                    # Store folders
                    cur.execute("""
                        SELECT id, name, parent_folder_id, created_by
                        FROM project_folders WHERE project_id = %s
                    """, (proj_id,))
                    folders = cur.fetchall()
                    for fold in folders:
                        cur.execute("""
                            INSERT INTO task_deposit_items (original_user_id, original_username, project_id,
                                                            project_name, item_type, item_data)
                            VALUES (%s, %s, %s, %s, 'folder', %s)
                        """, (user_id, user.get('username', 'unknown'), proj_id, proj_name,
                              json.dumps(dict(fold))))
                    # Store comments
                    cur.execute("""
                        SELECT id, file_id, user_id, comment, created_at
                        FROM project_file_comments
                        WHERE file_id IN (SELECT id FROM project_files WHERE project_id = %s)
                    """, (proj_id,))
                    comments = cur.fetchall()
                    for comm in comments:
                        cur.execute("""
                            INSERT INTO task_deposit_items (original_user_id, original_username, project_id,
                                                            project_name, item_type, item_data)
                            VALUES (%s, %s, %s, %s, 'comment', %s)
                        """, (user_id, user.get('username', 'unknown'), proj_id, proj_name,
                              json.dumps(dict(comm))))

                # ----- Reassign or set NULL all foreign key references -----
                # Find a global admin to reassign project ownership (if any)
                cur.execute("SELECT user_id FROM users WHERE role = 'admin' AND user_id != %s LIMIT 1", (user_id,))
                admin_row = cur.fetchone()
                admin_id = admin_row['user_id'] if admin_row else None

                # Projects created_by
                if admin_id:
                    cur.execute("UPDATE projects SET created_by = %s WHERE created_by = %s", (admin_id, user_id))
                else:
                    # No other admin – set to NULL (requires column to allow NULL)
                    cur.execute("UPDATE projects SET created_by = NULL WHERE created_by = %s", (user_id,))

                # Project folders created_by
                if admin_id:
                    cur.execute("UPDATE project_folders SET created_by = %s WHERE created_by = %s", (admin_id, user_id))
                else:
                    cur.execute("UPDATE project_folders SET created_by = NULL WHERE created_by = %s", (user_id,))

                # Project files uploaded_by
                if admin_id:
                    cur.execute("UPDATE project_files SET uploaded_by = %s WHERE uploaded_by = %s", (admin_id, user_id))
                else:
                    cur.execute("UPDATE project_files SET uploaded_by = NULL WHERE uploaded_by = %s", (user_id,))

                # Project file versions uploaded_by
                if admin_id:
                    cur.execute("""
                        UPDATE project_file_versions SET uploaded_by = %s
                        WHERE uploaded_by = %s
                    """, (admin_id, user_id))
                else:
                    cur.execute("UPDATE project_file_versions SET uploaded_by = NULL WHERE uploaded_by = %s", (user_id,))

                # Project members added_by
                if admin_id:
                    cur.execute("UPDATE project_members SET added_by = %s WHERE added_by = %s", (admin_id, user_id))
                else:
                    cur.execute("UPDATE project_members SET added_by = NULL WHERE added_by = %s", (user_id,))

                # Project file comments user_id (set to NULL, but keep comment)
                cur.execute("UPDATE project_file_comments SET user_id = NULL WHERE user_id = %s", (user_id,))
                # Project folder comments user_id
                cur.execute("UPDATE project_folder_comments SET user_id = NULL WHERE user_id = %s", (user_id,))
                # Task deposit transferred_to_user_id (set to NULL)
                cur.execute("UPDATE task_deposit_items SET transferred_to_user_id = NULL WHERE transferred_to_user_id = %s", (user_id,))
                # Clean up recycle_bin
                cur.execute("DELETE FROM recycle_bin WHERE user_id = %s", (user_id,))
                # Clean up project_recycle_bin (set uploaded_by to NULL)
                cur.execute("UPDATE project_recycle_bin SET uploaded_by = NULL WHERE uploaded_by = %s", (user_id,))
                # Clean up task_deposit_items (set original_user_id to NULL)
                cur.execute("UPDATE task_deposit_items SET original_user_id = NULL WHERE original_user_id = %s",
                            (user_id,))
                # Clean up task_deposit_permissions
                cur.execute("UPDATE task_deposit_permissions SET manager_id = NULL WHERE manager_id = %s", (user_id,))
                cur.execute("UPDATE task_deposit_permissions SET granted_by = NULL WHERE granted_by = %s", (user_id,))
                # ----- Delete user's memberships and chat data -----
                cur.execute("DELETE FROM project_members WHERE user_id = %s", (user_id,))
                cur.execute("DELETE FROM chat_messages WHERE thread_id IN (SELECT thread_id FROM chat_sessions WHERE user_id = %s)", (user_id,))
                cur.execute("DELETE FROM user_files WHERE user_id = %s", (user_id,))
                cur.execute("DELETE FROM file_usage WHERE user_id = %s", (user_id,))
                cur.execute("DELETE FROM feedback WHERE thread_id IN (SELECT thread_id FROM chat_sessions WHERE user_id = %s)", (user_id,))
                cur.execute("DELETE FROM consent WHERE thread_id IN (SELECT thread_id FROM chat_sessions WHERE user_id = %s)", (user_id,))
                cur.execute("DELETE FROM chat_sessions WHERE user_id = %s", (user_id,))

                # ----- Finally delete the user -----
                cur.execute("DELETE FROM users WHERE user_id = %s", (user_id,))

                conn.commit()

            # Clear session and redirect to home
            session.clear()
            session['consent_given'] = False
            session['consent_value'] = 0
            session['thread_id'] = str(uuid.uuid4())
            get_or_create_session(session['thread_id'])
            return jsonify({"success": True})

# Task deposit endpoints
@app.route('/admin/task_deposit', methods=['GET'])
def get_task_deposit():
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    is_admin_user = session.get('role') == 'admin'
    if not is_admin_user:
        # Check if manager has permission to view deposit (optional)
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT id,
                               original_user_id,
                               original_username,
                               project_id,
                               project_name,
                               item_type,
                               item_data,
                               stored_path,
                               transferred_to_user_id,
                               transferred_at,
                               created_at
                        FROM task_deposit_items
                        WHERE deleted_at IS NULL
                        ORDER BY created_at DESC
                        """)
            items = cur.fetchall()
            return jsonify({"items": items})


@app.route('/admin/task_deposit/transfer/<int:item_id>', methods=['POST'])
def transfer_task_deposit_item(item_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if session.get('role') != 'admin':
        return jsonify({"error": "Only admin can transfer deposit items"}), 403
    data = request.get_json()
    target_user_id = data.get('target_user_id')
    if not target_user_id:
        return jsonify({"error": "Missing target_user_id"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM users WHERE user_id = %s", (target_user_id,))
            if not cur.fetchone():
                return jsonify({"error": "Target user not found"}), 404
            cur.execute("""
                        UPDATE task_deposit_items
                        SET transferred_to_user_id = %s,
                            transferred_at         = NOW()
                        WHERE id = %s
                          AND deleted_at IS NULL
                        RETURNING id, item_type, item_data, stored_path
                        """, (target_user_id, item_id))
            item = cur.fetchone()
            if not item:
                return jsonify({"error": "Item not found or already deleted"}), 404
            # Optionally restore the item to the target user's projects
            # For simplicity, we just mark as transferred; actual restoration logic can be added.
            conn.commit()
            return jsonify({"success": True, "item": dict(item)})


# Permission helpers for projects
def is_admin():
    return session.get('role') == 'admin'

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_admin():
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated_function


def get_user_role_in_project(project_id, user_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT role FROM project_members WHERE project_id = %s AND user_id = %s",
                        (project_id, user_id))
            row = cur.fetchone()
            return row[0] if row else None


def can_manage_files(project_id, user_id):
    if is_admin():
        return True
    role = get_user_role_in_project(project_id, user_id)
    return role in ('admin', 'manager')


def can_edit_file(project_id, file_id, user_id):
    if is_admin():
        return True
    role = get_user_role_in_project(project_id, user_id)
    if role == 'manager':
        return True
    if role == 'member':
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT uploaded_by FROM project_files WHERE id = %s AND project_id = %s",
                            (file_id, project_id))
                row = cur.fetchone()
                return row and row[0] == user_id
    return False

def can_move_file(project_id, file_id, user_id):
    """All project members can move any file (no ownership restriction)."""
    if is_admin():
        return True
    role = get_user_role_in_project(project_id, user_id)
    if role in ('admin', 'manager', 'member'):
        return True
    return False

def can_edit_folder(project_id, folder_id, user_id):
    if is_admin():
        return True
    role = get_user_role_in_project(project_id, user_id)
    if role == 'manager':
        return True
    if role == 'member':
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT created_by FROM project_folders WHERE id = %s AND project_id = %s",
                            (folder_id, project_id))
                row = cur.fetchone()
                return row and row[0] == user_id
    return False


def can_manage_members(project_id, user_id):
    if is_admin():
        return True
    role = get_user_role_in_project(project_id, user_id)
    return role == 'manager'


def can_access_project(project_id, user_id):
    if is_admin():
        return True
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM project_members WHERE project_id = %s AND user_id = %s", (project_id, user_id))
            return cur.fetchone() is not None


def user_has_any_project(user_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM project_members WHERE user_id = %s LIMIT 1", (user_id,))
            return cur.fetchone() is not None


# Project management routes
@app.route('/admin/projects', methods=['POST'])
@admin_required
def create_project():
    data = request.get_json()
    name = data.get('name', '').strip()
    description = data.get('description', '').strip()
    if not name:
        return jsonify({"error": "Project name required"}), 400
    user_id = session.get('user_id')
    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO projects (name, description, created_by, status) VALUES (%s, %s, %s, 'active') RETURNING id",
                    (name, description, user_id))
                project_id = cur.fetchone()[0]
                cur.execute(
                    "INSERT INTO project_members (project_id, user_id, role, added_by) VALUES (%s, %s, 'admin', %s)",
                    (project_id, user_id, user_id))
                cur.execute(
                    "INSERT INTO project_folders (project_id, parent_folder_id, name, created_by) VALUES (%s, NULL, %s, %s)",
                    (project_id, name, user_id))
                conn.commit()
                return jsonify({"success": True, "id": project_id})


@app.route('/admin/projects', methods=['GET'])
def get_projects():
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"projects": [], "has_projects": False})
    if is_admin():
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, name, description, created_at, updated_at, status, archived_at, deletion_scheduled_at FROM projects ORDER BY CASE status WHEN 'active' THEN 1 WHEN 'archived' THEN 2 WHEN 'aborted' THEN 3 END, created_at DESC")
                projects = cur.fetchall()
                has_projects = len(projects) > 0
                return jsonify({"projects": projects, "has_projects": has_projects})
    else:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                            SELECT p.id,
                                   p.name,
                                   p.description,
                                   p.created_at,
                                   p.updated_at,
                                   p.status,
                                   p.archived_at,
                                   p.deletion_scheduled_at
                            FROM projects p
                                     JOIN project_members pm ON p.id = pm.project_id
                            WHERE pm.user_id = %s
                            ORDER BY CASE p.status
                                         WHEN 'active' THEN 1
                                         WHEN 'archived' THEN 2
                                         WHEN 'aborted' THEN 3 END, p.created_at DESC
                            """, (user_id,))
                projects = cur.fetchall()
                has_projects = len(projects) > 0
                return jsonify({"projects": projects, "has_projects": has_projects})


@app.route('/admin/projects/<int:project_id>', methods=['PUT'])
def update_project(project_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    # Allow admin or project manager
    if not is_admin() and not can_manage_files(project_id, user_id):
        return jsonify({"error": "Permission denied"}), 403

    data = request.get_json()
    name = data.get('name', '').strip()
    description = data.get('description', '').strip()
    if not name:
        return jsonify({"error": "Project name required"}), 400

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE projects
                SET name = %s, description = %s, updated_at = NOW()
                WHERE id = %s
                RETURNING id
            """, (name, description, project_id))
            if cur.fetchone():
                conn.commit()
                return jsonify({"success": True})
            else:
                return jsonify({"error": "Project not found"}), 404

@app.route('/admin/projects/<int:project_id>', methods=['DELETE'])
@admin_required
def delete_project(project_id):
    """Permanently delete a project that is archived or aborted."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Check project status
            cur.execute("SELECT status FROM projects WHERE id = %s", (project_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "Project not found"}), 404
            status = row[0]
            if status not in ('archived', 'aborted'):
                return jsonify({"error": "Only archived or aborted projects can be deleted"}), 400

            # Delete physical files stored on disk
            cur.execute("SELECT stored_path FROM project_files WHERE project_id = %s", (project_id,))
            for (stored_path,) in cur.fetchall():
                if stored_path and os.path.exists(stored_path):
                    try:
                        os.remove(stored_path)
                    except Exception as e:
                        logger.warning(f"Could not delete project file {stored_path}: {e}")

            # Delete the project (cascade will remove members, folders, files, versions, comments, recycle bin entries)
            cur.execute("DELETE FROM projects WHERE id = %s", (project_id,))
            conn.commit()
            return jsonify({"success": True})

@app.route('/admin/projects/<int:project_id>/files/<int:file_id>', methods=['DELETE'])
def delete_project_file(project_id, file_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not can_edit_file(project_id, file_id, user_id):
        return jsonify({"error": "Permission denied"}), 403

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get file details
                cur.execute("""
                    SELECT id, original_name, file_size, stored_path, uploaded_by, folder_id, filename, version, file_hash, project_id
                    FROM project_files
                    WHERE id = %s AND project_id = %s
                """, (file_id, project_id))
                file_record = cur.fetchone()
                if not file_record:
                    return jsonify({"error": "File not found"}), 404

                # Insert into project_recycle_bin
                cur.execute("""
                    INSERT INTO project_recycle_bin 
                    (original_table, original_id, project_id, folder_id, file_name, original_name, file_size, stored_path, file_hash, version, uploaded_by, deleted_at, expires_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW() + INTERVAL '3 days')
                """, (
                    'project_files', file_record['id'], file_record['project_id'], file_record['folder_id'],
                    file_record['filename'], file_record['original_name'], file_record['file_size'],
                    file_record['stored_path'], file_record['file_hash'], file_record['version'],
                    file_record['uploaded_by']
                ))

                # Delete from original table
                cur.execute("DELETE FROM project_files WHERE id = %s AND project_id = %s", (file_id, project_id))

                conn.commit()
                return jsonify({"success": True, "moved_to_recycle_bin": True})

@app.route('/admin/projects/<int:project_id>/abort', methods=['POST'])
@admin_required
def abort_project(project_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE projects SET status = 'aborted', archived_at = NOW() WHERE id = %s RETURNING id",
                        (project_id,))
            if cur.fetchone():
                conn.commit()
                return jsonify({"success": True})
            return jsonify({"error": "Project not found"}), 404


@app.route('/admin/projects/<int:project_id>/finish', methods=['POST'])
def finish_project(project_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not can_manage_files(project_id, user_id):
        return jsonify({"error": "Only admin or project manager can finish a project"}), 403
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM projects WHERE id = %s AND status = 'active'", (project_id,))
            project = cur.fetchone()
            if not project:
                return jsonify({"error": "Project not found or already finished/aborted"}), 404
            project_name = project[0]
            cur.execute("SELECT stored_path, original_name FROM project_files WHERE project_id = %s", (project_id,))
            files = cur.fetchall()
            if not files:
                return jsonify({"error": "No files to archive"}), 400
            PROJECT_FILES_ROOT = os.path.join(os.getcwd(), 'project_files')
            zip_dir = os.path.join(PROJECT_FILES_ROOT, 'archives')
            os.makedirs(zip_dir, exist_ok=True)
            safe_name = re.sub(r'[^\w\-_\.]', '_', project_name)
            zip_filename = f"project_{project_id}_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = os.path.join(zip_dir, zip_filename)
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for stored_path, original_name in files:
                    zipf.write(stored_path, original_name)
            cur.execute("UPDATE projects SET status = 'archived', archived_at = NOW() WHERE id = %s", (project_id,))
            conn.commit()
            return jsonify({
                "success": True,
                "download_url": f"/admin/projects/{project_id}/download_archive/{zip_filename}",
                "zip_filename": zip_filename
            })


@app.route('/admin/projects/<int:project_id>/download_archive/<zip_filename>', methods=['GET'])
def download_archive(project_id, zip_filename):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    PROJECT_FILES_ROOT = os.path.join(os.getcwd(), 'project_files')
    zip_dir = os.path.join(PROJECT_FILES_ROOT, 'archives')
    zip_path = os.path.join(zip_dir, zip_filename)
    if not os.path.exists(zip_path):
        return jsonify({"error": "Archive not found"}), 404
    return send_file(zip_path, as_attachment=True, download_name=zip_filename)


# Project members routes
@app.route('/admin/projects/<int:project_id>/members', methods=['GET'])
def get_project_members(project_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT u.user_id, u.username, pm.role, pm.added_at
                FROM project_members pm
                JOIN users u ON pm.user_id = u.user_id
                WHERE pm.project_id = %s AND u.username IS NOT NULL AND u.username != ''
                ORDER BY pm.role, u.username
            """, (project_id,))
            members = cur.fetchall()
            return jsonify({"members": members})


@app.route('/admin/projects/<int:project_id>/members/search', methods=['GET'])
def search_users_to_add(project_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not can_manage_members(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403

    query = request.args.get('q', '').strip()
    if len(query) < 2:
        return jsonify({"users": []})
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Exclude users already in project, exclude current user, exclude global admins
            cur.execute("""
                SELECT user_id, username
                FROM users
                WHERE username ILIKE %s
                  AND user_id NOT IN (SELECT user_id FROM project_members WHERE project_id = %s)
                  AND user_id != %s
                  AND role != 'admin'
                LIMIT 20
            """, (f'%{query}%', project_id, user_id))
            users = cur.fetchall()
            return jsonify({"users": users})

@app.route('/admin/projects/<int:project_id>/all_users', methods=['GET'])
def get_all_users_for_project(project_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    current_user_id = session.get('user_id')
    if not current_user_id:
        return jsonify({"error": "Not logged in"}), 401
    if not is_admin() and not can_manage_members(project_id, current_user_id):
        return jsonify({"error": "Access denied"}), 403

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT user_id, username
                FROM users
                WHERE username IS NOT NULL AND username != ''
                  AND user_id NOT IN (SELECT user_id FROM project_members WHERE project_id = %s)
                  AND user_id != %s
                  AND role != 'admin'
                ORDER BY username
                LIMIT 100
            """, (project_id, current_user_id))
            users = cur.fetchall()
            return jsonify({"users": users})

@app.route('/admin/projects/<int:project_id>/members', methods=['POST'])
def add_project_member(project_id):
    """Add a new member to a project. Only admin or project manager can add members."""
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not can_manage_members(project_id, user_id):
        return jsonify({"error": "Only admin or project manager can add members"}), 403

    data = request.get_json()
    new_user_id = data.get('user_id')
    role = data.get('role', 'member')
    if role == 'manager' and not is_admin():
        return jsonify({"error": "Only admin can add managers"}), 403
    if role not in ('member', 'manager'):
        return jsonify({"error": "Invalid role"}), 400

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Prevent adding a global admin as a member
            cur.execute("SELECT role FROM users WHERE user_id = %s", (new_user_id,))
            row = cur.fetchone()
            if row and row[0] == 'admin':
                return jsonify({"error": "Cannot add a global admin as a project member"}), 403

            # Check if user exists
            if not row:
                return jsonify({"error": "User not found"}), 404

            # Check if already a member
            cur.execute("SELECT 1 FROM project_members WHERE project_id = %s AND user_id = %s",
                        (project_id, new_user_id))
            if cur.fetchone():
                return jsonify({"error": "User already a member"}), 409

            # Add member
            cur.execute("""
                INSERT INTO project_members (project_id, user_id, role, added_by)
                VALUES (%s, %s, %s, %s)
            """, (project_id, new_user_id, role, user_id))
            conn.commit()
            return jsonify({"success": True})

@app.route('/admin/projects/<int:project_id>/members/<user_id>', methods=['PUT'])
@admin_required
def update_member_role(project_id, user_id):
    """Update a member's role in a project. Only global admin can change roles."""
    data = request.get_json()
    new_role = data.get('role')
    if new_role not in ('member', 'manager'):
        return jsonify({"error": "Invalid role"}), 400

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Check if the target user is a global admin
            cur.execute("SELECT role FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if row and row[0] == 'admin':
                return jsonify({"error": "Cannot modify global admin's role"}), 403

            cur.execute("""
                UPDATE project_members
                SET role = %s
                WHERE project_id = %s AND user_id = %s
                RETURNING user_id
            """, (new_role, project_id, user_id))
            if cur.rowcount == 0:
                return jsonify({"error": "Member not found"}), 404
            conn.commit()
            return jsonify({"success": True})


@app.route('/admin/projects/<int:project_id>/members/<user_id>', methods=['DELETE'])
def remove_project_member(project_id, user_id):
    """Remove a member from a project. Only admin or project manager can remove members."""
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    current_user_id = session.get('user_id')
    if not can_manage_members(project_id, current_user_id):
        return jsonify({"error": "Only admin or project manager can remove members"}), 403

    # Check if target is global admin
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT role FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if row and row[0] == 'admin':
                return jsonify({"error": "Cannot remove a global admin"}), 403

            # Also prevent removing the last manager? (optional, but good practice)
            cur.execute("""
                SELECT role FROM project_members
                WHERE project_id = %s AND user_id = %s
            """, (project_id, user_id))
            target_member = cur.fetchone()
            if not target_member:
                return jsonify({"error": "Member not found"}), 404

            target_role = target_member[0]
            if target_role == 'admin':
                return jsonify({"error": "Cannot remove the project admin"}), 403
            if target_role == 'manager' and not is_admin():
                return jsonify({"error": "Only admin can remove managers"}), 403

            cur.execute("DELETE FROM project_members WHERE project_id = %s AND user_id = %s",
                        (project_id, user_id))
            if cur.rowcount == 0:
                return jsonify({"error": "Member not found"}), 404
            conn.commit()
            return jsonify({"success": True})

@app.route('/admin/projects/<int:project_id>/transfer_manager/<user_id>', methods=['POST'])
def transfer_manager_role(project_id, user_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    current_user_id = session.get('user_id')
    current_role = get_user_role_in_project(project_id, current_user_id)
    if current_role != 'manager':
        return jsonify({"error": "Only a manager can transfer manager rights"}), 403
    target_role = get_user_role_in_project(project_id, user_id)
    if target_role != 'member':
        return jsonify({"error": "Target user must be a member"}), 400
    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                cur.execute("UPDATE project_members SET role = 'member' WHERE project_id = %s AND user_id = %s",
                            (project_id, current_user_id))
                cur.execute("UPDATE project_members SET role = 'manager' WHERE project_id = %s AND user_id = %s",
                            (project_id, user_id))
                conn.commit()
    return jsonify({"success": True})


# Project folders and files (abbreviated but with permission checks)
def ensure_root_folder(project_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM project_folders WHERE project_id = %s AND parent_folder_id IS NULL",
                        (project_id,))
            if not cur.fetchone():
                cur.execute("SELECT name FROM projects WHERE id = %s", (project_id,))
                row = cur.fetchone()
                if row:
                    project_name = row[0]
                    cur.execute(
                        "INSERT INTO project_folders (project_id, parent_folder_id, name, created_by) VALUES (%s, NULL, %s, %s)",
                        (project_id, project_name, session.get('user_id')))
                    conn.commit()
                    logger.info(f"Created missing root folder for project {project_id}")


def build_folder_path(folder_id, folder_dict):
    parts = []
    current_id = folder_id
    while current_id:
        folder = folder_dict.get(current_id)
        if not folder:
            break
        parts.insert(0, folder['name'])
        current_id = folder['parent_folder_id']
    return '/' + '/'.join(parts) if parts else '/'


@app.route('/admin/projects/<int:project_id>/folders', methods=['GET'])
def get_folders(project_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    ensure_root_folder(project_id)
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, parent_folder_id, name FROM project_folders WHERE project_id = %s ORDER BY parent_folder_id, name",
                    (project_id,))
                folders = cur.fetchall()
                if not folders:
                    return jsonify({"folders": []})
                folder_dict = {f['id']: f for f in folders}
                for f in folder_dict.values():
                    f['children'] = []
                root_folders = []
                for f in folder_dict.values():
                    if f['parent_folder_id'] is None:
                        root_folders.append(f)
                    else:
                        parent = folder_dict.get(f['parent_folder_id'])
                        if parent:
                            parent['children'].append(f)
                        else:
                            root_folders.append(f)
                for f in folder_dict.values():
                    f['path'] = build_folder_path(f['id'], folder_dict)
                return jsonify({"folders": root_folders})
    except Exception as e:
        logger.error(f"Error in get_folders: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route('/admin/projects/<int:project_id>/folders', methods=['POST'])
def create_folder(project_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    data = request.get_json()
    name = data.get('name', '').strip()
    parent_folder_id = data.get('parent_folder_id')
    if not name:
        return jsonify({"error": "Folder name required"}), 400
    if parent_folder_id is None:
        return jsonify({"error": "Cannot create root folder. Only one root folder exists per project."}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s",
                        (parent_folder_id, project_id))
            if not cur.fetchone():
                return jsonify({"error": "Parent folder not found"}), 404
            cur.execute(
                "INSERT INTO project_folders (project_id, parent_folder_id, name, created_by) VALUES (%s, %s, %s, %s) RETURNING id",
                (project_id, parent_folder_id, name, user_id))
            new_id = cur.fetchone()[0]
            conn.commit()
            return jsonify({"success": True, "id": new_id})


@app.route('/admin/projects/<int:project_id>/folders/<int:folder_id>', methods=['DELETE'])
def delete_folder(project_id, folder_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not can_edit_folder(project_id, folder_id, user_id):
        return jsonify({"error": "Permission denied"}), 403

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get all subfolder IDs recursively
                cur.execute("""
                    WITH RECURSIVE folder_tree AS (
                        SELECT id, name, parent_folder_id, created_at, created_by
                        FROM project_folders
                        WHERE id = %s AND project_id = %s
                        UNION ALL
                        SELECT pf.id, pf.name, pf.parent_folder_id, pf.created_at, pf.created_by
                        FROM project_folders pf
                        INNER JOIN folder_tree ft ON pf.parent_folder_id = ft.id
                    )
                    SELECT * FROM folder_tree
                """, (folder_id, project_id))
                folders = cur.fetchall()

                # Store folder IDs in order (top-down for later restoration)
                folder_ids = [f['id'] for f in folders]
                # Move each folder to recycle bin
                for f in folders:
                    cur.execute("""
                        INSERT INTO project_folders_recycle_bin
                        (original_id, project_id, name, parent_folder_id, original_parent_id, created_at, created_by, deleted_at, expires_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW() + INTERVAL '3 days')
                    """, (
                        f['id'], project_id, f['name'], f['parent_folder_id'], f['parent_folder_id'],
                        f['created_at'], f['created_by']
                    ))

                # Move all files inside these folders to project_recycle_bin
                if folder_ids:
                    placeholders = ','.join(['%s'] * len(folder_ids))
                    cur.execute(f"""
                        SELECT id, original_name, file_size, stored_path, file_hash, version, uploaded_by, folder_id
                        FROM project_files
                        WHERE project_id = %s AND folder_id IN ({placeholders})
                    """, [project_id] + folder_ids)
                    files = cur.fetchall()
                    for f in files:
                        cur.execute("""
                            INSERT INTO project_recycle_bin 
                            (original_table, original_id, project_id, folder_id, file_name, original_name, file_size, stored_path, file_hash, version, uploaded_by, deleted_at, expires_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW() + INTERVAL '3 days')
                        """, (
                            'project_files', f['id'], project_id, f['folder_id'],
                            f['original_name'], f['original_name'], f['file_size'],
                            f['stored_path'], f['file_hash'], f['version'],
                            f['uploaded_by']
                        ))
                    # Delete the files
                    cur.execute(f"""
                        DELETE FROM project_files
                        WHERE project_id = %s AND folder_id IN ({placeholders})
                    """, [project_id] + folder_ids)

                # Delete the folders themselves (they are already backed up)
                cur.execute(f"""
                    DELETE FROM project_folders
                    WHERE project_id = %s AND id IN ({','.join(['%s']*len(folder_ids))})
                """, [project_id] + folder_ids)

                conn.commit()
                return jsonify({
                    "success": True,
                    "folders_moved": len(folders),
                    "files_moved": len(files) if files else 0
                })

@app.route('/admin/projects/<int:project_id>/folders/<int:folder_id>/rename', methods=['PUT'])
def rename_folder(project_id, folder_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not can_edit_folder(project_id, folder_id, user_id):
        return jsonify({"error": "Permission denied"}), 403
    data = request.get_json()
    new_name = data.get('name', '').strip()
    if not new_name:
        return jsonify({"error": "Folder name required"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT parent_folder_id FROM project_folders WHERE id = %s AND project_id = %s",
                        (folder_id, project_id))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "Folder not found"}), 404
            parent_id = row[0]
            cur.execute(
                "SELECT id FROM project_folders WHERE project_id = %s AND parent_folder_id = %s AND name = %s AND id != %s",
                (project_id, parent_id, new_name, folder_id))
            if cur.fetchone():
                return jsonify({"error": "A folder with this name already exists in this location"}), 400
            cur.execute("UPDATE project_folders SET name = %s WHERE id = %s", (new_name, folder_id))
            conn.commit()
            return jsonify({"success": True})


# Project files management
PROJECT_FILES_ROOT = os.path.join(os.getcwd(), 'project_files')
os.makedirs(PROJECT_FILES_ROOT, exist_ok=True)


def get_project_file_path(project_id, unique_filename):
    project_dir = os.path.join(PROJECT_FILES_ROOT, str(project_id))
    os.makedirs(project_dir, exist_ok=True)
    return os.path.join(project_dir, unique_filename)

@app.route('/admin/projects/<int:project_id>/folders/<int:folder_id>/upload', methods=['POST'])
def upload_project_file(project_id, folder_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403

    # Check project status
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM projects WHERE id = %s", (project_id,))
            row = cur.fetchone()
            if not row or row[0] != 'active':
                return jsonify({"error": "Project is not active. Cannot upload."}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Verify folder exists
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s", (folder_id, project_id))
            if not cur.fetchone():
                return jsonify({"error": "Folder not found"}), 404

    original_name = file.filename
    file_bytes = file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    file.seek(0)

    # Check for duplicate by hash
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, original_name, stored_path, version, folder_id FROM project_files WHERE project_id = %s AND file_hash = %s", (project_id, file_hash))
            duplicate = cur.fetchone()
            if duplicate:
                return jsonify({
                    "duplicate": True,
                    "existing_file": {
                        "id": duplicate['id'],
                        "original_name": duplicate['original_name'],
                        "folder_id": duplicate['folder_id'],
                        "version": duplicate['version']
                    },
                    "new_filename": original_name
                })

    # Save original file (no text extraction)
    ext = os.path.splitext(original_name)[1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = get_project_file_path(project_id, unique_name)
    file.save(stored_path)
    file_size = os.path.getsize(stored_path)

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO project_files (project_id, folder_id, filename, original_name, file_size,
                                               stored_path, uploaded_by, file_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (project_id, folder_id, unique_name, original_name, file_size, stored_path, user_id, file_hash))
                file_id = cur.fetchone()[0]
                conn.commit()
                return jsonify({"success": True, "file_id": file_id, "original_name": original_name, "version": 1})

@app.route('/admin/projects/<int:project_id>/files/<int:file_id>/new_version', methods=['POST'])
def new_file_version(project_id, file_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT stored_path, version, original_name, folder_id, filename
                        FROM project_files
                        WHERE id = %s
                          AND project_id = %s
                        """, (file_id, project_id))
            existing = cur.fetchone()
            if not existing:
                return jsonify({"error": "File not found"}), 404

            original_name = file.filename
            file_bytes = file.read()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            file.seek(0)

            # Extract text from new version
            file_content, _ = extract_text_from_file(file)
            if not file_content or file_content.startswith("["):
                return jsonify({"error": "Could not extract text from new version"}), 400

            ext = os.path.splitext(original_name)[1]
            unique_name = f"{uuid.uuid4().hex}{ext}"
            stored_path = get_project_file_path(project_id, unique_name)
            file.save(stored_path)
            file_size = os.path.getsize(stored_path)
            new_version = existing['version'] + 1

            # Store old version in versions table
            cur.execute("""
                        INSERT INTO project_file_versions (file_id, version, stored_path, file_size, uploaded_by)
                        VALUES (%s, %s, %s, %s, %s)
                        """, (file_id, existing['version'], existing['stored_path'], file_size, user_id))

            # Update main record with new version, new file, new hash, new content
            cur.execute("""
                        UPDATE project_files
                        SET version       = %s,
                            stored_path   = %s,
                            file_size     = %s,
                            uploaded_at   = NOW(),
                            uploaded_by   = %s,
                            file_hash     = %s,
                            original_name = %s,
                            content       = %s
                        WHERE id = %s
                        """,
                        (new_version, stored_path, file_size, user_id, file_hash, original_name, file_content, file_id))

            # Log new version action
            cur.execute("""
                        INSERT INTO project_file_usage (file_id, user_id, action, details)
                        VALUES (%s, %s, 'new_version', %s)
                        """, (file_id, user_id, json.dumps({'version': new_version, 'size': file_size})))

            conn.commit()
            return jsonify(
                {"success": True, "file_id": file_id, "original_name": original_name, "version": new_version})


@app.route('/admin/projects/<int:project_id>/folders/<int:folder_id>/files', methods=['GET'])
def list_project_files(project_id, folder_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT id,
                               original_name,
                               file_size,
                               version,
                               uploaded_at,
                               uploaded_by,
                               (SELECT username FROM users WHERE user_id = project_files.uploaded_by) as uploaded_by_name
                        FROM project_files
                        WHERE project_id = %s
                          AND folder_id = %s
                        ORDER BY uploaded_at DESC
                        """, (project_id, folder_id))
            files = cur.fetchall()
            for f in files:
                f['file_size_kb'] = round(f['file_size'] / 1024, 1)
                f['uploaded_at_str'] = f['uploaded_at'].strftime('%Y-%m-%d %H:%M:%S')
                f['can_delete'] = can_edit_file(project_id, f['id'], user_id)
                f['can_rename'] = can_edit_file(project_id, f['id'], user_id)
            return jsonify({"files": files})


@app.route('/admin/projects/<int:project_id>/files', methods=['GET'])
def list_root_files(project_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT id,
                               original_name,
                               file_size,
                               version,
                               uploaded_at,
                               uploaded_by,
                               (SELECT username FROM users WHERE user_id = project_files.uploaded_by) as uploaded_by_name,
                               NULL                                                                   as folder_name
                        FROM project_files
                        WHERE project_id = %s
                          AND folder_id IS NULL
                        ORDER BY uploaded_at DESC
                        """, (project_id,))
            files = cur.fetchall()
            for f in files:
                f['file_size_kb'] = round(f['file_size'] / 1024, 1)
                f['uploaded_at_str'] = f['uploaded_at'].strftime('%Y-%m-%d %H:%M:%S')
                f['can_delete'] = can_edit_file(project_id, f['id'], user_id)
                f['can_rename'] = can_edit_file(project_id, f['id'], user_id)
            return jsonify({"files": files})


@app.route('/admin/projects/<int:project_id>/files/<int:file_id>/versions', methods=['GET'])
def get_file_versions(project_id, file_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT version,
                               file_size,
                               uploaded_at,
                               uploaded_by,
                               (SELECT username FROM users WHERE user_id = fv.uploaded_by) as uploaded_by_name
                        FROM project_file_versions fv
                        WHERE file_id = %s
                        ORDER BY version DESC
                        """, (file_id,))
            versions = cur.fetchall()
            return jsonify({"versions": versions})


@app.route('/admin/projects/<int:project_id>/files/<int:file_id>/download', methods=['GET'])
def download_project_file(project_id, file_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    version = request.args.get('version', type=int)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if version:
                cur.execute(
                    "SELECT stored_path, original_name FROM project_file_versions WHERE file_id = %s AND version = %s",
                    (file_id, version))
            else:
                cur.execute("SELECT stored_path, original_name FROM project_files WHERE id = %s", (file_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "File not found"}), 404
            stored_path, original_name = row
    return send_file(stored_path, as_attachment=True, download_name=original_name)


@app.route('/admin/projects/<int:project_id>/files/<int:file_id>/comments', methods=['GET'])
def get_file_comments(project_id, file_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT c.id, c.comment, c.created_at, u.username
                        FROM project_file_comments c
                                 JOIN users u ON c.user_id = u.user_id
                        WHERE c.file_id = %s
                        ORDER BY c.created_at ASC
                        """, (file_id,))
            comments = cur.fetchall()
            return jsonify({"comments": comments})


@app.route('/admin/projects/<int:project_id>/files/<int:file_id>/comments', methods=['POST'])
def add_file_comment(project_id, file_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    data = request.get_json()
    comment = data.get('comment', '').strip()
    if not comment:
        return jsonify({"error": "Comment cannot be empty"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO project_file_comments (file_id, user_id, comment) VALUES (%s, %s, %s)",
                        (file_id, user_id, comment))
            conn.commit()
            return jsonify({"success": True})


@app.route('/admin/projects/<int:project_id>/files/<int:file_id>/move', methods=['POST'])
def move_file(project_id, file_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not can_move_file(project_id, file_id, user_id):
        return jsonify({"error": "Permission denied"}), 403
    data = request.get_json()
    target_folder_id = data.get('folder_id')
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if target_folder_id:
                cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s", (target_folder_id, project_id))
                if not cur.fetchone():
                    return jsonify({"error": "Target folder not found in this project"}), 404
            cur.execute("UPDATE project_files SET folder_id = %s WHERE id = %s AND project_id = %s", (target_folder_id, file_id, project_id))
            if cur.rowcount == 0:
                return jsonify({"error": "File not found"}), 404
            conn.commit()
            return jsonify({"success": True})

@app.route('/admin/projects/<int:project_id>/files/batch_move', methods=['POST'])
def batch_move_files(project_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    file_ids = data.get('file_ids', [])
    target_folder_id = data.get('folder_id')
    if not file_ids:
        return jsonify({"error": "No files selected"}), 400
    if not target_folder_id:
        return jsonify({"error": "Target folder required"}), 400

    # Check if user is a project member (any role)
    role = get_user_role_in_project(project_id, user_id)
    if not role and not is_admin():
        return jsonify({"error": "You are not a member of this project"}), 403

    # Verify target folder exists
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s", (target_folder_id, project_id))
            if not cur.fetchone():
                return jsonify({"error": "Target folder not found in this project"}), 404

            # Verify all files belong to the project
            placeholders = ','.join(['%s'] * len(file_ids))
            cur.execute(f"""
                SELECT id FROM project_files 
                WHERE id IN ({placeholders}) AND project_id = %s
            """, file_ids + [project_id])
            found = cur.fetchall()
            if len(found) != len(file_ids):
                return jsonify({"error": "Some files not found in this project"}), 404

            # Perform the move (all members allowed)
            cur.execute(f"""
                UPDATE project_files SET folder_id = %s 
                WHERE id IN ({placeholders}) AND project_id = %s
            """, [target_folder_id] + file_ids + [project_id])
            conn.commit()
            return jsonify({"success": True, "moved_count": len(file_ids)})

@app.route('/admin/projects/<int:project_id>/batch_download', methods=['POST'])
def batch_download_files(project_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    data = request.get_json()
    file_ids = data.get('file_ids', [])
    if not file_ids:
        return jsonify({"error": "No files selected"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            placeholders = ','.join(['%s'] * len(file_ids))
            cur.execute(
                f"SELECT stored_path, original_name FROM project_files WHERE id IN ({placeholders}) AND project_id = %s",
                file_ids + [project_id])
            files = cur.fetchall()
            if not files:
                return jsonify({"error": "No valid files found"}), 404
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for stored_path, original_name in files:
                    zipf.write(stored_path, original_name)
            zip_buffer.seek(0)
            return send_file(zip_buffer, as_attachment=True, download_name=f"project_{project_id}_files.zip",
                             mimetype='application/zip')


@app.route('/admin/projects/<int:project_id>/files/search', methods=['GET'])
def search_project_files(project_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    query = request.args.get('q', '').strip()
    if len(query) < 2:
        return jsonify({"files": []})
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT f.id, f.original_name, f.file_size, f.uploaded_at, fo.name as folder_name
                        FROM project_files f
                                 LEFT JOIN project_folders fo ON f.folder_id = fo.id
                        WHERE f.project_id = %s
                          AND f.original_name ILIKE %s
                        ORDER BY f.uploaded_at DESC
                        LIMIT 50
                        """, (project_id, f'%{query}%'))
            files = cur.fetchall()
            for f in files:
                f['file_size_kb'] = round(f['file_size'] / 1024, 1)
            return jsonify({"files": files})


@app.route('/admin/projects/<int:project_id>/folders/<int:folder_id>/comments', methods=['GET'])
def get_folder_comments(project_id, folder_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT c.id, c.comment, c.created_at, u.username
                        FROM project_folder_comments c
                                 JOIN users u ON c.user_id = u.user_id
                        WHERE c.folder_id = %s
                        ORDER BY c.created_at ASC
                        """, (folder_id,))
            comments = cur.fetchall()
            return jsonify({"comments": comments})


@app.route('/admin/projects/<int:project_id>/folders/<int:folder_id>/comments', methods=['POST'])
def add_folder_comment(project_id, folder_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    data = request.get_json()
    comment = data.get('comment', '').strip()
    if not comment:
        return jsonify({"error": "Comment cannot be empty"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO project_folder_comments (folder_id, user_id, comment) VALUES (%s, %s, %s)",
                        (folder_id, user_id, comment))
            conn.commit()
            return jsonify({"success": True})


@app.route('/admin/projects/<int:project_id>/files/<int:file_id>/rename', methods=['PUT'])
def rename_project_file(project_id, file_id):
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not can_edit_file(project_id, file_id, user_id):
        return jsonify({"error": "Permission denied"}), 403
    data = request.get_json()
    new_name = data.get('original_name', '').strip()
    if not new_name:
        return jsonify({"error": "New name required"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE project_files SET original_name = %s WHERE id = %s AND project_id = %s",
                        (new_name, file_id, project_id))
            if cur.rowcount == 0:
                return jsonify({"error": "File not found"}), 404
            conn.commit()
    return jsonify({"success": True})


# File station routes (same as before)
@app.route('/upload_file', methods=['POST'])
def upload_file():
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403

    logger.info(f"Upload request headers: {dict(request.headers)}")
    logger.info(f"Files in request: {request.files}")
    logger.info(f"Form data: {request.form}")

    if 'file' not in request.files:
        logger.warning("No file in request")
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    logger.info(f"Filename: {file.filename}, content_type: {file.content_type}")

    if file.filename == '':
        logger.warning("Empty filename")
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        logger.warning(f"File type not allowed: {file.filename}")
        return jsonify({"error": f"不支持的文件类型: {file.filename}"}), 400

    user_id = get_user_id()
    thread_id = session.get('thread_id')
    logger.info(f"User ID: {user_id}, Thread ID: {thread_id}")

    if not thread_id:
        # Auto-create thread if missing
        thread_id = str(uuid.uuid4())
        session['thread_id'] = thread_id
        get_or_create_session(thread_id)
        logger.info(f"Created new thread {thread_id} for user {user_id}")

    # Read file content and compute hash
    file_bytes = file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    file.seek(0)

    # ---------- Anonymous user branch ----------
    if session.get('consent_value', 0) != 1:
        anon_id = user_id
        temp_dir = get_anon_temp_dir(anon_id)
        existing_files = [f for f in os.listdir(temp_dir) if f.endswith('.txt')]
        if len(existing_files) >= 5:
            return jsonify({"error": "Anonymous users can only store up to 5 files."}), 400
        original_size = len(file_bytes)
        if original_size > 5 * 1024 * 1024:
            return jsonify({"error": "File exceeds 5MB limit for anonymous users."}), 400

        # Extract text (without caching/DB)
        file_content, _ = extract_text_from_file(file)
        if not file_content or file_content.startswith("["):
            return jsonify({"error": "Could not extract text from file"}), 400

        safe_name = re.sub(r'[^\w\-_\. ]', '_', file.filename) + '.txt'
        file_path = os.path.join(temp_dir, safe_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(file_content)

        # Add to in‑memory cache for this session
        add_to_cache(thread_id, file.filename, file_content, user_id)
        # No DB record, no original file stored
        return jsonify({"success": True, "filename": file.filename})

    # ---------- Registered user branch ----------
    # Check for duplicate by hash
    existing = None
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, filename, content, original_stored_path
                FROM user_files
                WHERE user_id = %s AND file_hash = %s
            """, (user_id, file_hash))
            existing = cur.fetchone()
            if existing and request.form.get('force') != 'true':
                return jsonify({
                    "exists": True,
                    "file_id": existing[0],
                    "filename": existing[1],
                    "content": existing[2],
                    "original_path": existing[3] if existing[3] else None
                })

    # Extract text content (will be stored permanently)
    file_content, _ = extract_text_from_file(file)
    if not file_content or file_content.startswith("["):
        return jsonify({"error": "Could not extract text from file"}), 400

    # Save original file to disk with 3‑day expiration
    ext = os.path.splitext(file.filename)[1]
    unique_name = f"{file_hash}_{int(time.time())}{ext}"
    original_dir = os.path.join(USER_FILES_ORIGINAL_ROOT, user_id)
    os.makedirs(original_dir, exist_ok=True)
    original_path = os.path.join(original_dir, unique_name)
    file.seek(0)
    file.save(original_path)
    original_size = os.path.getsize(original_path)

    # Store in DB: text permanent, original file expires in 3 days
    add_to_cache(thread_id, file.filename, file_content, user_id)
    record_file_usage(thread_id, file.filename, 'standalone_upload', "上传文件供日后使用")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if existing and request.form.get('force') == 'true':
                # Delete old physical file if it exists
                old_path = existing[3]
                if old_path and os.path.exists(old_path):
                    os.remove(old_path)
                # Update existing record
                cur.execute("""
                    UPDATE user_files
                    SET content = %s, size_bytes = %s,
                        original_stored_path = %s, file_hash = %s, filename = %s,
                        expires_at = NOW() + INTERVAL '3 days',
                        original_expires_at = NOW() + INTERVAL '3 days'
                    WHERE id = %s
                """, (file_content, len(file_content), original_path, file_hash, file.filename, existing[0]))
            else:
                cur.execute("""
                    INSERT INTO user_files
                        (user_id, thread_id, filename, content, size_bytes, expires_at,
                         original_stored_path, file_hash, original_expires_at)
                    VALUES (%s, %s, %s, %s, %s, NOW() + INTERVAL '3 days', %s, %s, NOW() + INTERVAL '3 days')
                    ON CONFLICT (thread_id, filename) DO UPDATE SET
                        content = EXCLUDED.content,
                        size_bytes = EXCLUDED.size_bytes,
                        expires_at = NOW() + INTERVAL '3 days',
                        original_stored_path = EXCLUDED.original_stored_path,
                        file_hash = EXCLUDED.file_hash,
                        original_expires_at = NOW() + INTERVAL '3 days'
                """, (user_id, thread_id, file.filename, file_content, len(file_content),
                      original_path, file_hash))
            conn.commit()

    return jsonify({"success": True, "filename": file.filename})

@app.route('/download_original_file', methods=['POST'])
def download_original_file():
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403

    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({"error": "Missing filename"}), 400

    user_id = get_user_id()
    thread_id = session.get('thread_id')
    if not thread_id:
        return jsonify({"error": "No active session"}), 400

    if session.get('consent_value', 0) != 1:
        return jsonify({
            "error": "anonymous_not_allowed",
            "message": "匿名用户无法下载原文件。请注册或登录账户后使用此功能。"
        }), 403

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT original_stored_path
                FROM user_files
                WHERE user_id = %s AND thread_id = %s AND filename = %s
                  AND (original_expires_at IS NULL OR original_expires_at > NOW())
            """, (user_id, thread_id, filename))
            row = cur.fetchone()
            if not row or not row[0]:
                return jsonify({"error": "Original file not found or expired"}), 404
            original_path = row[0]
            if not os.path.exists(original_path):
                return jsonify({"error": "File missing on server"}), 404
            return send_file(original_path, as_attachment=True, download_name=filename)

@app.route('/delete_file_station', methods=['POST'])
def delete_file_station():
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403

    data = request.get_json()
    file_id = data.get('file_id')
    if not file_id:
        return jsonify({"error": "Missing file_id"}), 400

    user_id = get_user_id()

    # Anonymous user branch
    if session.get('consent_value', 0) != 1:
        anon_id = user_id
        temp_dir = get_anon_temp_dir(anon_id)
        fpath = os.path.join(temp_dir, file_id)
        if os.path.exists(fpath):
            os.remove(fpath)
            return jsonify({"success": True})
        else:
            return jsonify({"error": "File not found"}), 404

    # Logged-in user – move to recycle bin
    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get file details
                cur.execute("""
                    SELECT id, filename, content, size_bytes, original_stored_path, file_hash, thread_id
                    FROM user_files
                    WHERE id = %s AND user_id = %s
                """, (file_id, user_id))
                file_record = cur.fetchone()
                if not file_record:
                    return jsonify({"error": "File not found or not owned"}), 404

                # Insert into recycle_bin (omit original_data)
                cur.execute("""
                    INSERT INTO recycle_bin 
                    (original_table, original_id, user_id, file_name, file_content, file_size, 
                     original_stored_path, file_hash, thread_id, deleted_at, expires_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW() + INTERVAL '3 days')
                """, (
                    'user_files', file_record['id'], user_id, file_record['filename'],
                    file_record['content'], file_record['size_bytes'], file_record['original_stored_path'],
                    file_record['file_hash'], file_record['thread_id']
                ))

                # Delete from original table
                cur.execute("DELETE FROM user_files WHERE id = %s AND user_id = %s", (file_id, user_id))

                conn.commit()
                return jsonify({"success": True, "moved_to_recycle_bin": True})

@app.route('/get_file_station', methods=['GET'])
def get_file_station():
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = get_user_id()
    if session.get('consent_value', 0) != 1:
        anon_id = user_id
        temp_dir = get_anon_temp_dir(anon_id)
        files = []
        try:
            for fname in os.listdir(temp_dir):
                if fname.endswith('.txt'):
                    fpath = os.path.join(temp_dir, fname)
                    stat = os.stat(fpath)
                    original_name = fname[:-4] if fname.endswith('.txt') else fname
                    files.append({
                        "id": fname,
                        "filename": original_name,
                        "size_bytes": stat.st_size,
                        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "expires_at": None,
                        "usage": []
                    })
        except Exception as e:
            logger.error(f"Failed to list anon files: {e}")
            return jsonify({"error": "无法读取临时文件"}), 500
        return jsonify({"files": files})
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                db_execute_readonly(cur)
                cur.execute("""
                    SELECT uf.id::text AS id,
                           uf.filename,
                           uf.size_bytes,
                           uf.created_at,
                           uf.expires_at,
                           uf.meta_data,
                           COALESCE(
                               (SELECT json_agg(
                                   json_build_object(
                                       'usage_type', fu.usage_type,
                                       'question', fu.question,
                                       'timestamp', fu.timestamp,
                                       'thread_id', fu.thread_id
                                   ) ORDER BY fu.timestamp DESC
                               )
                               FROM file_usage fu
                               WHERE fu.user_id = uf.user_id AND fu.filename = uf.filename
                               LIMIT 10),
                               '[]'::json
                           ) AS usage
                    FROM user_files uf
                    WHERE uf.user_id = %s AND (uf.expires_at IS NULL OR uf.expires_at > NOW())
                    ORDER BY uf.created_at DESC
                    LIMIT 20
                """, (user_id,))
                files = cur.fetchall()
                return jsonify({"files": files})
    except Exception as e:
        logger.error(f"DB error in get_file_station: {e}", exc_info=True)
        return jsonify({"error": f"数据库查询失败: {str(e)}"}), 500

@app.route('/get_recycle_bin', methods=['GET'])
def get_recycle_bin():
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    # Ensure recycle_bin table has the required columns (safe migration)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DO $$ 
                BEGIN
                    BEGIN
                        ALTER TABLE recycle_bin ADD COLUMN deletion_reason TEXT DEFAULT 'manual';
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                END $$;
            """)
            conn.commit()

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # ----- Chat files recycle bin (with deletion_reason) -----
            cur.execute("""
                SELECT id, original_table, original_id, file_name, file_size, deleted_at, expires_at, deletion_reason, 'chat' as source
                FROM recycle_bin
                WHERE user_id = %s AND expires_at > NOW()
                ORDER BY deleted_at DESC
            """, (user_id,))
            chat_items = cur.fetchall()

            # ----- Project files recycle bin -----
            cur.execute("""
                SELECT prb.id, prb.original_table, prb.original_id, prb.file_name, prb.file_size, 
                       prb.deleted_at, prb.expires_at, p.name as project_name, 'project' as source
                FROM project_recycle_bin prb
                JOIN projects p ON prb.project_id = p.id
                WHERE prb.expires_at > NOW()
                ORDER BY prb.deleted_at DESC
            """)
            project_items = cur.fetchall()

            # ----- Project folders recycle bin -----
            cur.execute("""
                SELECT pfrb.id, pfrb.original_id, pfrb.name, pfrb.original_parent_id, pfrb.deleted_at, pfrb.expires_at, 
                       p.name as project_name, 'folder' as source
                FROM project_folders_recycle_bin pfrb
                JOIN projects p ON pfrb.project_id = p.id
                WHERE pfrb.expires_at > NOW()
                ORDER BY pfrb.deleted_at DESC
            """)
            folder_items = cur.fetchall()

            return jsonify({
                "chat_items": chat_items,
                "project_items": project_items,
                "folder_items": folder_items
            })

@app.route('/restore_from_recycle_bin', methods=['POST'])
def restore_from_recycle_bin():
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    item_id = data.get('item_id')
    source = data.get('source')        # 'chat', 'project', 'folder'
    section = data.get('section')      # 'chat', 'project_files', 'project_folders'
    restore_all = data.get('restore_all', False)

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # ---- Restore all items in a section ----
                if restore_all:
                    restored_count = 0
                    if section == 'chat':
                        cur.execute("SELECT * FROM recycle_bin WHERE user_id = %s AND expires_at > NOW()", (user_id,))
                        items = cur.fetchall()
                        for item in items:
                            meta_data = {}
                            if item.get('deletion_reason') == 'chat_deleted':
                                meta_data['restored_from'] = 'chat_deletion'
                                meta_data['original_thread_id'] = item.get('original_thread_id')
                            meta_data_json = json.dumps(meta_data)
                            cur.execute("""
                                INSERT INTO user_files (user_id, thread_id, filename, content, size_bytes, expires_at,
                                                        original_stored_path, file_hash, meta_data)
                                VALUES (%s, %s, %s, %s, %s, NOW() + INTERVAL '3 days', %s, %s, %s::jsonb)
                            """, (user_id, None, item['file_name'], item['file_content'], item['file_size'],
                                  item['original_stored_path'], item['file_hash'], meta_data_json))
                            cur.execute("DELETE FROM recycle_bin WHERE id = %s", (item['id'],))
                            restored_count += 1
                    elif section == 'project_files':
                        cur.execute("SELECT * FROM project_recycle_bin WHERE expires_at > NOW()")
                        items = cur.fetchall()
                        for item in items:
                            folder_id = item['folder_id']
                            if folder_id:
                                cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s",
                                            (folder_id, item['project_id']))
                                if not cur.fetchone():
                                    restore_folder_path_for_file(item, conn, cur)
                            cur.execute("""
                                INSERT INTO project_files (project_id, folder_id, filename, original_name, file_size,
                                                           stored_path, version, uploaded_by, file_hash)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (item['project_id'], item['folder_id'], item['file_name'], item['original_name'],
                                  item['file_size'], item['stored_path'], item['version'],
                                  item['uploaded_by'], item['file_hash']))
                            cur.execute("DELETE FROM project_recycle_bin WHERE id = %s", (item['id'],))
                            restored_count += 1
                    elif section == 'project_folders':
                        cur.execute("SELECT * FROM project_folders_recycle_bin WHERE expires_at > NOW()")
                        folders = cur.fetchall()
                        for folder in folders:
                            restore_folder_recursive(folder, conn, cur)
                            restored_count += 1
                    else:
                        return jsonify({"error": "Invalid section"}), 400
                    return jsonify({"success": True, "restored_count": restored_count})

                # ---- Single item restore ----
                if source == 'chat':
                    cur.execute("SELECT * FROM recycle_bin WHERE id = %s AND user_id = %s AND expires_at > NOW()",
                                (item_id, user_id))
                    item = cur.fetchone()
                    if not item:
                        return jsonify({"error": "Item not found or expired"}), 404
                    meta_data = {}
                    if item.get('deletion_reason') == 'chat_deleted':
                        meta_data['restored_from'] = 'chat_deletion'
                        meta_data['original_thread_id'] = item.get('original_thread_id')
                    meta_data_json = json.dumps(meta_data)
                    cur.execute("""
                        INSERT INTO user_files (user_id, thread_id, filename, content, size_bytes, expires_at,
                                                original_stored_path, file_hash, meta_data)
                        VALUES (%s, %s, %s, %s, %s, NOW() + INTERVAL '3 days', %s, %s, %s::jsonb)
                    """, (user_id, None, item['file_name'], item['file_content'], item['file_size'],
                          item['original_stored_path'], item['file_hash'], meta_data_json))
                    cur.execute("DELETE FROM recycle_bin WHERE id = %s", (item_id,))

                elif source == 'project':
                    cur.execute("SELECT * FROM project_recycle_bin WHERE id = %s", (item_id,))
                    item = cur.fetchone()
                    if not item:
                        return jsonify({"error": "Item not found"}), 404
                    folder_id = item['folder_id']
                    if folder_id:
                        cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s",
                                    (folder_id, item['project_id']))
                        if not cur.fetchone():
                            restore_folder_path_for_file(item, conn, cur)
                    cur.execute("""
                        INSERT INTO project_files (project_id, folder_id, filename, original_name, file_size,
                                                   stored_path, version, uploaded_by, file_hash)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (item['project_id'], item['folder_id'], item['file_name'], item['original_name'],
                          item['file_size'], item['stored_path'], item['version'],
                          item['uploaded_by'], item['file_hash']))
                    cur.execute("DELETE FROM project_recycle_bin WHERE id = %s", (item_id,))

                elif source == 'folder':
                    cur.execute("SELECT * FROM project_folders_recycle_bin WHERE id = %s", (item_id,))
                    folder = cur.fetchone()
                    if not folder:
                        return jsonify({"error": "Folder not found"}), 404
                    restore_folder_recursive(folder, conn, cur)

                else:
                    return jsonify({"error": "Invalid source"}), 400

                conn.commit()
                return jsonify({"success": True})

@app.route('/set_image_analysis', methods=['POST'])
def set_image_analysis():
    data = request.get_json()
    enabled = data.get('enabled', True)
    session['analyze_images'] = enabled
    return jsonify({"success": True})

@app.route('/empty_recycle_bin', methods=['POST'])
def empty_recycle_bin():
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    source = data.get('source')  # 'chat', 'project_files', 'project_folders', 'all'

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                if source == 'chat' or source == 'all':
                    # Delete physical files for chat recycle bin
                    cur.execute("SELECT original_stored_path FROM recycle_bin WHERE user_id = %s", (user_id,))
                    paths = cur.fetchall()
                    for row in paths:
                        if row[0] and os.path.exists(row[0]):
                            try:
                                os.remove(row[0])
                            except:
                                pass
                    cur.execute("DELETE FROM recycle_bin WHERE user_id = %s", (user_id,))

                if source == 'project_files' or source == 'all':
                    # Delete physical files for project recycle bin
                    cur.execute("SELECT stored_path FROM project_recycle_bin")
                    paths = cur.fetchall()
                    for row in paths:
                        if row[0] and os.path.exists(row[0]):
                            try:
                                os.remove(row[0])
                            except:
                                pass
                    cur.execute("DELETE FROM project_recycle_bin")

                if source == 'project_folders' or source == 'all':
                    # Delete folder recycle bin (no physical files)
                    cur.execute("DELETE FROM project_folders_recycle_bin")

                conn.commit()
                return jsonify({"success": True})

# Scheduled job to clean expired recycle bin items and delete physical files
def cleanup_expired_recycle_bin():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Chat recycle bin
            cur.execute("SELECT original_stored_path FROM recycle_bin WHERE expires_at <= NOW()")
            paths = cur.fetchall()
            for row in paths:
                if row[0] and os.path.exists(row[0]):
                    try:
                        os.remove(row[0])
                        logger.info(f"Deleted expired recycle file: {row[0]}")
                    except Exception as e:
                        logger.warning(f"Failed to delete expired file {row[0]}: {e}")
            cur.execute("DELETE FROM recycle_bin WHERE expires_at <= NOW()")

            # Project recycle bin
            cur.execute("SELECT stored_path FROM project_recycle_bin WHERE expires_at <= NOW()")
            paths = cur.fetchall()
            for row in paths:
                if row[0] and os.path.exists(row[0]):
                    try:
                        os.remove(row[0])
                        logger.info(f"Deleted expired project recycle file: {row[0]}")
                    except Exception as e:
                        logger.warning(f"Failed to delete expired file {row[0]}: {e}")
            cur.execute("DELETE FROM project_recycle_bin WHERE expires_at <= NOW()")
            conn.commit()
            logger.info("Cleaned up expired recycle bin items")

def restore_folder_recursive(folder_item, conn, cur, target_parent_id=None):
    """Restore a folder and all its contents (subfolders and files) from project_folders_recycle_bin."""
    # Insert the folder itself
    parent_id = target_parent_id if target_parent_id is not None else folder_item['original_parent_id']
    cur.execute("""
        INSERT INTO project_folders (id, project_id, parent_folder_id, name, created_at, created_by)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """, (folder_item['original_id'], folder_item['project_id'], parent_id,
          folder_item['name'], folder_item['created_at'], folder_item['created_by']))
    # Restore all files that were in this folder
    cur.execute("""
        SELECT * FROM project_recycle_bin
        WHERE project_id = %s AND folder_id = %s
    """, (folder_item['project_id'], folder_item['original_id']))
    files = cur.fetchall()
    for f in files:
        cur.execute("""
            INSERT INTO project_files (project_id, folder_id, filename, original_name, file_size, stored_path, version, uploaded_by, file_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (f['project_id'], folder_item['original_id'], f['file_name'], f['original_name'],
              f['file_size'], f['stored_path'], f['version'], f['uploaded_by'], f['file_hash']))
        cur.execute("DELETE FROM project_recycle_bin WHERE id = %s", (f['id'],))
    # Restore subfolders recursively
    cur.execute("""
        SELECT * FROM project_folders_recycle_bin
        WHERE project_id = %s AND original_parent_id = %s
    """, (folder_item['project_id'], folder_item['original_id']))
    subfolders = cur.fetchall()
    for sf in subfolders:
        restore_folder_recursive(sf, conn, cur, target_parent_id=folder_item['original_id'])
    # Finally delete the folder from recycle bin
    cur.execute("DELETE FROM project_folders_recycle_bin WHERE id = %s", (folder_item['id'],))

def restore_folder_path_for_file(file_item, conn, cur):
    """Restore the folder hierarchy (empty folders) leading to a file, but not other files."""
    # Get the original parent folder ID
    folder_id = file_item['folder_id']
    if folder_id is None:
        # File was in root, nothing to restore
        return
    # Check if the folder already exists
    cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s", (folder_id, file_item['project_id']))
    if cur.fetchone():
        return
    # Get the folder from recycle bin
    cur.execute("SELECT * FROM project_folders_recycle_bin WHERE original_id = %s AND project_id = %s", (folder_id, file_item['project_id']))
    folder = cur.fetchone()
    if not folder:
        # Folder not in recycle bin? Should not happen if it was deleted with files.
        return
    # Recursively restore parent chain (without files)
    if folder['original_parent_id']:
        # Restore parent folder first (recursive)
        cur.execute("SELECT * FROM project_folders_recycle_bin WHERE original_id = %s AND project_id = %s", (folder['original_parent_id'], file_item['project_id']))
        parent = cur.fetchone()
        if parent:
            restore_folder_path_for_file(parent, conn, cur)  # parent will restore its folder only
    # Now restore this folder (only the folder, not its other files)
    cur.execute("""
        INSERT INTO project_folders (id, project_id, parent_folder_id, name, created_at, created_by)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """, (folder['original_id'], folder['project_id'], folder['original_parent_id'],
          folder['name'], folder['created_at'], folder['created_by']))
    # Do NOT restore any files inside this folder (only the folder itself)
    # Delete the folder from recycle bin
    cur.execute("DELETE FROM project_folders_recycle_bin WHERE id = %s", (folder['id'],))

# Scheduled jobs
def delete_expired_original_files():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, original_stored_path
                FROM user_files
                WHERE original_expires_at IS NOT NULL AND original_expires_at <= NOW()
                  AND original_stored_path IS NOT NULL
            """)
            expired = cur.fetchall()
            for file_id, original_path in expired:
                if original_path and os.path.exists(original_path):
                    try:
                        os.remove(original_path)
                        logger.info(f"Deleted expired original file: {original_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete expired file {original_path}: {e}")
                # Clear the stored path so it cannot be downloaded again
                cur.execute("UPDATE user_files SET original_stored_path = NULL WHERE id = %s", (file_id,))
            conn.commit()

def cleanup_old_anon_temp_files(days=1):
    now = time.time()
    for item in os.listdir(TEMP_ROOT):
        item_path = os.path.join(TEMP_ROOT, item)
        if os.path.isdir(item_path):
            if (now - os.path.getctime(item_path)) > days * 86400:
                shutil.rmtree(item_path)
                logger.info(f"Removed old anonymous temp dir: {item_path}")


def schedule_project_deletion_cleanup():
    cutoff = utc_now() - timedelta(days=3)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM projects WHERE status = 'archived' AND archived_at < %s", (cutoff,))
            to_delete = cur.fetchall()
            for (project_id,) in to_delete:
                logger.info(f"Auto-deleting archived project {project_id} after 3 days")
                cur.execute("DELETE FROM projects WHERE id = %s", (project_id,))
            conn.commit()

@app.route('/search_chat', methods=['GET'])
def search_chat():
    if not session.get('consent_given'):
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    q = request.args.get('q', '').strip()
    if len(q) < 2:
        return jsonify({"error": "Search query must be at least 2 characters"}), 400

    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    fuzzy = request.args.get('fuzzy', 'false').lower() == 'true'
    role = request.args.get('role', 'assistant')  # 'assistant', 'user', or 'both'

    # Build the search pattern
    if fuzzy:
        search_pattern = f"%{q}%"
    else:
        search_pattern = q  # exact match (case-insensitive via ILIKE)

    # Date filtering
    date_condition = ""
    params = [user_id, search_pattern]
    if start_date:
        date_condition += " AND cm.timestamp >= %s"
        params.append(start_date)
    if end_date:
        date_condition += " AND cm.timestamp <= %s"
        params.append(end_date)

    # Role condition
    if role == 'assistant':
        role_condition = " AND cm.role = 'assistant'"
    elif role == 'user':
        role_condition = " AND cm.role = 'user'"
    else:  # both
        role_condition = ""

    query = f"""
        SELECT cs.thread_id, cs.title, cm.role, cm.content, cm.timestamp,
               SUBSTRING(cm.content, 1, 200) as snippet
        FROM chat_messages cm
        JOIN chat_sessions cs ON cm.thread_id = cs.thread_id
        WHERE cs.user_id = %s
          AND cm.content ILIKE %s
          {role_condition}
          {date_condition}
        ORDER BY cm.timestamp DESC
        LIMIT 100
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            results = cur.fetchall()
            for row in results:
                # Convert timestamp to Beijing time for display
                if row['timestamp']:
                    row['timestamp_str'] = row['timestamp'].astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')
                # Highlight the search term (simple)
                if fuzzy:
                    # Escape regex special chars
                    import re
                    escaped = re.escape(q)
                    row['highlighted_snippet'] = re.sub(f"({escaped})", r'<mark>\1</mark>', row['snippet'], flags=re.IGNORECASE)
                else:
                    row['highlighted_snippet'] = row['snippet']
            return jsonify({"results": results})

# App startup
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=cleanup_old_sessions, trigger="interval", days=1, args=[15])
    scheduler.add_job(func=delete_expired_original_files, trigger="interval", hours=6)
    scheduler.add_job(func=cleanup_stale_tasks, trigger="interval", minutes=5)
    scheduler.add_job(func=cleanup_stale_message_responses, trigger="interval", hours=1)
    scheduler.add_job(func=cleanup_old_anon_temp_files, trigger="interval", days=1, args=[1])
    scheduler.add_job(func=schedule_project_deletion_cleanup, trigger="interval", days=1)
    scheduler.add_job(func=cleanup_expired_recycle_bin, trigger="interval", days=3)
    scheduler.add_job(func=cleanup_orphan_users, trigger="interval", days=3)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())


def shutdown_agent():
    global _agent
    _agent = None

atexit.register(shutdown_agent)

def shutdown_db_pool():
    db_pool.closeall()
    logger.info("Database pool closed.")

atexit.register(shutdown_db_pool)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    init_postgres_tables()
    cleanup_old_sessions(days=15)
    _init_async_checkpointer()
    try:
        import filelock
    except ImportError:
        logger.warning(
            "filelock not installed. Anonymous JSON file writes may have race conditions. Install with: pip install filelock")
    app.run(host='0.0.0.0', port=5000, threaded=True)