"""
modules/database.py
SQLite tabanlı kapsamlı loglama sistemi.
Tüm işlemler (search, DSPy, LLM) burada loglanır.
"""

import sqlite3
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool
import logging

logger = logging.getLogger(__name__)

DB_PATH = Path("logs/studio.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


class Base(DeclarativeBase):
    pass


class SessionLog(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True)
    model_name = Column(String(256))
    model_type = Column(String(64))
    created_at = Column(DateTime, default=datetime.utcnow)


class SearchLog(Base):
    __tablename__ = "search_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    original_prompt = Column(Text)
    search_query = Column(Text)
    raw_results = Column(Text)       # JSON
    ranked_results = Column(Text)    # JSON
    relevance_scores = Column(Text)  # JSON
    num_results = Column(Integer)
    duration_ms = Column(Float)


class DSPyLog(Base):
    __tablename__ = "dspy_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    original_prompt = Column(Text)
    detected_mode = Column(String(128))
    mode_reason = Column(Text)
    enriched_prompt = Column(Text)
    dspy_steps = Column(Text)       # JSON - DSPy iç işlemleri
    duration_ms = Column(Float)


class LLMLog(Base):
    __tablename__ = "llm_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_name = Column(String(256))
    model_type = Column(String(64))
    # Parametreler
    temperature = Column(Float)
    max_tokens = Column(Integer)
    top_p = Column(Float)
    top_k = Column(Integer)
    repetition_penalty = Column(Float)
    # Mesajlar
    system_prompt = Column(Text)
    final_prompt = Column(Text)     # DSPy sonrası zenginleştirilmiş
    raw_prompt = Column(Text)       # Orijinal
    response = Column(Text)
    # Metrikler
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    duration_ms = Column(Float)
    tokens_per_second = Column(Float)


class ErrorLog(Base):
    __tablename__ = "error_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    module = Column(String(128))
    error_type = Column(String(128))
    error_message = Column(Text)
    stack_trace = Column(Text)
    context = Column(Text)          # JSON - hata anındaki bağlam


class GeneralLog(Base):
    __tablename__ = "general_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String(16))       # INFO, DEBUG, WARNING
    module = Column(String(128))
    message = Column(Text)
    data = Column(Text)              # JSON - ek veri


class DatabaseManager:
    """Thread-safe SQLite veritabanı yöneticisi."""

    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"DatabaseManager başlatıldı: {db_path}")

    def _get_session(self) -> Session:
        return self.SessionLocal()

    # ---------- SESSION ----------
    def log_session(self, session_id: str, model_name: str, model_type: str):
        try:
            with self._get_session() as session:
                session.add(SessionLog(
                    session_id=session_id,
                    model_name=model_name,
                    model_type=model_type
                ))
                session.commit()
        except Exception as e:
            logger.error(f"Session log hatası: {e}")

    # ---------- SEARCH ----------
    def log_search(self, session_id: str, original_prompt: str, search_query: str,
                   raw_results: list, ranked_results: list, relevance_scores: list,
                   duration_ms: float):
        try:
            with self._get_session() as session:
                session.add(SearchLog(
                    session_id=session_id,
                    original_prompt=original_prompt,
                    search_query=search_query,
                    raw_results=json.dumps(raw_results, ensure_ascii=False),
                    ranked_results=json.dumps(ranked_results, ensure_ascii=False),
                    relevance_scores=json.dumps(relevance_scores, ensure_ascii=False),
                    num_results=len(ranked_results),
                    duration_ms=duration_ms
                ))
                session.commit()
        except Exception as e:
            logger.error(f"Search log hatası: {e}")

    # ---------- DSPY ----------
    def log_dspy(self, session_id: str, original_prompt: str, detected_mode: str,
                 mode_reason: str, enriched_prompt: str, dspy_steps: list,
                 duration_ms: float):
        try:
            with self._get_session() as session:
                session.add(DSPyLog(
                    session_id=session_id,
                    original_prompt=original_prompt,
                    detected_mode=detected_mode,
                    mode_reason=mode_reason,
                    enriched_prompt=enriched_prompt,
                    dspy_steps=json.dumps(dspy_steps, ensure_ascii=False),
                    duration_ms=duration_ms
                ))
                session.commit()
        except Exception as e:
            logger.error(f"DSPy log hatası: {e}")

    # ---------- LLM ----------
    def log_llm(self, session_id: str, model_name: str, model_type: str,
                params: dict, system_prompt: str, final_prompt: str,
                raw_prompt: str, response: str, input_tokens: int,
                output_tokens: int, duration_ms: float, tokens_per_second: float):
        try:
            with self._get_session() as session:
                session.add(LLMLog(
                    session_id=session_id,
                    model_name=model_name,
                    model_type=model_type,
                    temperature=params.get("temperature"),
                    max_tokens=params.get("max_tokens"),
                    top_p=params.get("top_p"),
                    top_k=params.get("top_k"),
                    repetition_penalty=params.get("repetition_penalty"),
                    system_prompt=system_prompt,
                    final_prompt=final_prompt,
                    raw_prompt=raw_prompt,
                    response=response,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    tokens_per_second=tokens_per_second,
                ))
                session.commit()
        except Exception as e:
            logger.error(f"LLM log hatası: {e}")

    # ---------- ERROR ----------
    def log_error(self, session_id: str, module: str, error: Exception, context: dict = None):
        try:
            with self._get_session() as session:
                session.add(ErrorLog(
                    session_id=session_id,
                    module=module,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    stack_trace=traceback.format_exc(),
                    context=json.dumps(context or {}, ensure_ascii=False, default=str)
                ))
                session.commit()
        except Exception as e:
            logger.error(f"Error log hatası: {e}")

    # ---------- GENERAL ----------
    def log_general(self, session_id: str, level: str, module: str,
                    message: str, data: Any = None):
        try:
            with self._get_session() as session:
                session.add(GeneralLog(
                    session_id=session_id,
                    level=level,
                    module=module,
                    message=message,
                    data=json.dumps(data, ensure_ascii=False, default=str) if data else None
                ))
                session.commit()
        except Exception as e:
            logger.error(f"General log hatası: {e}")

    # ---------- QUERY ----------
    def get_all_logs(self, session_id: Optional[str] = None, limit: int = 500) -> dict:
        """Tüm log tablolarından veri çeker."""
        result = {}
        tables = {
            "search": SearchLog,
            "dspy": DSPyLog,
            "llm": LLMLog,
            "errors": ErrorLog,
            "general": GeneralLog,
        }
        try:
            with self._get_session() as session:
                for name, model in tables.items():
                    q = session.query(model)
                    if session_id:
                        q = q.filter(model.session_id == session_id)
                    rows = q.order_by(model.timestamp.desc()).limit(limit).all()
                    result[name] = [
                        {c.name: getattr(r, c.name) for c in model.__table__.columns}
                        for r in rows
                    ]
        except Exception as e:
            logger.error(f"Log sorgu hatası: {e}")
        return result

    def clear_logs(self, table: str = "all"):
        """Logları siler."""
        tables_map = {
            "search": SearchLog,
            "dspy": DSPyLog,
            "llm": LLMLog,
            "errors": ErrorLog,
            "general": GeneralLog,
            "sessions": SessionLog,
        }
        try:
            with self._get_session() as session:
                if table == "all":
                    for m in tables_map.values():
                        session.query(m).delete()
                elif table in tables_map:
                    session.query(tables_map[table]).delete()
                session.commit()
            return True
        except Exception as e:
            logger.error(f"Log silme hatası: {e}")
            return False

    def get_stats(self) -> dict:
        """Veritabanı istatistikleri."""
        try:
            with self._get_session() as session:
                return {
                    "total_searches": session.query(SearchLog).count(),
                    "total_llm_calls": session.query(LLMLog).count(),
                    "total_errors": session.query(ErrorLog).count(),
                    "total_dspy_calls": session.query(DSPyLog).count(),
                    "db_size_mb": round(Path(self.db_path).stat().st_size / 1024 / 1024, 2)
                    if Path(self.db_path).exists() else 0
                }
        except Exception as e:
            logger.error(f"Stats hatası: {e}")
            return {}
