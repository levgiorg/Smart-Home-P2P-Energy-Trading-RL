from __future__ import annotations

"""SQLite-backed local experiment registry."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    mechanism TEXT,
    seed INTEGER,
    status TEXT DEFAULT 'running',
    start_time TEXT,
    end_time TEXT,
    gpu TEXT,
    config_json TEXT,
    git_commit TEXT,
    git_branch TEXT,
    git_dirty INTEGER,
    mean_reward REAL,
    std_reward REAL,
    best_reward REAL,
    total_episodes INTEGER,
    training_time_sec REAL,
    results_dir TEXT
)
"""


_ALLOWED_SORT_COLUMNS: frozenset[str] = frozenset(
    {"mean_reward", "best_reward", "total_episodes", "training_time_sec", "start_time"}
)


class ExperimentRegistry:
    """SQLite-backed local experiment tracker."""

    def __init__(self, db_path: str = "results/experiments.db") -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE)
            conn.commit()

    def register_run(self, config: Any, git_info: dict | None = None) -> str:
        """Register a new run and return its run_id."""
        run_id = uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        git = git_info or {}

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO runs (run_id, agent, mechanism, seed, status, start_time, config_json,
                   git_commit, git_branch, git_dirty)
                   VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    getattr(config, "agent", "unknown"),
                    getattr(getattr(config, "market", None), "mechanism_type", None),
                    getattr(config, "seed", None),
                    now,
                    config.model_dump_json()
                    if hasattr(config, "model_dump_json")
                    else json.dumps({}),
                    git.get("commit"),
                    git.get("branch"),
                    int(git.get("dirty", False)),
                ),
            )
            conn.commit()
        return run_id

    def update_status(
        self,
        run_id: str,
        status: str,
        metrics: dict | None = None,
        results_dir: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        metrics = metrics or {}
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE runs SET status=?, end_time=?, mean_reward=?, std_reward=?,
                   best_reward=?, total_episodes=?, training_time_sec=?, results_dir=?
                   WHERE run_id=?""",
                (
                    status,
                    now,
                    metrics.get("mean_reward"),
                    metrics.get("std_reward"),
                    metrics.get("best_reward"),
                    metrics.get("total_episodes"),
                    metrics.get("training_time_sec"),
                    results_dir,
                    run_id,
                ),
            )
            conn.commit()

    def list_runs(
        self,
        agent: str | None = None,
        mechanism: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM runs WHERE 1=1"
        params: list[Any] = []
        if agent:
            query += " AND agent=?"
            params.append(agent)
        if mechanism:
            query += " AND mechanism=?"
            params.append(mechanism)
        if status:
            query += " AND status=?"
            params.append(status)
        query += " ORDER BY start_time DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def best_run(self, agent: str, metric: str = "mean_reward") -> dict | None:
        if metric not in _ALLOWED_SORT_COLUMNS:
            raise ValueError(
                f"Invalid sort column: {metric!r}. Allowed: {sorted(_ALLOWED_SORT_COLUMNS)}"
            )
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f"SELECT * FROM runs WHERE agent=? AND status='completed' ORDER BY {metric} DESC LIMIT 1",  # noqa: S608
                (agent,),
            ).fetchone()
        return dict(row) if row else None

    def compare(self, run_ids: list[str]) -> list[dict]:
        placeholders = ",".join("?" * len(run_ids))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM runs WHERE run_id IN ({placeholders})", run_ids
            ).fetchall()
        return [dict(r) for r in rows]
