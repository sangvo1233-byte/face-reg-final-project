"""
Database — SQLite data access layer for the face attendance system.

Tables: students, sessions, attendance, face_embeddings
"""
import sqlite3
import io
import numpy as np
from datetime import datetime
from loguru import logger

import config


class Database:
    """SQLite: students, sessions, attendance, face_embeddings."""

    def __init__(self):
        self.db_path = str(config.SQLITE_DB_PATH)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                class_name TEXT DEFAULT '',
                photo_path TEXT,
                enrolled_at DATETIME,
                is_active BOOLEAN DEFAULT 1
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                class_name TEXT DEFAULT '',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                ended_at DATETIME,
                status TEXT DEFAULT 'active',
                total_students INTEGER DEFAULT 0,
                present_count INTEGER DEFAULT 0,
                absent_count INTEGER DEFAULT 0
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                student_id TEXT NOT NULL,
                status TEXT DEFAULT 'present',
                confidence REAL DEFAULT 0,
                scanned_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                photo_path TEXT,
                UNIQUE(session_id, student_id),
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                quality_score REAL DEFAULT 0,
                source TEXT DEFAULT 'camera',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        """)

        conn.commit()
        conn.close()
        logger.info("Database initialized")

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Students ────────────────────────────────────────────

    def add_student(self, student_id: str, name: str,
                    class_name: str = '', photo_path: str = None) -> bool:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO students (id, name, class_name, photo_path, enrolled_at) VALUES (?, ?, ?, ?, ?)",
                (student_id, name, class_name, photo_path, datetime.now().isoformat())
            )
            conn.commit()
            logger.info(f"Student added: {name} ({student_id})")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Student already exists: {student_id}")
            return False
        finally:
            conn.close()

    def update_student(self, student_id: str, **kwargs) -> bool:
        allowed = {'name', 'class_name', 'photo_path', 'is_active'}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [student_id]
        conn = self._conn()
        c = conn.execute(f"UPDATE students SET {set_clause} WHERE id = ?", values)
        conn.commit()
        conn.close()
        return c.rowcount > 0

    def get_student(self, student_id: str) -> dict | None:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM students WHERE id = ? AND is_active = 1", (student_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_student_any(self, student_id: str) -> dict | None:
        """Return student regardless of is_active status (for archived views)."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM students WHERE id = ?", (student_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_all_students(self, active_only: bool = True,
                         archived_only: bool = False) -> list[dict]:
        conn = self._conn()
        if archived_only:
            query = "SELECT * FROM students WHERE is_active = 0 ORDER BY name"
        elif active_only:
            query = "SELECT * FROM students WHERE is_active = 1 ORDER BY name"
        else:
            query = "SELECT * FROM students ORDER BY name"
        rows = conn.execute(query).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def delete_student(self, student_id: str) -> bool:
        return self.update_student(student_id, is_active=0)

    def restore_student(self, student_id: str) -> bool:
        """Restore a soft-deleted student back to active."""
        conn = self._conn()
        c = conn.execute(
            "UPDATE students SET is_active = 1 WHERE id = ? AND is_active = 0",
            (student_id,)
        )
        conn.commit()
        conn.close()
        if c.rowcount > 0:
            logger.info(f"Student restored: {student_id}")
        return c.rowcount > 0

    def get_student_count(self) -> int:
        conn = self._conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM students WHERE is_active = 1").fetchone()
        conn.close()
        return row['cnt']

    # ── Face Embeddings ─────────────────────────────────────

    def save_embedding(self, student_id: str, embedding: np.ndarray,
                       quality_score: float = 0.0, source: str = 'camera') -> int:
        buf = io.BytesIO()
        np.save(buf, embedding)
        blob = buf.getvalue()
        conn = self._conn()
        c = conn.execute(
            "INSERT INTO face_embeddings (student_id, embedding, quality_score, source) VALUES (?, ?, ?, ?)",
            (student_id, blob, quality_score, source)
        )
        conn.commit()
        conn.close()
        return c.lastrowid

    def get_all_embeddings(self) -> tuple[np.ndarray, list[dict]]:
        """Load all active embeddings and return (NxD matrix, identity list)."""
        conn = self._conn()
        rows = conn.execute("""
            SELECT fe.student_id, fe.embedding, s.name
            FROM face_embeddings fe
            JOIN students s ON fe.student_id = s.id
            WHERE fe.is_active = 1 AND s.is_active = 1
            ORDER BY fe.student_id, fe.id
        """).fetchall()
        conn.close()

        if not rows:
            return np.empty((0, config.EMBEDDING_DIM)), []

        from collections import defaultdict
        student_embeds = defaultdict(list)
        student_names = {}
        for row in rows:
            sid = row['student_id']
            buf = io.BytesIO(row['embedding'])
            emb = np.load(buf)
            student_embeds[sid].append(emb)
            student_names[sid] = row['name']

        embeddings = []
        identities = []
        for sid, emb_list in student_embeds.items():
            if len(emb_list) == 1:
                mean_emb = emb_list[0]
            else:
                mean_emb = np.mean(np.stack(emb_list), axis=0)
                norm = np.linalg.norm(mean_emb)
                if norm > 0:
                    mean_emb = mean_emb / norm
            embeddings.append(mean_emb)
            identities.append({'student_id': sid, 'name': student_names[sid]})

        return np.stack(embeddings), identities

    def delete_embeddings(self, student_id: str) -> int:
        conn = self._conn()
        c = conn.execute(
            "UPDATE face_embeddings SET is_active = 0 WHERE student_id = ?", (student_id,)
        )
        conn.commit()
        conn.close()
        return c.rowcount

    def replace_student_embeddings(
        self,
        student_id: str,
        name: str,
        class_name: str,
        embeddings: list[dict],
        photo_path: str | None = None,
    ) -> dict:
        """Atomically upsert student metadata and replace active embeddings."""
        conn = self._conn()
        try:
            conn.execute("BEGIN")
            existing = conn.execute(
                "SELECT id FROM students WHERE id = ?", (student_id,)
            ).fetchone()

            if existing:
                if photo_path is not None:
                    conn.execute(
                        """UPDATE students
                           SET name = ?, class_name = ?, photo_path = ?, is_active = 1
                           WHERE id = ?""",
                        (name, class_name, photo_path, student_id),
                    )
                else:
                    conn.execute(
                        """UPDATE students
                           SET name = ?, class_name = ?, is_active = 1
                           WHERE id = ?""",
                        (name, class_name, student_id),
                    )
            else:
                conn.execute(
                    """INSERT INTO students
                       (id, name, class_name, photo_path, enrolled_at, is_active)
                       VALUES (?, ?, ?, ?, ?, 1)""",
                    (student_id, name, class_name, photo_path, datetime.now().isoformat()),
                )

            old_count = conn.execute(
                "UPDATE face_embeddings SET is_active = 0 WHERE student_id = ? AND is_active = 1",
                (student_id,),
            ).rowcount

            saved = 0
            for item in embeddings:
                buf = io.BytesIO()
                np.save(buf, item["embedding"])
                conn.execute(
                    """INSERT INTO face_embeddings
                       (student_id, embedding, quality_score, source)
                       VALUES (?, ?, ?, ?)""",
                    (
                        student_id,
                        buf.getvalue(),
                        item.get("quality", 0.0),
                        item.get("source", "camera"),
                    ),
                )
                saved += 1

            conn.commit()
            return {"old_count": old_count, "saved": saved}
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_embedding_count(self, student_id: str = None) -> int:
        conn = self._conn()
        if student_id:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM face_embeddings WHERE student_id = ? AND is_active = 1",
                (student_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM face_embeddings WHERE is_active = 1"
            ).fetchone()
        conn.close()
        return row['cnt']

    # ── Sessions ────────────────────────────────────────────

    def create_session(self, name: str, class_name: str = '') -> int:
        """Create a new attendance session."""
        conn = self._conn()
        total = self.get_student_count()
        c = conn.execute(
            "INSERT INTO sessions (name, class_name, total_students) VALUES (?, ?, ?)",
            (name, class_name, total)
        )
        conn.commit()
        session_id = c.lastrowid
        conn.close()
        logger.info(f"Session created: #{session_id} '{name}'")
        return session_id

    def end_session(self, session_id: int) -> dict:
        """Close an active session and compute summary statistics."""
        conn = self._conn()
        # Count present students
        present = conn.execute(
            "SELECT COUNT(*) as cnt FROM attendance WHERE session_id = ?", (session_id,)
        ).fetchone()['cnt']
        session = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()

        total = session['total_students'] if session else 0
        absent = total - present

        conn.execute(
            """UPDATE sessions SET status = 'ended', ended_at = ?,
               present_count = ?, absent_count = ? WHERE id = ?""",
            (datetime.now().isoformat(), present, absent, session_id)
        )
        conn.commit()
        conn.close()
        logger.info(f"Session #{session_id} ended: {present}/{total} present")
        return {'present': present, 'absent': absent, 'total': total}

    def get_session(self, session_id: int) -> dict | None:
        conn = self._conn()
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_active_session(self) -> dict | None:
        """Return the most recent active session, or None."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE status = 'active' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_all_sessions(self, limit: int = 50) -> list[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── Attendance ──────────────────────────────────────────

    def mark_attendance(self, session_id: int, student_id: str,
                        confidence: float = 0.0, photo_path: str = None) -> dict:
        """Mark a student as present in the given session."""
        conn = self._conn()
        # Check if already marked in this session
        existing = conn.execute(
            "SELECT * FROM attendance WHERE session_id = ? AND student_id = ?",
            (session_id, student_id)
        ).fetchone()
        if existing:
            conn.close()
            return {'success': False, 'message': 'Already marked present'}

        conn.execute(
            """INSERT INTO attendance (session_id, student_id, status, confidence, photo_path)
               VALUES (?, ?, 'present', ?, ?)""",
            (session_id, student_id, confidence, photo_path)
        )
        # Update present_count on the session
        conn.execute(
            "UPDATE sessions SET present_count = present_count + 1 WHERE id = ?",
            (session_id,)
        )
        conn.commit()
        conn.close()

        student = self.get_student(student_id)
        name = student['name'] if student else student_id
        logger.info(f"Attendance marked: {name} in session #{session_id}")
        return {'success': True, 'message': f'{name} - Present'}

    def get_session_attendance(self, session_id: int) -> list[dict]:
        """Return all attendance records for a given session."""
        conn = self._conn()
        rows = conn.execute("""
            SELECT a.*, s.name as student_name, s.class_name
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.session_id = ?
            ORDER BY a.scanned_at
        """, (session_id,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_session_result(self, session_id: int) -> dict:
        """Return present and absent student lists for a session.

        Uses a snapshot of students enrolled at or before the session's
        creation time, so historical reports are not affected by later
        student additions or deletions.
        """
        conn = self._conn()

        # Get session creation timestamp
        session_row = conn.execute(
            "SELECT created_at FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session_row:
            conn.close()
            return {'present': [], 'absent': [], 'present_count': 0,
                    'absent_count': 0, 'total': 0}

        created_at = session_row['created_at']

        # Students who were enrolled at or before session creation (snapshot)
        snapshot_rows = conn.execute(
            """SELECT * FROM students
               WHERE enrolled_at <= ? AND is_active = 1
               ORDER BY name""",
            (created_at,)
        ).fetchall()
        snapshot = [dict(r) for r in snapshot_rows]

        # Who actually attended this session
        attended_ids = set(
            r['student_id'] for r in conn.execute(
                "SELECT student_id FROM attendance WHERE session_id = ?",
                (session_id,)
            ).fetchall()
        )
        conn.close()

        present = [s for s in snapshot if s['id'] in attended_ids]
        absent  = [s for s in snapshot if s['id'] not in attended_ids]
        return {
            'present': present,
            'absent': absent,
            'present_count': len(present),
            'absent_count': len(absent),
            'total': len(snapshot)
        }

    def get_student_history(self, student_id: str, limit: int = 50) -> list[dict]:
        """Return attendance history for a single student."""
        conn = self._conn()
        rows = conn.execute("""
            SELECT a.*, se.name as session_name, se.created_at as session_date
            FROM attendance a
            JOIN sessions se ON a.session_id = se.id
            WHERE a.student_id = ?
            ORDER BY a.scanned_at DESC LIMIT ?
        """, (student_id, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]


# ── Singleton ───────────────────────────────────────────────

_db = None

def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db
