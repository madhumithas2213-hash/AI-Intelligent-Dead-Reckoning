"""
Database Module for AI-IDR Telemetry and Navigation System.
Manages persistent storage of real smartphone sensor readings, navigation sessions,
and AI-IDR Dead Reckoning estimates using SQLite with WAL (Write-Ahead Logging).
"""

import sqlite3
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Default database file path in the project backend or root
DB_PATH = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "idr_telemetry.db")))


def get_db_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    Establish a connection to the SQLite database with WAL mode enabled
    for high concurrency and rapid write throughput.
    """
    conn = sqlite3.connect(str(db_path), timeout=20.0)
    conn.row_factory = sqlite3.Row
    # High-performance PRAGMA settings
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 10000;")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """
    Initialize tables for sessions, sensor telemetry, and navigation estimates.
    Strictly adheres to all required schema fields.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # 1. Navigation Sessions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS navigation_sessions (
            session_id TEXT PRIMARY KEY,
            device_id TEXT DEFAULT 'SMARTPHONE_CLIENT',
            start_timestamp REAL NOT NULL,
            end_timestamp REAL,
            sample_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'ACTIVE', -- ACTIVE, COMPLETED, ABORTED
            notes TEXT
        );
    """)

    # 2. Real Sensor Telemetry Table (All 20 mandatory sensor channels)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_telemetry (
            sensor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            accelerometer_x REAL,
            accelerometer_y REAL,
            accelerometer_z REAL,
            gyroscope_x REAL,
            gyroscope_y REAL,
            gyroscope_z REAL,
            magnetometer_x REAL,
            magnetometer_y REAL,
            magnetometer_z REAL,
            heading REAL,
            latitude REAL,
            longitude REAL,
            gps_speed REAL,
            gps_accuracy REAL,
            altitude REAL,
            pitch REAL,
            roll REAL,
            yaw REAL,
            FOREIGN KEY (session_id) REFERENCES navigation_sessions (session_id) ON DELETE CASCADE
        );
    """)

    # Indexes for fast time-series queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_session_time ON sensor_telemetry (session_id, timestamp);")

    # 3. Live Navigation Estimates Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS navigation_estimates (
            estimate_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            predicted_velocity REAL,
            motion_state TEXT,
            estimated_lat REAL,
            estimated_lon REAL,
            mode TEXT, -- 'GNSS+INS', 'DEAD_RECKONING', 'GNSS_RECOVERED'
            is_outage INTEGER DEFAULT 0,
            drift_error_m REAL DEFAULT 0.0,
            confidence_pct REAL DEFAULT 95.0,
            FOREIGN KEY (session_id) REFERENCES navigation_sessions (session_id) ON DELETE CASCADE
        );
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_estimates_session_time ON navigation_estimates (session_id, timestamp);")

    conn.commit()
    conn.close()


def create_or_get_session(session_id: str, device_id: str = "SMARTPHONE_CLIENT") -> Dict[str, Any]:
    """Create a new session or return existing one."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = time.time()

    cursor.execute("SELECT * FROM navigation_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    if row is None:
        cursor.execute("""
            INSERT INTO navigation_sessions (session_id, device_id, start_timestamp, status)
            VALUES (?, ?, ?, 'ACTIVE')
        """, (session_id, device_id, now))
        conn.commit()
        session_data = {
            "session_id": session_id,
            "device_id": device_id,
            "start_timestamp": now,
            "status": "ACTIVE",
            "sample_count": 0
        }
    else:
        session_data = dict(row)

    conn.close()
    return session_data


def stop_session(session_id: str) -> Dict[str, Any]:
    """Mark session as completed and update sample count."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = time.time()

    cursor.execute("SELECT COUNT(*) as count FROM sensor_telemetry WHERE session_id = ?", (session_id,))
    count_row = cursor.fetchone()
    sample_count = count_row["count"] if count_row else 0

    cursor.execute("""
        UPDATE navigation_sessions
        SET end_timestamp = ?, sample_count = ?, status = 'COMPLETED'
        WHERE session_id = ?
    """, (now, sample_count, session_id))
    conn.commit()
    conn.close()

    return {
        "session_id": session_id,
        "end_timestamp": now,
        "sample_count": sample_count,
        "status": "COMPLETED"
    }


def insert_sensor_telemetry_batch(session_id: str, readings: List[Dict[str, Any]]) -> int:
    """
    Insert a batch of real smartphone sensor readings efficiently.
    Uses executemany to prevent database lock contention.
    """
    if not readings:
        return 0

    conn = get_db_connection()
    cursor = conn.cursor()

    rows = []
    for r in readings:
        rows.append((
            session_id,
            float(r.get("timestamp", time.time())),
            r.get("accelerometer_x"),
            r.get("accelerometer_y"),
            r.get("accelerometer_z"),
            r.get("gyroscope_x"),
            r.get("gyroscope_y"),
            r.get("gyroscope_z"),
            r.get("magnetometer_x"),
            r.get("magnetometer_y"),
            r.get("magnetometer_z"),
            r.get("heading"),
            r.get("latitude"),
            r.get("longitude"),
            r.get("gps_speed"),
            r.get("gps_accuracy"),
            r.get("altitude"),
            r.get("pitch"),
            r.get("roll"),
            r.get("yaw"),
        ))

    cursor.executemany("""
        INSERT INTO sensor_telemetry (
            session_id, timestamp,
            accelerometer_x, accelerometer_y, accelerometer_z,
            gyroscope_x, gyroscope_y, gyroscope_z,
            magnetometer_x, magnetometer_y, magnetometer_z,
            heading, latitude, longitude,
            gps_speed, gps_accuracy, altitude,
            pitch, roll, yaw
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    # Increment sample count
    cursor.execute("""
        UPDATE navigation_sessions
        SET sample_count = sample_count + ?
        WHERE session_id = ?
    """, (len(rows), session_id))

    conn.commit()
    conn.close()
    return len(rows)


def insert_navigation_estimate(
    session_id: str,
    timestamp: float,
    predicted_velocity: float,
    motion_state: str,
    estimated_lat: float,
    estimated_lon: float,
    mode: str,
    is_outage: bool = False,
    drift_error_m: float = 0.0,
    confidence_pct: float = 95.0
) -> None:
    """Record an AI-IDR navigation estimate."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO navigation_estimates (
            session_id, timestamp, predicted_velocity, motion_state,
            estimated_lat, estimated_lon, mode, is_outage, drift_error_m, confidence_pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id, timestamp, predicted_velocity, motion_state,
        estimated_lat, estimated_lon, mode, 1 if is_outage else 0, drift_error_m, confidence_pct
    ))
    conn.commit()
    conn.close()


def list_sessions() -> List[Dict[str, Any]]:
    """List all navigation sessions with sample counts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT session_id, device_id, start_timestamp, end_timestamp, sample_count, status, notes
        FROM navigation_sessions
        ORDER BY start_timestamp DESC
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_session_telemetry(session_id: str, limit: int = 2000) -> List[Dict[str, Any]]:
    """Retrieve time-series telemetry for a specific session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM sensor_telemetry
        WHERE session_id = ?
        ORDER BY timestamp ASC
        LIMIT ?
    """, (session_id, limit))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# Ensure database is initialized on import
init_db()
