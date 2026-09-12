"""
Automated Test Suite for Real Smartphone Sensor Collection,
Database Ingestion, AI Inference Pipeline, and GNSS Outage Simulator.
"""

import time
import math
import numpy as np

from backend.database import (
    init_db,
    create_or_get_session,
    stop_session,
    insert_sensor_telemetry_batch,
    insert_navigation_estimate,
    list_sessions,
    get_session_telemetry
)
from backend.live_pipeline import LiveNavigationPipeline


def test_database_schema_and_batch_ingestion():
    """Verify database creates tables and stores all 20 required sensor channels."""
    init_db()
    session_id = f"IDR_UNIT_TEST_{int(time.time())}"
    sess = create_or_get_session(session_id, device_id="SMARTPHONE_TEST_DEVICE")
    assert sess["session_id"] == session_id
    assert sess["status"] == "ACTIVE"

    # Prepare batch of real readings with all 20 fields
    sample_readings = []
    base_time = time.time()
    for i in range(10):
        sample_readings.append({
            "timestamp": base_time + i * 0.02,
            "accelerometer_x": 0.05 * math.sin(i),
            "accelerometer_y": 0.85 + 0.1 * math.cos(i),
            "accelerometer_z": 9.81,
            "gyroscope_x": 0.01,
            "gyroscope_y": 0.005,
            "gyroscope_z": 0.04,
            "magnetometer_x": 12.5,
            "magnetometer_y": 8.4,
            "magnetometer_z": 36.2,
            "heading": 45.0,
            "latitude": 12.971600 + i * 0.00001,
            "longitude": 77.594600 + i * 0.00001,
            "gps_speed": 8.5,
            "gps_accuracy": 3.5,
            "altitude": 920.0,
            "pitch": 2.1,
            "roll": 1.4,
            "yaw": 45.0
        })

    inserted = insert_sensor_telemetry_batch(session_id, sample_readings)
    assert inserted == 10

    # Retrieve and verify all columns
    stored = get_session_telemetry(session_id, limit=20)
    assert len(stored) == 10
    first = stored[0]
    required_cols = [
        "sensor_id", "session_id", "timestamp",
        "accelerometer_x", "accelerometer_y", "accelerometer_z",
        "gyroscope_x", "gyroscope_y", "gyroscope_z",
        "magnetometer_x", "magnetometer_y", "magnetometer_z",
        "heading", "latitude", "longitude",
        "gps_speed", "gps_accuracy", "altitude",
        "pitch", "roll", "yaw"
    ]
    for col in required_cols:
        assert col in first.keys(), f"Missing required column: {col}"

    # Stop session
    stopped = stop_session(session_id)
    assert stopped["status"] == "COMPLETED"
    assert stopped["sample_count"] == 10


def test_live_pipeline_stationary_zupt():
    """Verify zero-velocity detection locks speed and calibrates bias."""
    pipeline = LiveNavigationPipeline(session_id="IDR_TEST_ZUPT")
    pipeline.reset()

    # Stationary readings (gravity only, minimal gyro)
    for i in range(15):
        sample = {
            "timestamp": 100.0 + i * 0.02,
            "accelerometer_x": 0.01,
            "accelerometer_y": 0.01,
            "accelerometer_z": 9.81,
            "gyroscope_x": 0.001,
            "gyroscope_y": 0.001,
            "gyroscope_z": 0.002,
            "heading": 0.0,
            "latitude": 12.9716,
            "longitude": 77.5946,
            "gps_speed": 0.0
        }
        res = pipeline.process_live_sample(sample)

    assert res["motion_state"] == "STATIONARY"
    assert res["predicted_velocity_mps"] == 0.0
    assert res["predicted_velocity_kmh"] == 0.0


def test_live_pipeline_driving_and_outage_simulation():
    """Verify dynamic driving, Dead Reckoning propagation, and outage modes."""
    pipeline = LiveNavigationPipeline(session_id="IDR_TEST_OUTAGE")
    pipeline.reset()

    # Phase 1: Mode 1 - GNSS Available driving
    for i in range(25):
        sample = {
            "timestamp": 200.0 + i * 0.05,
            "accelerometer_x": 0.1,
            "accelerometer_y": 1.2,
            "accelerometer_z": 9.81,
            "gyroscope_x": 0.0,
            "gyroscope_y": 0.0,
            "gyroscope_z": 0.02,
            "heading": 90.0,
            "latitude": 12.9716 + i * 0.00005,
            "longitude": 77.5946 + i * 0.00005,
            "gps_speed": 10.0
        }
        res = pipeline.process_live_sample(sample)

    assert res["mode"] == "GNSS+INS"
    assert res["is_outage"] is False
    assert res["predicted_velocity_kmh"] > 0

    # Phase 2: Trigger GNSS Outage -> Mode 2 (Dead Reckoning)
    outage_info = pipeline.set_gnss_outage(True)
    assert outage_info["mode"] == "DEAD_RECKONING"

    prev_lat = res["estimated_lat"]
    prev_lon = res["estimated_lon"]

    # Continue driving with GPS unavailable
    for i in range(20):
        sample = {
            "timestamp": 201.25 + i * 0.05,
            "accelerometer_x": 0.05,
            "accelerometer_y": 1.0,
            "accelerometer_z": 9.81,
            "gyroscope_x": 0.0,
            "gyroscope_y": 0.0,
            "gyroscope_z": 0.01,
            "heading": 90.0,
            "latitude": None,
            "longitude": None,
            "gps_speed": None
        }
        res_dr = pipeline.process_live_sample(sample)

    assert res_dr["mode"] == "DEAD_RECKONING"
    assert res_dr["is_outage"] is True
    # Position must have advanced despite no GPS
    assert res_dr["estimated_lat"] != prev_lat or res_dr["estimated_lon"] != prev_lon
    assert res_dr["drift_error_m"] >= 0.0

    # Phase 3: Restore GNSS
    restore_info = pipeline.set_gnss_outage(False)
    assert restore_info["mode"] == "GNSS_RECOVERED"
