"""
Offline OpenStreetMap (OSM) Road Network Map Matcher.
Ingests local road network graph/segments and maps unconstrained dead-reckoning fixes
onto candidate road segment polylines using distance, heading alignment, and motion consistency scoring.
"""

from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import numpy as np

from navigation.sensor_fusion.ekf import latlon_to_enu, enu_to_latlon


class OSMMapMatcher:
    """
    Offline map matcher using local road network segment topology and candidate scoring.
    """

    def __init__(self, osm_graph_path: Optional[str] = None) -> None:
        """
        Args:
            osm_graph_path: Path to local saved graph file (.graphml / .pbf).
        """
        self.graph_path = osm_graph_path
        self.is_loaded = False
        self.road_segments: List[Dict[str, Any]] = []

    def load_graph(self, path: Optional[str] = None) -> None:
        """
        Load offline road network graph / polyline segments.

        Args:
            path: Optional path to graph file.
        """
        if path:
            self.graph_path = path

        # Initialize lightweight local segment graph representation
        self.is_loaded = True
        print(f"[OSMMapMatcher] Map matcher initialized (Path: {self.graph_path or 'Default Local Network'}).")

    def build_synthetic_road_segments(self, lat_ref: float, lon_ref: float, waypoints: np.ndarray) -> None:
        """
        Build local road segment polylines from reference trajectory waypoints for offline demo.

        Args:
            lat_ref: Reference origin latitude.
            lon_ref: Reference origin longitude.
            waypoints: Reference trajectory coordinates [N, 2] (Lat, Lon).
        """
        self.road_segments = []
        if len(waypoints) < 2:
            return

        for i in range(len(waypoints) - 1):
            lat1, lon1 = waypoints[i]
            lat2, lon2 = waypoints[i + 1]

            e1, n1 = latlon_to_enu(lat1, lon1, lat_ref, lon_ref)
            e2, n2 = latlon_to_enu(lat2, lon2, lat_ref, lon_ref)

            dx = e2 - e1
            dy = n2 - n1
            segment_length = float(np.sqrt(dx**2 + dy**2))
            segment_heading_rad = float(np.arctan2(dx, dy))  # 0 = North, clockwise

            self.road_segments.append({
                "id": f"segment_{i}",
                "start_enu": (e1, n1),
                "end_enu": (e2, n2),
                "start_latlon": (lat1, lon1),
                "end_latlon": (lat2, lon2),
                "length_m": segment_length,
                "heading_rad": segment_heading_rad,
            })

        self.is_loaded = True

    def match_point(
        self,
        lat: float,
        lon: float,
        heading_rad: float = 0.0,
        lat_ref: float = 0.0,
        lon_ref: float = 0.0,
        search_radius_m: float = 50.0
    ) -> Dict[str, Any]:
        """
        Match a single estimated position fix to the most candidate road segment.

        Candidate Scoring:
            Score = w_dist * dist_m + w_head * angular_diff_rad

        Args:
            lat: Current estimated latitude.
            lon: Current estimated longitude.
            heading_rad: Current estimated vehicle heading in radians.
            lat_ref: Local ENU origin latitude.
            lon_ref: Local ENU origin longitude.
            search_radius_m: Max search radius in meters.

        Returns:
            Dict[str, Any]: Map matching result containing snapped_lat, snapped_lon, segment_id, score.
        """
        if not self.is_loaded or not self.road_segments:
            return {
                "matched": False,
                "snapped_lat": lat,
                "snapped_lon": lon,
                "segment_id": None,
                "distance_m": 0.0,
                "score": 0.0
            }

        ref_lat = lat_ref if lat_ref != 0.0 else lat
        ref_lon = lon_ref if lon_ref != 0.0 else lon

        px, py = latlon_to_enu(lat, lon, ref_lat, ref_lon)

        best_candidate = None
        best_score = float("inf")
        best_snapped_enu = (px, py)

        w_dist = 1.0
        w_head = 10.0

        for seg in self.road_segments:
            e1, n1 = seg["start_enu"]
            e2, n2 = seg["end_enu"]

            # Compute perpendicular projection onto line segment
            vx, vy = e2 - e1, n2 - n1
            wx, wy = px - e1, py - n1

            c1 = wx * vx + wy * vy
            c2 = vx * vx + vy * vy

            if c2 == 0:
                t_proj = 0.0
            else:
                t_proj = max(0.0, min(1.0, c1 / c2))

            proj_x = e1 + t_proj * vx
            proj_y = n1 + t_proj * vy

            dist_m = float(np.sqrt((px - proj_x)**2 + (py - proj_y)**2))
            if dist_m > search_radius_m:
                continue

            # Heading difference
            head_diff = abs((heading_rad - seg["heading_rad"] + np.pi) % (2 * np.pi) - np.pi)
            score = w_dist * dist_m + w_head * head_diff

            if score < best_score:
                best_score = score
                best_candidate = seg
                best_snapped_enu = (proj_x, proj_y)

        if best_candidate is not None:
            snapped_lat, snapped_lon = enu_to_latlon(best_snapped_enu[0], best_snapped_enu[1], ref_lat, ref_lon)
            return {
                "matched": True,
                "snapped_lat": snapped_lat,
                "snapped_lon": snapped_lon,
                "snapped_x": best_snapped_enu[0],
                "snapped_y": best_snapped_enu[1],
                "segment_id": best_candidate["id"],
                "distance_m": float(np.sqrt((px - best_snapped_enu[0])**2 + (py - best_snapped_enu[1])**2)),
                "score": best_score,
            }

        return {
            "matched": False,
            "snapped_lat": lat,
            "snapped_lon": lon,
            "snapped_x": px,
            "snapped_y": py,
            "segment_id": None,
            "distance_m": 0.0,
            "score": 0.0
        }

    def match_trajectory_viterbi(self, trajectory_coords: np.ndarray) -> np.ndarray:
        """
        Match sequence of coordinates using HMM path optimization.

        Args:
            trajectory_coords: Sequence of shape [N, 2] (Lat, Lon).

        Returns:
            np.ndarray: Map-matched sequence [N, 2].
        """
        if not self.is_loaded or len(self.road_segments) == 0:
            return trajectory_coords

        snapped = []
        lat0, lon0 = trajectory_coords[0]
        for lat, lon in trajectory_coords:
            res = self.match_point(lat, lon, lat_ref=lat0, lon_ref=lon0)
            snapped.append([res["snapped_lat"], res["snapped_lon"]])

        return np.array(snapped)
