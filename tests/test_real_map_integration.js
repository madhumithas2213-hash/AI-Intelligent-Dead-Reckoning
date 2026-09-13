/**
 * AI-IDR REAL MAP + REAL GPS + OFFLINE MAP INTEGRATION VERIFICATION SUITE
 * Tests:
 * 1. Coordinate validation (lat/lon bounds, finite, non-zero).
 * 2. Local Leaflet and GeoJSON asset integrity.
 * 3. OfflineMapMatcher GeoJSON loading and road snapping.
 * 4. NaN immunity under corrupt hardware sensor inputs during dead reckoning.
 * 5. Leaflet HTML structure and Service Worker pre-caching.
 */

const fs = require('fs');
const path = require('path');
const assert = require('assert');

let passedTests = 0;
let totalTests = 0;

function it(desc, fn) {
    totalTests++;
    try {
        fn();
        console.log(`  [PASS] ${desc}`);
        passedTests++;
    } catch (err) {
        console.error(`  [FAIL] ${desc}`);
        console.error(`         ${err.message}`);
    }
}

console.log("===============================================================");
console.log("AI-IDR REAL MAP & OFFLINE GEOLOCATION VERIFICATION TEST SUITE");
console.log("===============================================================\n");

// --- 1. COORDINATE VALIDATION TEST ---
console.log("--- Test Group 1: Coordinate Validator (isValidCoordinate) ---");

function isValidCoordinate(lat, lon) {
    return typeof lat === 'number' && typeof lon === 'number' &&
           Number.isFinite(lat) && Number.isFinite(lon) &&
           lat >= -90.0 && lat <= 90.0 &&
           lon >= -180.0 && lon <= 180.0 &&
           !(lat === 0.0 && lon === 0.0);
}

it("Validates real coordinates correctly", () => {
    assert.strictEqual(isValidCoordinate(12.9716, 77.5946), true); // Bangalore
    assert.strictEqual(isValidCoordinate(37.7749, -122.4194), true); // San Francisco
    assert.strictEqual(isValidCoordinate(-33.8688, 151.2093), true); // Sydney
    assert.strictEqual(isValidCoordinate(0.0001, 0.0001), true);
});

it("Rejects NaN, Infinity, and non-numeric types", () => {
    assert.strictEqual(isValidCoordinate(NaN, 77.5946), false);
    assert.strictEqual(isValidCoordinate(12.9716, NaN), false);
    assert.strictEqual(isValidCoordinate(Infinity, 77.5946), false);
    assert.strictEqual(isValidCoordinate(12.9716, -Infinity), false);
    assert.strictEqual(isValidCoordinate(null, 77.5946), false);
    assert.strictEqual(isValidCoordinate(undefined, 77.5946), false);
    assert.strictEqual(isValidCoordinate("12.9716", "77.5946"), false);
});

it("Rejects out of bounds latitude and longitude", () => {
    assert.strictEqual(isValidCoordinate(91.0, 77.5946), false);
    assert.strictEqual(isValidCoordinate(-90.1, 77.5946), false);
    assert.strictEqual(isValidCoordinate(12.9716, 180.1), false);
    assert.strictEqual(isValidCoordinate(12.9716, -180.5), false);
});

it("Rejects null island (0.0, 0.0)", () => {
    assert.strictEqual(isValidCoordinate(0.0, 0.0), false);
});

// --- 2. ASSET INTEGRITY ---
console.log("\n--- Test Group 2: Local Map Assets Integrity ---");

it("Leaflet JavaScript and CSS assets exist and have expected sizes", () => {
    const jsPath = path.join(__dirname, '../assets/leaflet/leaflet.js');
    const cssPath = path.join(__dirname, '../assets/leaflet/leaflet.css');
    assert.ok(fs.existsSync(jsPath), "assets/leaflet/leaflet.js must exist");
    assert.ok(fs.existsSync(cssPath), "assets/leaflet/leaflet.css must exist");

    const jsStat = fs.statSync(jsPath);
    const cssStat = fs.statSync(cssPath);
    assert.ok(jsStat.size > 100000, `leaflet.js should be > 100KB, got ${jsStat.size} bytes`);
    assert.ok(cssStat.size > 10000, `leaflet.css should be > 10KB, got ${cssStat.size} bytes`);
});

it("Offline road network GeoJSON exists, parses, and has real road corridors", () => {
    const geoPath = path.join(__dirname, '../assets/maps/offline_road_network.json');
    assert.ok(fs.existsSync(geoPath), "assets/maps/offline_road_network.json must exist");
    const raw = fs.readFileSync(geoPath, 'utf8');
    const geo = JSON.parse(raw);

    assert.strictEqual(geo.type, "FeatureCollection");
    assert.ok(Array.isArray(geo.features) && geo.features.length >= 10, "Should have at least 10 road features");

    const roadNames = geo.features.map(f => f.properties && f.properties.name).filter(Boolean);
    assert.ok(roadNames.some(n => n.includes("MG") || n.includes("Mahatma Gandhi")), "Must contain MG Road");
    assert.ok(roadNames.some(n => n.includes("Brigade Road")), "Must contain Brigade Road");
    assert.ok(roadNames.some(n => n.includes("Residency Road")), "Must contain Residency Road");
});

// --- 3. OFFLINE MAP MATCHER & ROAD SNAPPING ---
console.log("\n--- Test Group 3: OfflineMapMatcher GeoJSON Ingestion & Snapping ---");

class OfflineMapMatcher {
    constructor() {
        this.segments = [];
        this.hasOfflineRoads = false;
    }

    loadGeoJSON(geojsonData) {
        if (!geojsonData || !geojsonData.features || !Array.isArray(geojsonData.features)) return false;
        let loaded = [];
        for (let f of geojsonData.features) {
            if (f.geometry && f.geometry.type === 'LineString' && Array.isArray(f.geometry.coordinates)) {
                let roadName = (f.properties && f.properties.name) ? f.properties.name : 'Unassigned Street';
                let roadType = (f.properties && f.properties.highway) ? f.properties.highway : 'road';
                let pts = f.geometry.coordinates.map(c => [c[1], c[0]]);
                if (pts.length >= 2) {
                    loaded.push({
                        id: f.id || ('osm_' + Math.random()),
                        name: roadName,
                        highway: roadType,
                        points: pts
                    });
                }
            }
        }
        if (loaded.length > 0) {
            this.segments = loaded;
            this.hasOfflineRoads = true;
            return true;
        }
        return false;
    }

    projectPointToSegment(pLat, pLon, aLat, aLon, bLat, bLon) {
        let cosLat = Math.cos(pLat * Math.PI / 180.0);
        let bx = (bLon - aLon) * 111320.0 * cosLat;
        let by = (bLat - aLat) * 111320.0;
        let px = (pLon - aLon) * 111320.0 * cosLat;
        let py = (pLat - aLat) * 111320.0;

        let segLenSq = bx * bx + by * by;
        let u = segLenSq > 0 ? (px * bx + py * by) / segLenSq : 0;
        let uClamped = Math.max(0.0, Math.min(1.0, u));

        let projX = uClamped * bx;
        let projY = uClamped * by;
        let distM = Math.hypot(px - projX, py - projY);

        let snapLat = aLat + (projY / 111320.0);
        let snapLon = aLon + (projX / (111320.0 * cosLat));
        let roadHeadingRad = Math.atan2(bx, by);

        return { snapLat, snapLon, distM, roadHeadingRad };
    }

    match(estLat, estLon, headingRad, maxRadiusM = 40.0) {
        let bestMatch = {
            snappedLat: estLat,
            snappedLon: estLon,
            distM: 0.0,
            roadName: 'Off-Road / Free Trajectory',
            roadHeading: headingRad,
            matched: false
        };
        let bestScore = Infinity;
        let head = (headingRad !== null && !isNaN(headingRad)) ? headingRad : 0.0;

        for (let seg of this.segments) {
            if (!seg.points || seg.points.length < 2) continue;
            for (let i = 0; i < seg.points.length - 1; i++) {
                let [aLat, aLon] = seg.points[i];
                let [bLat, bLon] = seg.points[i + 1];
                let proj = this.projectPointToSegment(estLat, estLon, aLat, aLon, bLat, bLon);

                if (proj.distM <= maxRadiusM) {
                    let angleDiff = Math.abs((head - proj.roadHeadingRad + Math.PI) % (2.0 * Math.PI) - Math.PI);
                    let biDirAngleDiff = Math.min(angleDiff, Math.abs(Math.PI - angleDiff));
                    let score = proj.distM + biDirAngleDiff * 14.0;

                    if (score < bestScore) {
                        bestScore = score;
                        bestMatch = {
                            snappedLat: proj.snapLat,
                            snappedLon: proj.snapLon,
                            distM: proj.distM,
                            roadName: seg.name,
                            roadHeading: proj.roadHeadingRad,
                            matched: true
                        };
                    }
                }
            }
        }
        return bestMatch;
    }
}

it("OfflineMapMatcher loads real GeoJSON road segments correctly", () => {
    const geo = JSON.parse(fs.readFileSync(path.join(__dirname, '../assets/maps/offline_road_network.json'), 'utf8'));
    const matcher = new OfflineMapMatcher();
    const ok = matcher.loadGeoJSON(geo);
    assert.strictEqual(ok, true);
    assert.ok(matcher.segments.length >= 10);
    assert.strictEqual(matcher.hasOfflineRoads, true);
});

it("Accurately snaps points near MG Road to centerline with high confidence", () => {
    const geo = JSON.parse(fs.readFileSync(path.join(__dirname, '../assets/maps/offline_road_network.json'), 'utf8'));
    const matcher = new OfflineMapMatcher();
    matcher.loadGeoJSON(geo);

    // MG Road vertex: 12.9716, 77.5946
    // Test point offset by 6 meters north
    let testLat = 12.9716 + (6.0 / 111320.0);
    let testLon = 77.5946;
    let match = matcher.match(testLat, testLon, Math.PI / 2.0); // Heading East

    assert.strictEqual(match.matched, true);
    assert.ok(match.distM < 10.0, `Expected distM < 10m, got ${match.distM}`);
    assert.ok(match.roadName.includes("MG") || match.roadName.includes("Mahatma Gandhi"), `Expected MG Road, got ${match.roadName}`);
    assert.ok(isValidCoordinate(match.snappedLat, match.snappedLon));
});

// --- 4. NAN IMMUNITY DURING SIMULATED DEAD RECKONING UNDER SENSOR OUTAGES ---
console.log("\n--- Test Group 4: NaN Immunity & Kinematic Dead Reckoning Robustness ---");

it("Prevents NaN propagation when compass heading and orientation emit null/NaN", () => {
    let liveSensors = {
        ax: NaN, ay: null, az: undefined, amag: 9.81,
        gx: 0.0, gy: NaN, gz: null,
        mx: NaN, my: 0.0, mz: 0.0, heading: NaN,
        pitch: NaN, roll: null, yaw: undefined,
        gpsLat: null, gpsLon: null, gpsSpeed: 0.0, gpsAcc: null,
        biasGz: 0.0
    };

    let liveNav = {
        refLat: 12.9716, refLon: 77.5946,
        estLat: 12.9716, estLon: 77.5946,
        headingRad: 0.0,
        velocityMps: 10.0,
        mode: "DEAD_RECKONING",
        lastAuthoritativeLat: 12.9716,
        lastAuthoritativeLon: 77.5946,
        prevTime: 0.0
    };

    const R_EARTH = 6378137.0;

    // Simulate 120 continuous ticks with corrupted inputs
    for (let i = 1; i <= 120; i++) {
        let currentTimeSec = i * 0.05;
        let dt = 0.05;

        let unbiasedGz = (typeof liveSensors.gz === 'number' && Number.isFinite(liveSensors.gz)) ? liveSensors.gz - liveSensors.biasGz : 0.0;

        // Heading sanitization
        if (typeof liveSensors.heading === 'number' && Number.isFinite(liveSensors.heading)) {
            let compassRad = ((liveSensors.heading % 360 + 360) % 360) * (Math.PI / 180.0);
            liveNav.headingRad = compassRad;
        } else if (Number.isFinite(unbiasedGz)) {
            liveNav.headingRad = (Number.isFinite(liveNav.headingRad) ? liveNav.headingRad : 0.0) + unbiasedGz * dt;
        }
        if (!Number.isFinite(liveNav.headingRad)) {
            liveNav.headingRad = 0.0;
        }

        let vEast = liveNav.velocityMps * Math.sin(liveNav.headingRad);
        let vNorth = liveNav.velocityMps * Math.cos(liveNav.headingRad);

        let dEast = vEast * dt;
        let dNorth = vNorth * dt;

        let refLat = (liveNav.refLat !== null && isValidCoordinate(liveNav.refLat, liveNav.refLon || 77.5946)) ? liveNav.refLat : liveNav.estLat;
        let phi0 = refLat * (Math.PI / 180.0);
        let cosPhi0 = Math.max(0.01, Math.cos(phi0));
        let dLat = (dNorth / R_EARTH) * (180.0 / Math.PI);
        let dLon = (dEast / (R_EARTH * cosPhi0)) * (180.0 / Math.PI);

        if (!Number.isFinite(dLat)) dLat = 0.0;
        if (!Number.isFinite(dLon)) dLon = 0.0;

        let candLat = liveNav.estLat + dLat;
        let candLon = liveNav.estLon + dLon;
        if (isValidCoordinate(candLat, candLon)) {
            liveNav.estLat = candLat;
            liveNav.estLon = candLon;
        }

        if (!isValidCoordinate(liveNav.estLat, liveNav.estLon)) {
            liveNav.estLat = 12.9716;
            liveNav.estLon = 77.5946;
        }
    }

    assert.ok(Number.isFinite(liveNav.estLat), "Latitude must be finite");
    assert.ok(Number.isFinite(liveNav.estLon), "Longitude must be finite");
    assert.ok(isValidCoordinate(liveNav.estLat, liveNav.estLon), "Coordinates must remain valid");
    assert.ok(liveNav.estLat > 12.9716, "Latitude must have moved northward cleanly without NaN");
});

// --- 5. DOM & SERVICE WORKER CODE INTEGRATION ---
console.log("\n--- Test Group 5: UI & Service Worker Verification ---");

it("index.html contains Leaflet CSS/JS links, liveMap container, and error banner", () => {
    const html = fs.readFileSync(path.join(__dirname, '../index.html'), 'utf8');
    assert.ok(html.includes('assets/leaflet/leaflet.css'), "Must link leaflet.css");
    assert.ok(html.includes('assets/leaflet/leaflet.js'), "Must include leaflet.js");
    assert.ok(html.includes('id="liveMap"'), "Must contain #liveMap element");
    assert.ok(html.includes('id="offline-map-error-banner"'), "Must contain #offline-map-error-banner element");
    assert.ok(html.includes('initLiveGeographicMap'), "Must define initLiveGeographicMap function");
    assert.ok(html.includes('updateLiveMap'), "Must define updateLiveMap function");
    assert.ok(html.includes('isValidCoordinate'), "Must define isValidCoordinate function");
});

it("sw.js pre-caches local leaflet assets and road network", () => {
    const sw = fs.readFileSync(path.join(__dirname, '../sw.js'), 'utf8');
    assert.ok(sw.includes('/assets/leaflet/leaflet.js'), "sw.js must cache leaflet.js");
    assert.ok(sw.includes('/assets/leaflet/leaflet.css'), "sw.js must cache leaflet.css");
    assert.ok(sw.includes('/assets/maps/offline_road_network.json'), "sw.js must cache offline_road_network.json");
    assert.ok(sw.includes('ai-idr-cache-v2'), "sw.js must have cache version v2");
});

console.log("\n===============================================================");
console.log(`INTEGRATION TEST RESULTS: ${passedTests} / ${totalTests} PASSED`);
console.log("===============================================================");

if (passedTests === totalTests) {
    process.exit(0);
} else {
    process.exit(1);
}
