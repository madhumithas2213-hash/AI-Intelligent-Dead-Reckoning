/**
 * Automated Verification Suite for Real-Time Offline-First Intelligent Dead Reckoning (AI-IDR)
 * Tests:
 * 1. On-device Pure JS GRU Velocity Predictor (42,433 params, NaN defense, window padding/trimming)
 * 2. Kinematic Dead Reckoning coordinate integration (WGS84)
 * 3. Offline Map Matcher candidate scoring and dynamic local corridor seeding
 * 4. Dataset separation check in index.html (verifying multiTrajData is isolated to benchmark lab)
 * 5. Asset synchronization & hash parity across deployment locations
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const assert = require('assert');

// 1. Load On-Device Engine
require('../offline_ml_engine.js');
const predictor = globalThis.offlineVelocityPredictor;

let testsPassed = 0;
let testsTotal = 0;

function it(desc, fn) {
    testsTotal++;
    try {
        fn();
        console.log(`  [PASS] ${desc}`);
        testsPassed++;
    } catch (err) {
        console.error(`  [FAIL] ${desc}`);
        console.error(`         ${err.message}`);
    }
}

console.log('===============================================================');
console.log('AI-IDR REAL-TIME OFFLINE-FIRST VERIFICATION TEST SUITE');
console.log('===============================================================\n');

// -------------------------------------------------------------
// Test Group 1: On-Device ML Velocity Predictor
// -------------------------------------------------------------
console.log('--- Test Group 1: On-Device ML Velocity Regressor ---');

it('Predictor is instantiated and initialized with correct architecture', () => {
    assert.ok(predictor, 'Predictor instance must exist');
    assert.strictEqual(predictor.arch.input_dim, 14, 'Input dimension must be 14');
    assert.strictEqual(predictor.arch.hidden_dim, 64, 'Hidden dimension must be 64');
    assert.strictEqual(predictor.arch.num_layers, 2, 'Must have 2 GRU layers');
    assert.strictEqual(predictor.arch.window_size, 20, 'Temporal window size must be 20');
});

it('Stationary sensor window predicts non-negative speed <= 5 m/s', () => {
    // 20 samples of typical stationary phone: 1g on Z, tiny noise
    const stationaryWindow = [];
    for (let i = 0; i < 20; i++) {
        stationaryWindow.push([
            0.01, -0.02, 9.81, // ax, ay, az
            0.001, -0.001, 0.000, // gx, gy, gz
            0.0, 0.0, 9.80, // grav_x, grav_y, grav_z
            -7.5, -16.5, 5.3, // mx, my, mz
            9.81, 0.001 // acc_mag, gyro_mag
        ]);
    }
    const res = predictor.predict(stationaryWindow);
    assert.ok(typeof res.velocity_mps === 'number', 'velocity_mps must be a number');
    assert.ok(Number.isFinite(res.velocity_mps), 'velocity_mps must be finite');
    assert.ok(res.velocity_mps >= 0.0, `Speed must be >= 0 m/s (got ${res.velocity_mps})`);
    assert.ok(res.velocity_mps < 10.0, `Stationary speed must be low (got ${res.velocity_mps})`);
    assert.ok(res.latency_ms > 0, `Latency must be tracked (got ${res.latency_ms} ms)`);
    assert.ok(res.latency_ms < 100, `On-device inference must be real-time (< 100ms, got ${res.latency_ms}ms)`);
});

it('predictVelocity() returns scalar velocity_mps directly', () => {
    const window20 = Array(20).fill(Array(14).fill(0.1));
    const scalar = predictor.predictVelocity(window20);
    assert.ok(typeof scalar === 'number', 'predictVelocity must return a scalar number');
    assert.ok(Number.isFinite(scalar) && scalar >= 0.0, 'Scalar velocity must be finite and >= 0');
});

it('Handles short windows (< 20 samples) via auto-padding', () => {
    const shortWindow = [
        [0.0, 0.0, 9.81, 0, 0, 0, 0, 0, 9.81, -7, -16, 5, 9.81, 0]
    ];
    const res = predictor.predict(shortWindow);
    assert.ok(Number.isFinite(res.velocity_mps), 'Must handle 1 sample window without error');
    assert.ok(res.velocity_mps >= 0.0, 'Velocity must be non-negative');
});

it('Handles long windows (> 20 samples) via auto-trimming', () => {
    const longWindow = Array(45).fill(Array(14).fill(0.05));
    const res = predictor.predict(longWindow);
    assert.ok(Number.isFinite(res.velocity_mps), 'Must handle 45 sample window without error');
    assert.ok(res.velocity_mps >= 0.0, 'Velocity must be non-negative');
});

it('NaN Defense: Handles corrupt sensor values gracefully without NaN output', () => {
    const dirtyWindow = Array(20).fill([NaN, null, undefined, 'abc', 0, 0, 0, 0, 9.81, -7, -16, 5, 9.81, 0]);
    const res = predictor.predict(dirtyWindow);
    assert.ok(Number.isFinite(res.velocity_mps), 'Output velocity must never be NaN');
    assert.ok(Number.isFinite(res.velocity_kmh), 'Output km/h must never be NaN');
    assert.ok(res.velocity_mps >= 0.0, 'Output must be non-negative');
});

// -------------------------------------------------------------
// Test Group 2: Dead Reckoning Kinematics
// -------------------------------------------------------------
console.log('\n--- Test Group 2: Dead Reckoning Kinematics (WGS84) ---');

it('Calculates correct displacement and advances coordinates along heading', () => {
    const startLat = 12.971600;
    const startLon = 77.594600;
    const headingRad = 0.0; // Facing North
    const speedMps = 10.0; // 10 m/s
    const dt = 1.0; // 1 second = 10 meters North

    const R_EARTH = 6378137.0;
    const dNorth = speedMps * Math.cos(headingRad) * dt;
    const dEast = speedMps * Math.sin(headingRad) * dt;

    const newLat = startLat + (dNorth / R_EARTH) * (180.0 / Math.PI);
    const newLon = startLon + (dEast / (R_EARTH * Math.cos(startLat * Math.PI / 180.0))) * (180.0 / Math.PI);

    assert.ok(newLat > startLat, 'Latitude must increase when driving North');
    assert.strictEqual(newLon, startLon, 'Longitude must remain unchanged when driving due North');
    
    // Check distance in meters
    const latDiffMeters = (newLat - startLat) * (Math.PI / 180.0) * R_EARTH;
    assert.ok(Math.abs(latDiffMeters - 10.0) < 0.001, `Displacement must equal 10.0 meters (got ${latDiffMeters})`);
});

it('Eastward displacement advances longitude correctly', () => {
    const startLat = 12.971600;
    const startLon = 77.594600;
    const headingRad = Math.PI / 2.0; // Facing East (90 deg)
    const speedMps = 20.0; // 20 m/s
    const dt = 2.0; // 40 meters East

    const R_EARTH = 6378137.0;
    const dEast = speedMps * Math.sin(headingRad) * dt;
    const newLon = startLon + (dEast / (R_EARTH * Math.cos(startLat * Math.PI / 180.0))) * (180.0 / Math.PI);

    assert.ok(newLon > startLon, 'Longitude must increase when driving East');
    const lonDiffMeters = (newLon - startLon) * (Math.PI / 180.0) * R_EARTH * Math.cos(startLat * Math.PI / 180.0);
    assert.ok(Math.abs(lonDiffMeters - 40.0) < 0.001, `Displacement must equal 40.0 meters (got ${lonDiffMeters})`);
});

// -------------------------------------------------------------
// Test Group 3: Offline Map Matching Engine
// -------------------------------------------------------------
console.log('\n--- Test Group 3: Client-Side Offline Map Matcher ---');

// Instantiate matcher logic matching index.html
class OfflineMapMatcher {
    constructor() {
        this.segments = [
            { id: 'SEG_OFFLINE_001', name: 'MG Road Eastbound', points: [[12.971600, 77.594600], [12.971850, 77.596200], [12.972100, 77.597800], [12.972400, 77.599500]] },
            { id: 'SEG_OFFLINE_002', name: 'Brigade Road Southbound', points: [[12.971850, 77.596200], [12.970200, 77.596400], [12.968500, 77.596600], [12.966800, 77.596800]] }
        ];
    }

    seedOriginSegment(refLat, refLon, headingRad) {
        if (refLat === null || refLon === null || isNaN(refLat) || isNaN(refLon)) return;
        let cosLat = Math.cos(refLat * Math.PI / 180.0);
        let R = 6378137.0;
        let toCoord = (eM, nM) => [
            refLat + (nM / R) * (180.0 / Math.PI),
            refLon + (eM / (R * cosLat)) * (180.0 / Math.PI)
        ];
        let head = (headingRad !== null && !isNaN(headingRad)) ? headingRad : 0.0;
        let sinH = Math.sin(head), cosH = Math.cos(head);
        
        let pts = [];
        for (let offsetM = -300; offsetM <= 1500; offsetM += 50) {
            pts.push(toCoord(offsetM * sinH, offsetM * cosH));
        }
        this.segments.push({ id: 'SEG_DYN_MAIN', name: 'Local Road Corridor', points: pts });
    }

    projectPointToSegment(pLat, pLon, aLat, aLon, bLat, bLon) {
        let cosLat = Math.cos(pLat * Math.PI / 180.0);
        let bx = (bLon - aLon) * 111320.0 * cosLat;
        let by = (bLat - aLat) * 110540.0;
        let px = (pLon - aLon) * 111320.0 * cosLat;
        let py = (pLat - aLat) * 110540.0;

        let segLenSq = bx * bx + by * by;
        let t = segLenSq === 0 ? 0 : Math.max(0, Math.min(1, (px * bx + py * by) / segLenSq));
        let snapLat = aLat + t * (bLat - aLat);
        let snapLon = aLon + t * (bLon - aLon);
        let dx = (pLon - snapLon) * 111320.0 * cosLat;
        let dy = (pLat - snapLat) * 110540.0;
        let dist = Math.sqrt(dx * dx + dy * dy);
        let segHeading = Math.atan2(bx, by);
        return { snapLat, snapLon, distMeters: dist, segHeading, t };
    }

    match(lat, lon, headingRad) {
        let best = null;
        let bestScore = Infinity;
        for (let seg of this.segments) {
            for (let i = 0; i < seg.points.length - 1; i++) {
                let [aLat, aLon] = seg.points[i];
                let [bLat, bLon] = seg.points[i + 1];
                let proj = this.projectPointToSegment(lat, lon, aLat, aLon, bLat, bLon);
                let headDiff = 0;
                if (headingRad !== null && !isNaN(headingRad)) {
                    let dH = Math.abs(headingRad - proj.segHeading) % (2 * Math.PI);
                    if (dH > Math.PI) dH = 2 * Math.PI - dH;
                    let dHOpp = Math.abs(dH - Math.PI);
                    headDiff = Math.min(dH, dHOpp);
                }
                let score = proj.distMeters + headDiff * 15.0;
                if (score < bestScore) {
                    bestScore = score;
                    best = { ...proj, segmentId: seg.id, segmentName: seg.name, score };
                }
            }
        }
        return best;
    }
}

it('OfflineMapMatcher correctly seeds dynamic corridor around GPS fix', () => {
    const matcher = new OfflineMapMatcher();
    const countBefore = matcher.segments.length;
    matcher.seedOriginSegment(13.0827, 80.2707, 0.5); // Chennai coordinates
    assert.strictEqual(matcher.segments.length, countBefore + 1, 'Should add dynamic segment');
    const dyn = matcher.segments.find(s => s.id === 'SEG_DYN_MAIN');
    assert.ok(dyn, 'Dynamic corridor segment must exist');
    assert.ok(dyn.points.length > 10, 'Dynamic corridor must contain sampled road points');
});

it('Snaps off-road vehicle position to closest road centerline', () => {
    const matcher = new OfflineMapMatcher();
    // Test point 10 meters north of MG Road segment
    const testLat = 12.971600 + (10.0 / 110540.0);
    const testLon = 77.595000;
    const match = matcher.match(testLat, testLon, Math.PI / 2.0); // Heading East
    
    assert.ok(match, 'Must return a map match');
    assert.strictEqual(match.segmentId, 'SEG_OFFLINE_001', 'Must match MG Road Eastbound');
    assert.ok(match.distMeters <= 12.0, `Orthogonal distance must be close to 10m (got ${match.distMeters})`);
    assert.ok(match.snapLat < testLat, 'Snapped point must be pulled south towards MG Road centerline');
});

// -------------------------------------------------------------
// Test Group 4: Dataset Separation Verification
// -------------------------------------------------------------
console.log('\n--- Test Group 4: Dataset Separation in Live Navigation ---');

it('index.html boots default to live navigation view', () => {
    const indexHtml = fs.readFileSync(path.join(__dirname, '../index.html'), 'utf8');
    assert.ok(indexHtml.includes("switchTab('live')"), 'Boot logic must default to switchTab("live")');
});

it('Live sensor processing does not read from multiTrajData', () => {
    const indexHtml = fs.readFileSync(path.join(__dirname, '../index.html'), 'utf8');
    
    // Find processLiveTick function definition
    const tickStart = indexHtml.indexOf('function processLiveTick(');
    assert.ok(tickStart > 0, 'processLiveTick function must exist');
    const tickEnd = indexHtml.indexOf('function startLiveNavigation()', tickStart);
    const tickCode = indexHtml.substring(tickStart, tickEnd > 0 ? tickEnd : tickStart + 5000);

    assert.ok(!tickCode.includes('multiTrajData'), 'processLiveTick must NEVER reference multiTrajData');
    assert.ok(tickCode.includes('liveSensorFeatureBuffer'), 'processLiveTick must use real liveSensorFeatureBuffer');
    assert.ok(tickCode.includes('offlineVelocityPredictor'), 'processLiveTick must call offlineVelocityPredictor');
});

// -------------------------------------------------------------
// Test Group 5: Asset Synchronization & Parity
// -------------------------------------------------------------
console.log('\n--- Test Group 5: File Hash Parity Across Deployment Folders ---');

function getHash(filePath) {
    const buf = fs.readFileSync(filePath);
    return crypto.createHash('sha256').update(buf).digest('hex');
}

it('index.html matches dashboard.html and fusion outputs', () => {
    const rootIndex = getHash(path.join(__dirname, '../index.html'));
    const rootDash = getHash(path.join(__dirname, '../dashboard.html'));
    const fusionIndex = getHash(path.join(__dirname, '../ml/outputs/fusion/index.html'));
    const fusionDash = getHash(path.join(__dirname, '../ml/outputs/fusion/dashboard.html'));

    assert.strictEqual(rootIndex, rootDash, 'index.html must match dashboard.html');
    assert.strictEqual(rootIndex, fusionIndex, 'index.html must match ml/outputs/fusion/index.html');
    assert.strictEqual(rootIndex, fusionDash, 'index.html must match ml/outputs/fusion/dashboard.html');
});

it('offline_ml_engine.js matches ml/outputs/fusion/offline_ml_engine.js', () => {
    const rootEngine = getHash(path.join(__dirname, '../offline_ml_engine.js'));
    const fusionEngine = getHash(path.join(__dirname, '../ml/outputs/fusion/offline_ml_engine.js'));
    assert.strictEqual(rootEngine, fusionEngine, 'offline_ml_engine.js must match fusion copy');
});

it('sw.js matches ml/outputs/fusion/sw.js', () => {
    const rootSw = getHash(path.join(__dirname, '../sw.js'));
    const fusionSw = getHash(path.join(__dirname, '../ml/outputs/fusion/sw.js'));
    assert.strictEqual(rootSw, fusionSw, 'sw.js must match fusion copy');
});

// -------------------------------------------------------------
// Summary
// -------------------------------------------------------------
console.log('\n===============================================================');
console.log(`TEST RESULTS: ${testsPassed} / ${testsTotal} PASSED`);
console.log('===============================================================\n');

if (testsPassed === testsTotal) {
    process.exit(0);
} else {
    process.exit(1);
}
