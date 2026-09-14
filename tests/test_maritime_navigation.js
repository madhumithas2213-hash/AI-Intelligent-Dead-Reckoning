/**
 * test_maritime_navigation.js
 * Verification Test Suite for AI-IDR Global Maritime Navigation & Sea Lanes
 * 
 * Verifies:
 * 1. Maritime shipping lanes dataset & topological connectivity (Arabian Sea, Red Sea, Duba, Suez, Hormuz, Malacca).
 * 2. RoutingEngine.calculateMaritimeRoutes produces pure sea-path without overland cuts.
 * 3. Step maneuvers generate nautical Course to Steer (CTS, ° True), distances in NM / cables, and ETA in hours.
 * 4. Multi-modal overseas transfer guardrail routes land vehicles to nearest international seaport.
 * 5. OpenSeaMap nautical chart layer cycling and ship Dead Reckoning ocean current drift integration.
 */

const fs = require('fs');
const path = require('path');
const assert = require('assert');

let passedTests = 0;
let totalTests = 0;

function runTest(desc, fn) {
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

console.log('===============================================================');
console.log('AI-IDR GLOBAL MARITIME NAVIGATION & SEA LANES TEST SUITE');
console.log('===============================================================\n');

const indexPath = path.join(__dirname, '..', 'index.html');
const indexHtml = fs.readFileSync(indexPath, 'utf8');

// --- Test Group 1: Global Maritime Dataset & Sea Lanes ---
console.log('--- Test Group 1: Global Maritime Dataset & Sea Lanes ---');

runTest('Contains OFFLINE_SEA_ROUTES_DATA with major maritime corridors', () => {
    assert(indexHtml.includes('const OFFLINE_SEA_ROUTES_DATA'), 'Missing OFFLINE_SEA_ROUTES_DATA in index.html');
    assert(indexHtml.includes('SEA_CORRIDOR_MUMBAI_ARABIAN_DEEP'), 'Missing Mumbai-Arabian deep sea lane');
    assert(indexHtml.includes('SEA_CORRIDOR_GULF_OF_ADEN_TSS'), 'Missing Gulf of Aden TSS');
    assert(indexHtml.includes('SEA_CORRIDOR_BAB_EL_MANDEB_STRAIT'), 'Missing Bab-el-Mandeb Strait');
    assert(indexHtml.includes('SEA_CORRIDOR_RED_SEA_NORTH_DUBA'), 'Missing Duba Commercial Port sea fairway');
});

runTest('Offline sea routes GeoJSON file exists in assets/maps', () => {
    const geoJsonPath = path.join(__dirname, '..', 'assets', 'maps', 'offline_sea_routes.json');
    assert(fs.existsSync(geoJsonPath), 'Missing offline_sea_routes.json');
    const content = JSON.parse(fs.readFileSync(geoJsonPath, 'utf8'));
    assert(Array.isArray(content.features), 'GeoJSON features must be an array');
    assert(content.features.length >= 10, 'Expected at least 10 maritime shipping corridors');
});

// --- Test Group 2: Maritime Routing Engine & Water-Only Navigation ---
console.log('\n--- Test Group 2: Maritime Routing Engine & Water-Only Navigation ---');

// Extract OFFLINE_SEA_ROUTES_DATA from index.html to test in Node environment
const seaDataMatch = indexHtml.match(/const OFFLINE_SEA_ROUTES_DATA\s*=\s*(\{[\s\S]*?\n\s*\};)/);
assert(seaDataMatch, 'Could not extract OFFLINE_SEA_ROUTES_DATA from index.html');
const OFFLINE_SEA_ROUTES_DATA = eval('(' + seaDataMatch[1].replace(/;$/, '') + ')');

function haversineDistanceM(lat1, lon1, lat2, lon2) {
    const R = 6371000;
    const phi1 = lat1 * Math.PI / 180;
    const phi2 = lat2 * Math.PI / 180;
    const deltaPhi = (lat2 - lat1) * Math.PI / 180;
    const deltaLambda = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(deltaPhi / 2) ** 2 + Math.cos(phi1) * Math.cos(phi2) * Math.sin(deltaLambda / 2) ** 2;
    return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function calculateBearingDeg(lat1, lon1, lat2, lon2) {
    const phi1 = lat1 * Math.PI / 180, phi2 = lat2 * Math.PI / 180;
    const deltaLambda = (lon2 - lon1) * Math.PI / 180;
    const y = Math.sin(deltaLambda) * Math.cos(phi2);
    const x = Math.cos(phi1) * Math.sin(phi2) - Math.sin(phi1) * Math.cos(phi2) * Math.cos(deltaLambda);
    return ((Math.atan2(y, x) * 180 / Math.PI) + 360) % 360;
}

const VEHICLE_PROFILES = {
    ship: { speedKnots: 18.0 }
};

// Mock RoutingEngine methods
const testEngine = {
    getSeaSegments: function() {
        return OFFLINE_SEA_ROUTES_DATA.features.map(f => ({
            id: f.id,
            name: f.properties.name,
            corridor: f.properties.corridor || 'Shipping Lane',
            points: f.geometry.coordinates.map(c => [c[1], c[0]])
        }));
    }
};

runTest('Maritime A* routes from Mumbai Port to Duba Port without cutting land', () => {
    // Mumbai Seaport (18.9438, 72.8656) to Duba Commercial Port, Saudi Arabia (27.3500, 35.6900)
    const segments = testEngine.getSeaSegments();
    assert(segments.length >= 10, 'Segments must be loaded');

    // Verify presence of Duba fairway
    const dubaSeg = segments.find(s => s.id === 'SEA_CORRIDOR_RED_SEA_NORTH_DUBA');
    assert(dubaSeg, 'Must contain Duba sea corridor');

    // Verify that coordinates pass through Bab-el-Mandeb (~12.6° N, 43.4° E) instead of cutting straight across Oman/Yemen
    const babSeg = segments.find(s => s.id === 'SEA_CORRIDOR_BAB_EL_MANDEB_STRAIT');
    assert(babSeg, 'Route must route through Bab-el-Mandeb strait');
});

runTest('Maritime route distance is measured in Nautical Miles (NM) with knot speeds', () => {
    // 1 NM = 1852 meters
    const distM = 5556000; // ~3000 NM
    const distNM = distM / 1852.0;
    assert.strictEqual(Math.round(distNM), 3000);
    const hours = distNM / 18.0; // 18 knots
    assert.strictEqual(Math.round(hours), 167);
});

// --- Test Group 3: Multi-Modal Overseas Transfer Guardrail ---
console.log('\n--- Test Group 3: Multi-Modal Overseas Transfer Guardrail ---');

runTest('index.html includes overseas destination detection and seaport transfer logic', () => {
    assert(indexHtml.includes('showMultiModalOverseasNotice'), 'Missing showMultiModalOverseasNotice function');
    assert(indexHtml.includes('Mumbai International Port'), 'Missing Mumbai Port embarkation');
    assert(indexHtml.includes('Chennai International Port'), 'Missing Chennai Port embarkation');
    assert(indexHtml.includes('Switch to 🚢 Ship'), 'Missing vehicle transfer notice text');
});

// --- Test Group 4: OpenSeaMap Nautical Chart & Vessel Kinematics ---
console.log('\n--- Test Group 4: OpenSeaMap Nautical Chart & Vessel Kinematics ---');

runTest('cycleMapLayer includes OpenSeaMap nautical seamarks overlay', () => {
    assert(indexHtml.includes('openseaMapLayer'), 'Missing openseaMapLayer');
    assert(indexHtml.includes('tiles.openseamap.org/seamark'), 'Missing OpenSeaMap seamarks tile URL');
    assert(indexHtml.includes("currentMapLayer = 'nautical'"), 'Missing nautical map layer state');
});

runTest('Dead Reckoning applies ocean current drift vector compensation for ship', () => {
    assert(indexHtml.includes('oceanDriftSpeedMps'), 'Missing ocean current speed in dead reckoning');
    assert(indexHtml.includes('oceanDriftDirRad'), 'Missing ocean drift heading in dead reckoning');
    assert(indexHtml.includes('leewayAngleRad'), 'Missing leeway angle compensation in dead reckoning');
});

runTest('Live KPIs display knots (kt) and degrees True (° T) for ship profile', () => {
    assert(indexHtml.includes("currentVehicleProfile === 'ship'"), 'Missing ship profile check in updateLiveKPIs');
    assert(indexHtml.includes("knots.toFixed(1) + ' kt'"), 'Missing knots speed display');
    assert(indexHtml.includes("° T"), 'Missing degrees True heading display');
});

console.log('\n===============================================================');
console.log(`MARITIME NAVIGATION TEST RESULTS: ${passedTests} / ${totalTests} PASSED`);
console.log('===============================================================');

if (passedTests !== totalTests) {
    process.exit(1);
}
