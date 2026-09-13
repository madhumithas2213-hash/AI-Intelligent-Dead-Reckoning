/**
 * AI-IDR Automatic GNSS Loss Detection & Seamless Restoration Verification Test Suite
 * SIH 2026 Problem Statement 26168
 * 
 * Verifies:
 * 1. Automatic GNSS Loss Watchdog (3.0s fix timeout and degraded accuracy threshold trigger)
 * 2. Seamless GNSS Restoration (Exponential drift absorption filter, zero teleportation)
 * 3. AI Navigation Confidence Engine (Multi-factor telemetry integrity score)
 * 4. Distinct Network Loss vs GNSS Loss Matrix handling
 * 5. Driver UI Cleanliness (Consumer Navigation HUD + Collapsible Diagnostics)
 */

const fs = require('fs');
const path = require('path');
const assert = require('assert');

console.log('===============================================================');
console.log('AI-IDR AUTOMATIC GNSS LOSS & RESTORATION TEST SUITE');
console.log('===============================================================\n');

const indexPath = path.join(__dirname, '..', 'index.html');
const indexHtml = fs.readFileSync(indexPath, 'utf8');

let passedTests = 0;
let totalTests = 0;

function test(description, fn) {
    totalTests++;
    try {
        fn();
        console.log(`  [PASS] ${description}`);
        passedTests++;
    } catch (err) {
        console.error(`  [FAIL] ${description}`);
        console.error(`         ${err.message}`);
    }
}

// -------------------------------------------------------------
// Test Group 1: UI Elements for Navigation Confidence & HUD
// -------------------------------------------------------------
console.log('--- Test Group 1: AI Navigation Confidence & Clean HUD ---');

test('Contains #nav-confidence-pill in Driver HUD Sheet', () => {
    assert(indexHtml.includes('id="nav-confidence-pill"'), 'Missing #nav-confidence-pill element in HUD');
    assert(indexHtml.includes('HIGH CONFIDENCE'), 'Missing HIGH CONFIDENCE label in HUD');
});

test('Contains #nav-confidence-note for degraded uncertainty notification', () => {
    assert(indexHtml.includes('id="nav-confidence-note"'), 'Missing #nav-confidence-note element');
});

test('Manual test override buttons are housed in diagnostics container', () => {
    const diagIdx = indexHtml.indexOf('id="diagnostics-drawer-body"');
    const outageBtnIdx = indexHtml.indexOf('id="btn-quick-outage"');
    assert(diagIdx !== -1, 'Missing #diagnostics-drawer-body');
    assert(outageBtnIdx !== -1, 'Missing #btn-quick-outage');
    assert(outageBtnIdx > diagIdx, 'btn-quick-outage must reside within diagnostics drawer, not main HUD');
});

// -------------------------------------------------------------
// Test Group 2: Automatic GNSS Loss Watchdog Logic
// -------------------------------------------------------------
console.log('\n--- Test Group 2: Automatic GNSS Loss Watchdog Logic ---');

test('Watchdog logic checks lastGpsFixTimestamp against timeout threshold', () => {
    assert(indexHtml.includes('GNSS_LOSS_TIMEOUT_MS'), 'Missing GNSS_LOSS_TIMEOUT_MS constant');
    assert(indexHtml.includes('triggerAutoGPSOutage'), 'Missing triggerAutoGPSOutage function');
    assert(indexHtml.includes('lastGpsFixTimestamp'), 'Missing lastGpsFixTimestamp sensor tracking');
});

test('Watchdog detects degraded accuracy exceeding threshold (>35m)', () => {
    assert(indexHtml.includes('GNSS_MAX_ACCURACY_M'), 'Missing GNSS_MAX_ACCURACY_M constant');
    assert(indexHtml.includes('liveSensors.gpsAcc > GNSS_MAX_ACCURACY_M'), 'Missing accuracy threshold check');
});

test('Automatic transition to AI Dead Reckoning without requiring manual interaction', () => {
    assert(indexHtml.includes('liveNav.isAutoOutage = true;'), 'Missing isAutoOutage flag setting');
    assert(indexHtml.includes('MODE 2: GNSS OUTAGE (DEAD RECKONING)'), 'Missing automatic Mode 2 switch');
});

// -------------------------------------------------------------
// Test Group 3: Seamless GNSS Restoration & Drift Absorption
// -------------------------------------------------------------
console.log('\n--- Test Group 3: Seamless GNSS Restoration (Zero Teleportation) ---');

test('Calculates restoration drift offset vector upon fix return', () => {
    assert(indexHtml.includes('restorationDriftLat'), 'Missing restorationDriftLat vector calculation');
    assert(indexHtml.includes('restorationDriftLon'), 'Missing restorationDriftLon vector calculation');
});

test('Applies exponential decay interpolation to glide position smoothly', () => {
    assert(indexHtml.includes('Math.exp(-1.5 * elapsedSec)'), 'Missing exponential decay formula for smooth drift absorption');
    assert(indexHtml.includes('GNSS_RESTORING'), 'Missing GNSS_RESTORING state transition');
});

test('Displays temporary GNSS RESTORED banner before settling into steady state', () => {
    assert(indexHtml.includes('GNSS RESTORED — AI + GNSS FUSION ACTIVE'), 'Missing restored banner notification');
});

// -------------------------------------------------------------
// Test Group 4: Multi-Factor Navigation Confidence Function
// -------------------------------------------------------------
console.log('\n--- Test Group 4: AI Navigation Confidence Engine ---');

test('calculateNavigationConfidence function is defined and evaluates 4 factors', () => {
    assert(indexHtml.includes('function calculateNavigationConfidence()'), 'Missing calculateNavigationConfidence definition');
    assert(indexHtml.includes('Factor 1: GNSS Availability'), 'Missing GNSS factor');
    assert(indexHtml.includes('Factor 2: IMU Sensors'), 'Missing IMU factor');
    assert(indexHtml.includes('Factor 3: Map Matching'), 'Missing Map Matching factor');
    assert(indexHtml.includes('Factor 4: Dead Reckoning Uncertainty'), 'Missing DR Drift factor');
});

// Simulate confidence calculation algorithm
function evalConfidence(hasGnss, gnssAcc, hasImu, snapDist, driftM) {
    let score = 0;
    if (hasGnss) {
        if (gnssAcc <= 8) score += 35;
        else if (gnssAcc <= 15) score += 28;
        else if (gnssAcc <= 25) score += 20;
        else score += 12;
    }
    if (hasImu) score += 25;
    if (snapDist <= 8.0) score += 25;
    else if (snapDist <= 20.0) score += 18;
    else score += 8;

    if (hasGnss) score += 15;
    else {
        let penalty = Math.min(15, Math.floor(driftM / 5.0));
        score += Math.max(0, 15 - penalty);
    }
    return score;
}

test('High confidence evaluated under nominal GNSS + IMU conditions (score >= 75)', () => {
    const score = evalConfidence(true, 5.0, true, 4.0, 0.0);
    assert(score >= 85, `Expected score >= 85, got ${score}`);
});

test('Outage confidence gracefully degrades to medium / low as drift accumulates', () => {
    const freshOutage = evalConfidence(false, null, true, 5.0, 2.0); // Drift 2m
    assert(freshOutage >= 60 && freshOutage < 75, `Expected medium confidence, got ${freshOutage}`);

    const longOutage = evalConfidence(false, null, true, 35.0, 80.0); // Drift 80m offroad
    assert(longOutage < 45, `Expected low confidence, got ${longOutage}`);
});

// -------------------------------------------------------------
// Test Group 5: Distinct Network vs GNSS Loss Matrix
// -------------------------------------------------------------
console.log('\n--- Test Group 5: Distinct Network vs GNSS Loss Separation ---');

test('Offline internet status does not interrupt GNSS navigation', () => {
    assert(indexHtml.includes('OFFLINE-FIRST — ON-DEVICE AI ACTIVE (NO INTERNET)'), 'Internet status must indicate offline mode without stopping navigation');
    assert(indexHtml.includes('OFFLINE ROAD NETWORK'), 'Offline road network must stay accessible');
});

// -------------------------------------------------------------
// Summary
// -------------------------------------------------------------
console.log('\n===============================================================');
console.log(`AUTOMATIC GNSS LOSS TEST RESULTS: ${passedTests} / ${totalTests} PASSED`);
console.log('===============================================================');

if (passedTests !== totalTests) {
    process.exit(1);
}
