/**
 * AI-IDR Mobile-First UI/UX Redesign Automated Verification Test
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');

console.log('===============================================================');
console.log('AI-IDR MOBILE-FIRST NAVIGATION UI/UX REDESIGN TEST SUITE');
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

console.log('--- Test Group 1: Mobile Top Header Bar ---');
test('Contains .mobile-header-bar with AI-IDR Nav branding', () => {
    assert(indexHtml.includes('class="mobile-header-bar"'), 'Missing .mobile-header-bar class');
    assert(indexHtml.includes('AI-IDR Nav'), 'Missing AI-IDR Nav text');
    assert(indexHtml.includes('id="m-driver-mode-badge"'), 'Missing #m-driver-mode-badge');
    assert(indexHtml.includes('id="m-header-status-text"'), 'Missing #m-header-status-text');
});

console.log('\n--- Test Group 2: Hero Map Container ---');
test('Contains .nav-hero-map-card and .nav-map-viewport with Leaflet map', () => {
    assert(indexHtml.includes('class="nav-hero-map-card"'), 'Missing .nav-hero-map-card');
    assert(indexHtml.includes('class="nav-map-viewport"'), 'Missing .nav-map-viewport');
    assert(indexHtml.includes('id="liveMap"'), 'Missing #liveMap');
    assert(indexHtml.includes('id="liveTrajCanvas"'), 'Missing #liveTrajCanvas');
    assert(indexHtml.includes('id="live-gnss-badge"'), 'Missing #live-gnss-badge');
    assert(indexHtml.includes('id="live-offline-badge"'), 'Missing #live-offline-badge');
    assert(indexHtml.includes('id="live-routing-badge"'), 'Missing #live-routing-badge');
    assert(indexHtml.includes('id="btn-map-follow-toggle"'), 'Missing #btn-map-follow-toggle');
    assert(indexHtml.includes('id="btn-map-recenter"'), 'Missing #btn-map-recenter');
    assert(indexHtml.includes('id="offline-map-error-banner"'), 'Missing #offline-map-error-banner');
});

console.log('\n--- Test Group 3: In-Car Driver Navigation HUD Sheet ---');
test('Contains .driver-hud-sheet with live Speed, Heading, Mode Pill, and Road Name', () => {
    assert(indexHtml.includes('class="driver-hud-sheet"'), 'Missing .driver-hud-sheet');
    assert(indexHtml.includes('id="driver-speed"'), 'Missing #driver-speed');
    assert(indexHtml.includes('id="driver-heading"'), 'Missing #driver-heading');
    assert(indexHtml.includes('id="driver-cardinal"'), 'Missing #driver-cardinal');
    assert(indexHtml.includes('id="driver-mode-pill"'), 'Missing #driver-mode-pill');
    assert(indexHtml.includes('id="driver-road"'), 'Missing #driver-road');
});

test('Contains Driver Quick Touch Action Buttons', () => {
    assert(indexHtml.includes('id="btn-quick-outage"'), 'Missing #btn-quick-outage');
    assert(indexHtml.includes('id="text-quick-outage"'), 'Missing #text-quick-outage');
    assert(indexHtml.includes('id="btn-quick-restore"'), 'Missing #btn-quick-restore');
    assert(indexHtml.includes('id="btn-quick-emulate"'), 'Missing #btn-quick-emulate');
    assert(indexHtml.includes('id="btn-quick-sensors"'), 'Missing #btn-quick-sensors');
    assert(indexHtml.includes('id="btn-quick-diagnostics"'), 'Missing #btn-quick-diagnostics');
    assert(indexHtml.includes('id="text-quick-diag"'), 'Missing #text-quick-diag');
});

test('Contains Real-Time Trajectory Comparison Benchmark Bar', () => {
    assert(indexHtml.includes('id="live-trajectory-comparison-bar"'), 'Missing #live-trajectory-comparison-bar');
    assert(indexHtml.includes('id="metric-pos-error"'), 'Missing #metric-pos-error');
    assert(indexHtml.includes('id="metric-drift-error"'), 'Missing #metric-drift-error');
    assert(indexHtml.includes('id="metric-speed-error"'), 'Missing #metric-speed-error');
    assert(indexHtml.includes('id="metric-heading-error"'), 'Missing #metric-heading-error');
    assert(indexHtml.includes('id="metric-snap-offset"'), 'Missing #metric-snap-offset');
    assert(indexHtml.includes('id="metric-active-road"'), 'Missing #metric-active-road');
});

console.log('\n--- Test Group 4: Collapsible Developer Diagnostics Drawer ---');
test('Contains toggle button and drawer body enclosing all developer metrics', () => {
    assert(indexHtml.includes('class="diagnostics-accordion-toggle"'), 'Missing .diagnostics-accordion-toggle');
    assert(indexHtml.includes('id="diagnostics-caret"'), 'Missing #diagnostics-caret');
    assert(indexHtml.includes('id="diagnostics-drawer-body"'), 'Missing #diagnostics-drawer-body');
    assert(indexHtml.includes('id="live-val-speed"'), 'Missing #live-val-speed');
    assert(indexHtml.includes('id="live-val-mode"'), 'Missing #live-val-mode');
    assert(indexHtml.includes('id="live-val-drift"'), 'Missing #live-val-drift');
    assert(indexHtml.includes('id="live-val-gnss"'), 'Missing #live-val-gnss');
    assert(indexHtml.includes('id="live-val-conf"'), 'Missing #live-val-conf');
    assert(indexHtml.includes('id="btn-request-sensors"'), 'Missing #btn-request-sensors');
    assert(indexHtml.includes('id="btn-request-gps"'), 'Missing #btn-request-gps');
    assert(indexHtml.includes('id="node-phone"'), 'Missing #node-phone');
    assert(indexHtml.includes('id="node-ui"'), 'Missing #node-ui');
});

console.log('\n--- Test Group 5: Fixed Mobile Bottom Navigation Bar ---');
test('Contains .mobile-bottom-nav with all 5 navigation tabs', () => {
    assert(indexHtml.includes('class="mobile-bottom-nav"'), 'Missing .mobile-bottom-nav');
    assert(indexHtml.includes('id="m-nav-live"'), 'Missing #m-nav-live');
    assert(indexHtml.includes('id="m-nav-sensors"'), 'Missing #m-nav-sensors');
    assert(indexHtml.includes('id="m-nav-home"'), 'Missing #m-nav-home');
    assert(indexHtml.includes('id="m-nav-demo"'), 'Missing #m-nav-demo');
    assert(indexHtml.includes('id="m-nav-performance"'), 'Missing #m-nav-performance');
});

console.log('\n--- Test Group 6: JavaScript Controller Logic ---');
test('Contains toggleDiagnosticsDrawer function definition', () => {
    assert(indexHtml.includes('function toggleDiagnosticsDrawer()'), 'Missing toggleDiagnosticsDrawer');
});

test('switchTab synchronizes mobile-nav-btn active classes', () => {
    assert(indexHtml.includes("document.querySelectorAll('.mobile-nav-btn').forEach"), 'Missing mobile-nav-btn sync');
    assert(indexHtml.includes("document.getElementById('m-nav-' + tabId)"), 'Missing m-nav- target');
});

test('updateLiveKPIs updates Driver HUD elements in real time', () => {
    assert(indexHtml.includes("document.getElementById('driver-speed')"), 'Missing driver-speed update');
    assert(indexHtml.includes("document.getElementById('driver-heading')"), 'Missing driver-heading update');
    assert(indexHtml.includes("document.getElementById('driver-cardinal')"), 'Missing driver-cardinal update');
    assert(indexHtml.includes("document.getElementById('driver-mode-pill')"), 'Missing driver-mode-pill update');
    assert(indexHtml.includes("document.getElementById('driver-road')"), 'Missing driver-road update');
});

console.log('\n===============================================================');
console.log(`MOBILE UI TEST RESULTS: ${passedTests} / ${totalTests} PASSED`);
console.log('===============================================================');

if (passedTests !== totalTests) {
    process.exit(1);
}
