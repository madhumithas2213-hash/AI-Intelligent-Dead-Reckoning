// tests/test_real_place_search.js
const fs = require('fs');
const path = require('path');
const assert = require('assert');

console.log('===============================================================');
console.log('AI-IDR REAL PLACE SEARCH & GEOCODING TEST SUITE');
console.log('===============================================================');

const indexPath = path.join(__dirname, '..', 'index.html');
const indexHtml = fs.readFileSync(indexPath, 'utf-8');

let passCount = 0;
function test(desc, fn) {
    try {
        fn();
        console.log(`  [PASS] ${desc}`);
        passCount++;
    } catch (e) {
        console.error(`  [FAIL] ${desc}: ${e.message}`);
        process.exitCode = 1;
    }
}

console.log('\n--- Test Group 1: Search UI Elements & Autocomplete ---');
test('Contains #nav-dest-input with city/landmark placeholder', () => {
    assert(indexHtml.includes('id="nav-dest-input"'), 'Missing #nav-dest-input');
    assert(indexHtml.includes('placeholder="Where to? Type city, landmark or choose preset below..."'), 'Missing updated placeholder');
});

test('Contains #nav-dest-suggestions dropdown and #nav-dest-status-pill', () => {
    assert(indexHtml.includes('id="nav-dest-suggestions"'), 'Missing #nav-dest-suggestions');
    assert(indexHtml.includes('id="nav-dest-status-pill"'), 'Missing #nav-dest-status-pill');
});

test('Includes CSS styling for suggestions dropdown and items', () => {
    assert(indexHtml.includes('.dest-suggestions-dropdown'), 'Missing .dest-suggestions-dropdown CSS');
    assert(indexHtml.includes('.dest-suggestion-item'), 'Missing .dest-suggestion-item CSS');
    assert(indexHtml.includes('.dest-suggestion-title'), 'Missing .dest-suggestion-title CSS');
    assert(indexHtml.includes('.dest-status-pill'), 'Missing .dest-status-pill CSS');
});

console.log('\n--- Test Group 2: Geocoding & OpenStreetMap Nominatim Integration ---');
test('Contains Nominatim API endpoint for real place geocoding', () => {
    assert(indexHtml.includes('nominatim.openstreetmap.org/search'), 'Missing Nominatim search endpoint');
});

test('onPreviewRouteClick is async and queries Nominatim when non-preset query is typed', () => {
    assert(indexHtml.includes('async function onPreviewRouteClick'), 'onPreviewRouteClick must be async');
    assert(indexHtml.includes('destLat = parseFloat(best.lat)'), 'Missing destLat parsing from geocoding');
    assert(indexHtml.includes('destLon = parseFloat(best.lon)'), 'Missing destLon parsing from geocoding');
});

test('Contains Enter key listener and debounced typing suggestions', () => {
    assert(indexHtml.includes('initDestinationSearchInput'), 'Missing initDestinationSearchInput function');
    assert(indexHtml.includes("e.key === 'Enter'"), 'Missing Enter key handler on search input');
    assert(indexHtml.includes('searchDebounceTimer'), 'Missing search debouncing');
});

console.log('\n--- Test Group 3: Real Map Pan & Route Preview ---');
test('Centers and fits map bounds for searched destinations', () => {
    assert(indexHtml.includes('liveLeafletMap.panTo([destLat, destLon])'), 'Must pan map to searched coordinates');
    assert(indexHtml.includes('liveLeafletMap.fitBounds'), 'Must fit bounds to route');
});

console.log('===============================================================');
console.log(`REAL PLACE SEARCH RESULTS: ${passCount} / 7 PASSED`);
console.log('===============================================================');
