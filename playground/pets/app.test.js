const test = require('node:test');
const assert = require('node:assert');
const { MAX_DEPTH } = require('./config');
const { max } = require('./pets/max');
const { buddy } = require('./pets/buddy');
const { mittens } = require('./pets/mittens');
const { whiskers } = require('./pets/whiskers');
const { daisy } = require('./pets/daisy');
const { milo } = require('./pets/milo');
const { coco } = require('./pets/coco');
const { ginger } = require('./pets/ginger');

test('MAX_DEPTH is 10', () => {
    assert.strictEqual(MAX_DEPTH, 10);
});

test('chain 1: max calls buddy', () => {
    let called = false;
    const originalLog = console.log;
    console.log = (msg) => {
        if (typeof msg === 'string' && msg.includes('buddy')) called = true;
    };
    max(MAX_DEPTH - 2);
    console.log = originalLog;
    assert.ok(called, 'buddy should be called from max');
});

test('chain 1: buddy calls both mittens and sparky', () => {
    let mittensCalled = false;
    let sparkyCalled = false;
    const originalLog = console.log;
    console.log = (msg) => {
        if (typeof msg === 'string') {
            if (msg.includes('mittens')) mittensCalled = true;
            if (msg.includes('sparky')) sparkyCalled = true;
        }
    };
    buddy(MAX_DEPTH - 3);
    console.log = originalLog;
    assert.ok(mittensCalled, 'mittens should be called from buddy');
    assert.ok(sparkyCalled, 'sparky should be called from buddy');
});

test('depth limit prevents stack overflow', () => {
    assert.strictEqual(whiskers(MAX_DEPTH), 'whiskers');
    assert.strictEqual(mittens(MAX_DEPTH), 'mittens');
    assert.strictEqual(buddy(MAX_DEPTH), 'buddy');
    assert.strictEqual(max(MAX_DEPTH), 'max');
});

test('chain 2: milo calls daisy', () => {
    let called = false;
    const originalLog = console.log;
    console.log = (msg) => {
        if (typeof msg === 'string' && msg.includes('daisy')) called = true;
    };
    milo(MAX_DEPTH - 2);
    console.log = originalLog;
    assert.ok(called, 'daisy should be called from milo');
});

test('chain 2: daisy calls both luna and coco', () => {
    let lunaCalled = false;
    let cocoCalled = false;
    const originalLog = console.log;
    console.log = (msg) => {
        if (typeof msg === 'string') {
            if (msg.includes('luna')) lunaCalled = true;
            if (msg.includes('coco')) cocoCalled = true;
        }
    };
    daisy(MAX_DEPTH - 3);
    console.log = originalLog;
    assert.ok(lunaCalled, 'luna should be called from daisy');
    assert.ok(cocoCalled, 'coco should be called from daisy');
});

test('recursion chain: coco > ginger > daisy', () => {
    let gingerCalled = false;
    let daisyCalled = false;
    const originalLog = console.log;
    console.log = (msg) => {
        if (typeof msg === 'string') {
            if (msg.includes('ginger')) gingerCalled = true;
            if (msg.includes('daisy')) daisyCalled = true;
        }
    };
    coco(MAX_DEPTH - 3);
    console.log = originalLog;
    assert.ok(gingerCalled, 'ginger should be called from coco');
    assert.ok(daisyCalled, 'daisy should be called from ginger (recursion)');
});

test('depth limit on recursion chain', () => {
    assert.strictEqual(coco(MAX_DEPTH), 'coco');
    assert.strictEqual(ginger(MAX_DEPTH), 'ginger');
    assert.strictEqual(daisy(MAX_DEPTH), 'daisy');
});
