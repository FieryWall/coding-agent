const { MAX_DEPTH } = require('../config');
const { printStack } = require('../printStack');

function whiskers(depth = 0) {
    printStack('whiskers (d)');
    if (depth >= MAX_DEPTH) return 'whiskers';
}

module.exports = { whiskers };
