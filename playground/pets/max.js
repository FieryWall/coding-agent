const { MAX_DEPTH } = require('../config');
const { printStack } = require('../printStack');
const { buddy } = require('./buddy');

function max(depth = 0) {
    printStack('max (a)');
    if (depth >= MAX_DEPTH) return 'max';
    return buddy(depth + 1);
}

module.exports = { max };
