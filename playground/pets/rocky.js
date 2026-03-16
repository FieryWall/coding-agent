const { MAX_DEPTH } = require('../config');
const { printStack } = require('../printStack');
const { charlie } = require('./charlie');

function rocky(depth = 0) {
    printStack('rocky (j)');
    if (depth >= MAX_DEPTH) return 'rocky';
    return charlie(depth + 1);
}

module.exports = { rocky };
