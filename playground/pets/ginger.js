const { MAX_DEPTH } = require('../config');
const { printStack } = require('../printStack');

function ginger(depth = 0) {
    printStack('ginger (m)');
    if (depth >= MAX_DEPTH) return 'ginger';
    const { daisy } = require('./daisy');
    return daisy.apply(this, [depth + 1]);
}

module.exports = { ginger };
