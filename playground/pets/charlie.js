const { MAX_DEPTH } = require('../config');
const { printStack } = require('../printStack');

function charlie(depth = 0) {
    printStack('charlie (k)');
    if (depth >= MAX_DEPTH) return 'charlie';
}

module.exports = { charlie };
