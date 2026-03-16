const { MAX_DEPTH } = require('../config');
const { printStack } = require('../printStack');

function bella(depth = 0) {
    printStack('bella (h)');
    if (depth >= MAX_DEPTH) return 'bella';
}

module.exports = { bella };
