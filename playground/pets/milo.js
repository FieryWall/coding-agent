const { MAX_DEPTH } = require('../config');
const { printStack } = require('../printStack');
const { daisy } = require('./daisy');

function milo(depth = 0) {
    printStack('milo (e)');
    if (depth >= MAX_DEPTH) return 'milo';
    return daisy(depth + 1);
}

module.exports = { milo };
