const { MAX_DEPTH } = require('../config');
const { printStack } = require('../printStack');
const { rocky } = require('./rocky');

function sparky(depth = 0) {
    printStack('sparky (i)');
    if (depth >= MAX_DEPTH) return 'sparky';
    return rocky(depth + 1);
}

module.exports = { sparky };
