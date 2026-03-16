const { MAX_DEPTH } = require('../config');
const { printStack } = require('../printStack');
const { mittens } = require('./mittens');
const { sparky } = require('./sparky');

function buddy(depth = 0) {
    printStack('buddy (b)');
    if (depth >= MAX_DEPTH) return 'buddy';
    mittens(depth + 1);
    return sparky(depth + 1);
}

module.exports = { buddy };
