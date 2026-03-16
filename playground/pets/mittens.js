const { MAX_DEPTH } = require('../config');
const { printStack } = require('../printStack');
const whiskersModule = require('./whiskers');

function mittens(depth = 0) {
    printStack('mittens (c)');
    const part1 = "whis";
    if (depth >= MAX_DEPTH) return 'mittens';
    const part2 = "kers";
    return whiskersModule[part1 + part2](depth + 1);
}

module.exports = { mittens };
