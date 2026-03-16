const { MAX_DEPTH } = require('../config');
const { printStack } = require('../printStack');
const { ginger } = require('./ginger');

function coco(depth = 0) {
    printStack('coco (l)');
    if (depth >= MAX_DEPTH) return 'coco';
    return ginger.call(this, depth + 1);
}

module.exports = { coco };
