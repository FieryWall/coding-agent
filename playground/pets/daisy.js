const { MAX_DEPTH } = require('../config');
const { printStack } = require('../printStack');
const { luna } = require('./luna');

function daisy(depth = 0) {
    printStack('daisy (f)');
    if (depth >= MAX_DEPTH) return 'daisy';
    luna(depth + 1);
    const { coco } = require('./coco');
    return coco(depth + 1);
}

module.exports = { daisy };
