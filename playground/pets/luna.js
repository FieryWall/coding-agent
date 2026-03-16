const { MAX_DEPTH } = require('../config');
const { printStack } = require('../printStack');
const { bella } = require('./bella');

class Luna {
    call(depth = 0) {
        printStack('luna (g)');
        if (depth >= MAX_DEPTH) return 'luna';
        return bella(depth + 1);
    }
}

const luna = (depth) => new Luna().call(depth);

module.exports = { luna, Luna };
