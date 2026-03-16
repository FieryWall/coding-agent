const { MAX_DEPTH } = require('./config');
const { max } = require('./pets/max');
const { buddy } = require('./pets/buddy');
const { mittens } = require('./pets/mittens');
const { whiskers } = require('./pets/whiskers');
const { sparky } = require('./pets/sparky');
const { rocky } = require('./pets/rocky');
const { charlie } = require('./pets/charlie');
const { milo } = require('./pets/milo');
const { daisy } = require('./pets/daisy');
const { luna } = require('./pets/luna');
const { bella } = require('./pets/bella');
const { coco } = require('./pets/coco');
const { ginger } = require('./pets/ginger');

console.log('='.repeat(50));
console.log('CHAIN 1: max > buddy > mittens > whiskers');
console.log('         + branch: buddy > sparky > rocky > charlie');
console.log('='.repeat(50));
max();

console.log('\n' + '='.repeat(50));
console.log('CHAIN 2: milo > daisy > luna > bella');
console.log('         + recursion: daisy > coco > ginger > daisy > ...');
console.log('='.repeat(50));
milo();

module.exports = {
    max, buddy, mittens, whiskers,
    sparky, rocky, charlie,
    milo, daisy, luna, bella,
    coco, ginger,
    MAX_DEPTH
};
