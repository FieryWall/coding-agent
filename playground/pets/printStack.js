function printStack(name) {
    const stack = new Error().stack;
    const cleaned = stack.split('\n').slice(1, 6)
        .map(line => line.replace(/\(.*[/\\]([^/\\]+:\d+:\d+)\)/, '($1)'))
        .join('\n');
    console.log(`\n=== ${name} called ===`);
    console.log(cleaned);
}

module.exports = { printStack };
