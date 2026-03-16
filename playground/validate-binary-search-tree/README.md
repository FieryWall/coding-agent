# LeetCode 98. Validate Binary Search Tree — Python Sandbox

A local sandbox for solving [98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/).

## Structure

```
validate-binary-search-tree/
├── solution.py   # Your solution (implement isValidBST)
├── runner.py     # Runs tests and measures execution time
└── README.md
```

## Quick Start

```bash
# Run tests (100 000 repetitions by default)
python runner.py

# Custom number of repetitions
python runner.py --repeat 1000000
python runner.py -r 1
```

## How to Use

1. Open `solution.py`
2. Implement the `isValidBST` method inside the `Solution` class
3. Run `python runner.py`

## Test Cases

The runner includes 15 test cases covering:
- Official LeetCode examples
- Edge cases: empty tree, single node, duplicates, INT_MIN/INT_MAX boundaries
- Tricky cases: values that violate ancestor bounds

## Solution Template

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        raise NotImplementedError
```

`TreeNode` is provided by `runner.py` — no need to define it in `solution.py`.
