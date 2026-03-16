import argparse
import importlib.util
import sys
import time
from collections import deque
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values: list) -> Optional[TreeNode]:
    if not values:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def load_solution():
    spec = importlib.util.spec_from_file_location(
        "solution",
        __file__.replace("runner.py", "solution.py"),
    )
    module = importlib.util.module_from_spec(spec)
    module.TreeNode = TreeNode
    module.Optional = Optional
    spec.loader.exec_module(module)
    return module.Solution()


def fmt_time(ms: float) -> str:
    if ms < 1.0:
        return f"{ms * 1000:.2f} μs"
    return f"{ms:.3f} ms"


def format_values(values: list) -> str:
    parts = ["null" if v is None else str(v) for v in values]
    return "[" + ",".join(parts) + "]"


TEST_CASES = [
    ([2, 1, 3], True, "Official example: left (1) < root (2) < right (3)"),
    ([5, 1, 4, None, None, 3, 6], False, "Official example: right child (4) < root (5)"),
    ([], True, "Empty tree"),
    ([1], True, "Single node"),
    ([2147483647], True, "INT_MAX boundary value"),
    ([-2147483648], True, "INT_MIN boundary value"),
    ([1, 1], False, "Duplicate: left child equals parent"),
    ([10, 5, 15, None, None, 6, 20], False, "Tricky: 6 in right subtree but 6 < root (10)"),
    ([5, 4, 6, None, None, 3, 7], False, "3 violates upper bound of ancestor 5"),
    ([1, None, 1], False, "Right child equals root"),
    ([0, -2147483648, 2147483647], True, "Full int range"),
    ([-2147483648, None, 2147483647], True, "Extreme values in right chain"),
    ([3, 1, 5, 0, 2, 4, 6], True, "Complete valid BST"),
    ([5, 4, None], True, "Left child only, less than parent"),
    ([5, None, 4], False, "Right child less than parent"),
]


def run_test(solution, values, expected, note, repeat):
    root = build_tree(values)
    label = format_values(values)

    times = []
    result = None
    error = None

    for _ in range(repeat):
        root = build_tree(values)
        try:
            start = time.perf_counter()
            result = solution.isValidBST(root)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        except NotImplementedError:
            raise
        except Exception as e:
            error = e
            break

    if error is not None:
        print(f"[ERROR] {label} → {expected}")
        print(f"        {type(error).__name__}: {error}")
        print(f"        {note}")
        return False, 0.0

    passed = result == expected
    total_ms = sum(times)

    if passed:
        timing = f"  ({fmt_time(total_ms)})"
        print(f"[PASS] {label} → {result}{timing}")
        print(f"       {note}")
    else:
        timing = f"  ({fmt_time(total_ms)})"
        print(f"[FAIL] {label} → {expected}  (got: {result}){timing}")
        print(f"       {note}")

    return passed, total_ms


def main():
    parser = argparse.ArgumentParser(description="LeetCode 98: Validate Binary Search Tree runner")
    parser.add_argument("-r", "--repeat", type=int, default=100_000, metavar="N",
                        help="Run each test N times and show total ms (default: 100000)")
    args = parser.parse_args()

    print("=" * 60)
    print("LeetCode 98. Validate Binary Search Tree")
    print(f"Running {len(TEST_CASES)} test cases")
    if args.repeat > 1:
        print(f"Benchmark mode: {args.repeat} repetitions per test")
    print("=" * 60)
    print()

    try:
        solution = load_solution()
    except Exception as e:
        print(f"[ERROR] Failed to load solution.py: {type(e).__name__}: {e}")
        sys.exit(1)

    passed_count = 0
    failed_count = 0
    not_implemented = False
    all_times = []

    for values, expected, note in TEST_CASES:
        try:
            ok, avg_ms = run_test(solution, values, expected, note, args.repeat)
            all_times.append(avg_ms)
            if ok:
                passed_count += 1
            else:
                failed_count += 1
        except NotImplementedError:
            not_implemented = True
            break
        print()

    print("=" * 60)

    if not_implemented:
        print("Solution not implemented — fill in isValidBST() in solution.py")
        print("=" * 60)
        sys.exit(1)

    total = passed_count + failed_count
    overall_avg = sum(all_times) / len(all_times) if all_times else 0.0
    print(f"Results: {passed_count}/{total} passed  |  {failed_count} failed  |  avg total per test ({args.repeat} runs): {fmt_time(overall_avg)}")
    print("=" * 60)

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
