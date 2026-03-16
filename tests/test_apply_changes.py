import tempfile
import os
import pytest
from coding_agent.tools import apply_changes, FileChange, apply_content_changes, ContentChange


class TestApplyChanges:
    def test_single_change(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write("line 1\nline 2\nline 3\nline 4\nline 5\n")
            path = f.name
        
        try:
            changes = [FileChange(start_line=2, end_line=2, content="modified line 2")]
            result = apply_changes(path, changes)
            
            assert "modified line 2" in result
            assert "line 1" in result
            assert "line 3" in result
        finally:
            os.unlink(path)

    def test_multiple_changes_applied_from_bottom(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write("line 1\nline 2\nline 3\nline 4\nline 5\n")
            path = f.name
        
        try:
            changes = [
                FileChange(start_line=2, end_line=2, content="new line 2"),
                FileChange(start_line=4, end_line=4, content="new line 4"),
            ]
            result = apply_changes(path, changes)
            
            lines = result.split('\n')
            assert lines[1] == "new line 2"
            assert lines[3] == "new line 4"
        finally:
            os.unlink(path)

    def test_replace_multiple_lines(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write("line 1\nline 2\nline 3\nline 4\nline 5\n")
            path = f.name
        
        try:
            changes = [FileChange(start_line=2, end_line=4, content="replaced\nblock")]
            result = apply_changes(path, changes)
            
            lines = result.strip().split('\n')
            assert len(lines) == 4
            assert lines[0] == "line 1"
            assert lines[1] == "replaced"
            assert lines[2] == "block"
            assert lines[3] == "line 5"
        finally:
            os.unlink(path)

    def test_expand_single_line_to_multiple(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write("line 1\nline 2\nline 3\n")
            path = f.name
        
        try:
            changes = [FileChange(start_line=2, end_line=2, content="expanded\nto\nmultiple\nlines")]
            result = apply_changes(path, changes)
            
            lines = result.strip().split('\n')
            assert len(lines) == 6
        finally:
            os.unlink(path)

    def test_order_independence(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write("a\nb\nc\nd\ne\n")
            path = f.name
        
        try:
            changes = [
                FileChange(start_line=4, end_line=4, content="D"),
                FileChange(start_line=2, end_line=2, content="B"),
            ]
            result = apply_changes(path, changes)
            
            lines = result.strip().split('\n')
            assert lines[1] == "B"
            assert lines[3] == "D"
        finally:
            os.unlink(path)


class TestApplyContentChanges:
    def test_single_content_change(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write("const { max } = require('./max');\nmax();\n")
            path = f.name
        
        try:
            changes = [ContentChange(original="const { max }", new_content="const { min }")]
            result = apply_content_changes(path, changes)
            
            assert "const { min }" in result
            assert "const { max }" not in result
        finally:
            os.unlink(path)

    def test_multiple_content_changes(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write("function max() {}\nmax();\nmodule.exports = { max };\n")
            path = f.name
        
        try:
            changes = [
                ContentChange(original="function max()", new_content="function min()"),
                ContentChange(original="max();", new_content="min();"),
                ContentChange(original="{ max }", new_content="{ min }"),
            ]
            result = apply_content_changes(path, changes)
            
            assert "function min()" in result
            assert "min();" in result
            assert "{ min }" in result
            assert "max" not in result
        finally:
            os.unlink(path)

    def test_content_change_preserves_similar_names(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write("const MAX_DEPTH = 10;\nfunction max() {}\n")
            path = f.name
        
        try:
            changes = [ContentChange(original="function max()", new_content="function min()")]
            result = apply_content_changes(path, changes)
            
            assert "MAX_DEPTH" in result
            assert "function min()" in result
        finally:
            os.unlink(path)

    def test_content_change_multiline(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write("function max() {\n    return 1;\n}\n")
            path = f.name
        
        try:
            changes = [ContentChange(
                original="function max() {\n    return 1;\n}",
                new_content="function min() {\n    return 2;\n}"
            )]
            result = apply_content_changes(path, changes)
            
            assert "function min()" in result
            assert "return 2;" in result
        finally:
            os.unlink(path)

    def test_changes_applied_from_end(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write("aaa\nbbb\naaa\n")
            path = f.name
        
        try:
            changes = [
                ContentChange(original="aaa", new_content="AAA"),
                ContentChange(original="bbb", new_content="BBB"),
            ]
            result = apply_content_changes(path, changes)
            
            assert "AAA\nBBB\nAAA" in result or result.strip() == "AAA\nBBB\naaa"
        finally:
            os.unlink(path)
