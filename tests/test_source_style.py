import ast
import io
import tokenize
from pathlib import Path


def test_python_source_has_no_comments_or_docstrings():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src").rglob("*.py")) + sorted(
        (root / "tests").rglob("*.py")
    )
    assert paths
    for path in paths:
        source = path.read_text()
        assert not any(
            token.type == tokenize.COMMENT
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
        ), path
        assert not any(
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            for node in ast.walk(ast.parse(source))
        ), path
