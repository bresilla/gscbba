import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("gscbba")
except PackageNotFoundError:
    manifest = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with manifest.open("rb") as stream:
        __version__ = tomllib.load(stream)["project"]["version"]
