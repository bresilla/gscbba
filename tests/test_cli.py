import pytest

from gscbba import __version__
from gscbba.__main__ import main


def test_cli_version(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_without_arguments_shows_help(capsys) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "--only" in output
    assert "--figures-only" in output


def test_cli_help(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "--quick" in capsys.readouterr().out
