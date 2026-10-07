import pytest

from g1_cbf.cli import _parser
from g1_cbf.config import RunConfig


def test_run_defaults():
    a = _parser().parse_args(["run"])
    assert a.mode == "both" and a.alpha == 3.5 and not a.viewer


def test_command_required():
    with pytest.raises(SystemExit):
        _parser().parse_args([])


def test_with_filter_returns_modified_copy():
    base = RunConfig()
    off = base.with_filter(False)
    assert base.safety.enabled and not off.safety.enabled
