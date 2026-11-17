"""v4 phase 1: the shuffled-ratings test is gone from the CLI, the config schema and the
package (DECISION(G6) retired)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from portstudy.config import StudyConfig
from portstudy.pipeline.cli import app


def test_cli_has_no_shuffle_command_or_skip_flag() -> None:
    runner = CliRunner()
    top = runner.invoke(app, ["--help"])
    assert top.exit_code == 0 and "shuffle" not in top.output
    result = runner.invoke(app, ["shuffle", "--help"])
    assert result.exit_code != 0
    eval_help = runner.invoke(app, ["eval", "--help"])
    assert eval_help.exit_code == 0 and "--skip-shuffle" not in eval_help.output


def test_eval_shuffle_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        StudyConfig.model_validate({"eval": {"shuffle": {"n_shuffles": 5}}})


def test_eval_shuffle_module_cannot_be_imported() -> None:
    with pytest.raises(ModuleNotFoundError):
        import portstudy.eval.shuffle  # noqa: F401
