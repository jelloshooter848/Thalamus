import pytest
from typer.testing import CliRunner

from thalamus.brain import create_brain
from thalamus.cli import app
from thalamus.config import MissingJevKeyError, Settings, load_settings


def test_missing_jev_key_refuses_to_start(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(MissingJevKeyError, match="requires a TypeSafe API key"):
        create_brain(Settings())


def test_cli_exits_with_clear_message_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env to pick up
    result = CliRunner().invoke(app, ["chat"])
    assert result.exit_code == 2
    assert "requires a TypeSafe API key" in result.output


def test_toml_overrides_nested_settings(tmp_path):
    path = tmp_path / "thalamus.toml"
    path.write_text('[models]\ndeep = "claude-sonnet-5-5"\n[thresholds]\nnogo = 0.9\n')
    settings = load_settings(path)
    assert settings.models.deep == "claude-sonnet-5-5"
    assert settings.thresholds.nogo == 0.9
    assert settings.models.fast == "claude-haiku-4-5"


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "thalamus.toml"
    path.write_text("[models]\ntypo = 1\n")
    with pytest.raises(ValueError, match="typo"):
        load_settings(path)
