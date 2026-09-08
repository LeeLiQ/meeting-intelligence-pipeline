"""Tests for _sync_env_from_template."""

from pathlib import Path

import pytest

from main import _sync_env_from_template


class TestSyncEnvFromTemplate:
    """Tests for the env file synchronization helper."""

    def test_creates_env_from_template_when_missing(self, tmp_path: Path):
        """If .env doesn't exist, it should be copied from the template."""
        template = tmp_path / ".env.template"
        template.write_text("API_KEY=placeholder\nSECRET=placeholder\n")
        env = tmp_path / ".env"

        # _sync_env_from_template calls sys.exit(1) when creating a new .env
        with pytest.raises(SystemExit) as exc_info:
            _sync_env_from_template(env, template)

        assert exc_info.value.code == 1
        assert env.exists()
        assert env.read_text() == template.read_text()

    def test_does_nothing_when_template_missing(self, tmp_path: Path):
        """If the template doesn't exist, nothing should happen."""
        env = tmp_path / ".env"
        template = tmp_path / ".env.template"  # Does not exist

        # Should not raise, should not create anything
        _sync_env_from_template(env, template)

        assert not env.exists()

    def test_appends_missing_keys(self, tmp_path: Path):
        """New keys in template should be appended to existing .env."""
        template = tmp_path / ".env.template"
        template.write_text("KEY_A=val_a\nKEY_B=val_b\nKEY_C=val_c\n")

        env = tmp_path / ".env"
        env.write_text("KEY_A=my_value\nKEY_B=my_other_value\n")

        _sync_env_from_template(env, template)

        content = env.read_text()
        assert "KEY_A=my_value" in content  # Original preserved
        assert "KEY_B=my_other_value" in content  # Original preserved
        assert "KEY_C=val_c" in content  # New key appended

    def test_does_not_duplicate_existing_keys(self, tmp_path: Path):
        """Keys already in .env should not be duplicated."""
        template = tmp_path / ".env.template"
        template.write_text("API_KEY=placeholder\n")

        env = tmp_path / ".env"
        env.write_text("API_KEY=real_key\n")

        _sync_env_from_template(env, template)

        content = env.read_text()
        assert content.count("API_KEY") == 1

    def test_skips_comments_in_template(self, tmp_path: Path):
        """Comments in the template should not be treated as keys."""
        template = tmp_path / ".env.template"
        template.write_text("# This is a comment\nAPI_KEY=placeholder\n")

        env = tmp_path / ".env"
        env.write_text("API_KEY=real\n")

        _sync_env_from_template(env, template)

        content = env.read_text()
        assert content == "API_KEY=real\n"  # Nothing appended
