"""Tests for agent/skill_utils.py — extract_skill_conditions metadata handling."""

import logging

import pytest

from agent.skill_utils import extract_skill_conditions, get_skill_model


def test_metadata_as_dict_with_hermes():
    """Normal case: metadata is a dict containing hermes keys."""
    frontmatter = {
        "metadata": {
            "hermes": {
                "fallback_for_toolsets": ["toolset_a"],
                "requires_toolsets": ["toolset_b"],
                "fallback_for_tools": ["tool_x"],
                "requires_tools": ["tool_y"],
            }
        }
    }
    result = extract_skill_conditions(frontmatter)
    assert result["fallback_for_toolsets"] == ["toolset_a"]
    assert result["requires_toolsets"] == ["toolset_b"]
    assert result["fallback_for_tools"] == ["tool_x"]
    assert result["requires_tools"] == ["tool_y"]


def test_metadata_as_string_does_not_crash():
    """Bug case: metadata is a non-dict truthy value (e.g. a YAML string)."""
    frontmatter = {"metadata": "some text"}
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": [],
    }


def test_metadata_as_none():
    """metadata key is present but set to null/None."""
    frontmatter = {"metadata": None}
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": [],
    }


def test_metadata_missing_entirely():
    """metadata key is absent from frontmatter."""
    frontmatter = {"name": "my-skill", "description": "Does stuff."}
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": [],
    }


# ── get_skill_model() tests ───────────────────────────────────────────────


def test_get_skill_model_no_config_no_frontmatter(monkeypatch, tmp_path):
    """No config, no frontmatter → None without exception."""
    monkeypatch.setattr("agent.skill_utils.get_config_path", lambda: tmp_path / "nonexistent.yaml")
    assert get_skill_model("rai-unknown") is None


def test_get_skill_model_system_default_fast(monkeypatch, tmp_path):
    """preferred_model=fast in frontmatter, no operator tiers → system default."""
    monkeypatch.setattr("agent.skill_utils.get_config_path", lambda: tmp_path / "nonexistent.yaml")
    fm = {"metadata": {"hermes": {"preferred_model": "fast"}}}
    assert get_skill_model("rai-session-start", fm) == "anthropic/claude-haiku-4-5"


def test_get_skill_model_system_default_balanced(monkeypatch, tmp_path):
    """preferred_model=balanced in frontmatter → system default."""
    monkeypatch.setattr("agent.skill_utils.get_config_path", lambda: tmp_path / "nonexistent.yaml")
    fm = {"metadata": {"hermes": {"preferred_model": "balanced"}}}
    assert get_skill_model("rai-story-implement", fm) == "anthropic/claude-sonnet-4-6"


def test_get_skill_model_config_override_via_tier(monkeypatch, tmp_path):
    """skills.models override + operator tier → resolved model ID."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "skills:\n"
        "  tiers:\n"
        "    fast: anthropic/claude-haiku-4-5\n"
        "  models:\n"
        "    rai-session-start: fast\n"
    )
    monkeypatch.setattr("agent.skill_utils.get_config_path", lambda: cfg)
    assert get_skill_model("rai-session-start") == "anthropic/claude-haiku-4-5"


def test_get_skill_model_direct_model_id(monkeypatch, tmp_path):
    """skills.models with direct model ID (contains '/') → pass-through."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "skills:\n"
        "  models:\n"
        "    rai-bugfix-fix: openrouter/qwen/qwen3-30b-a3b:free\n"
    )
    monkeypatch.setattr("agent.skill_utils.get_config_path", lambda: cfg)
    assert get_skill_model("rai-bugfix-fix") == "openrouter/qwen/qwen3-30b-a3b:free"


def test_get_skill_model_unknown_tier_warns(monkeypatch, tmp_path, caplog):
    """Unknown tier → warning emitted, None returned."""
    monkeypatch.setattr("agent.skill_utils.get_config_path", lambda: tmp_path / "nonexistent.yaml")
    fm = {"metadata": {"hermes": {"preferred_model": "ultra"}}}
    with caplog.at_level(logging.WARNING, logger="agent.skill_utils"):
        result = get_skill_model("rai-unknown", fm)
    assert result is None
    assert "ultra" in caplog.text


def test_get_skill_model_config_override_beats_frontmatter(monkeypatch, tmp_path):
    """Config override takes priority over frontmatter preferred_model."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "skills:\n"
        "  models:\n"
        "    rai-session-start: anthropic/claude-opus-4-7\n"
    )
    monkeypatch.setattr("agent.skill_utils.get_config_path", lambda: cfg)
    fm = {"metadata": {"hermes": {"preferred_model": "fast"}}}
    assert get_skill_model("rai-session-start", fm) == "anthropic/claude-opus-4-7"


def test_get_skill_model_malformed_config_no_exception(monkeypatch, tmp_path):
    """Malformed config.yaml → None without exception."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(":: invalid: yaml: [[\n")
    monkeypatch.setattr("agent.skill_utils.get_config_path", lambda: cfg)
    assert get_skill_model("rai-unknown") is None
