"""Tests for per-skill model override in the gateway.

Covers _apply_session_model_override consuming _pending_skill_model_overrides
(one-shot) and the interaction with existing /model session overrides.
"""

import threading
from unittest.mock import MagicMock, AsyncMock

import gateway.run as gateway_run


def _make_runner():
    """Minimal GatewayRunner with only the attributes touched by these tests."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_model_overrides = {}
    runner._pending_skill_model_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    return runner


class TestApplySessionModelOverride:
    def test_no_overrides_returns_config_model(self):
        runner = _make_runner()
        model, kwargs = runner._apply_session_model_override("s1", "default-model", {})
        assert model == "default-model"
        assert kwargs == {}

    def test_skill_override_replaces_model(self):
        runner = _make_runner()
        runner._pending_skill_model_overrides["s1"] = "anthropic/claude-haiku-4-5"
        model, kwargs = runner._apply_session_model_override("s1", "default-model", {})
        assert model == "anthropic/claude-haiku-4-5"

    def test_skill_override_is_consumed_once(self):
        """One-shot: second call for same session gets the config model."""
        runner = _make_runner()
        runner._pending_skill_model_overrides["s1"] = "anthropic/claude-haiku-4-5"
        runner._apply_session_model_override("s1", "default-model", {})
        # Second call — override must be gone
        model, _ = runner._apply_session_model_override("s1", "default-model", {})
        assert model == "default-model"
        assert "s1" not in runner._pending_skill_model_overrides

    def test_skill_override_does_not_affect_other_sessions(self):
        runner = _make_runner()
        runner._pending_skill_model_overrides["s1"] = "anthropic/claude-haiku-4-5"
        model, _ = runner._apply_session_model_override("s2", "default-model", {})
        assert model == "default-model"
        # s1 override must still be there
        assert "s1" in runner._pending_skill_model_overrides

    def test_skill_override_takes_precedence_over_session_override(self):
        """/model session override is present but skill override wins."""
        runner = _make_runner()
        runner._session_model_overrides["s1"] = {"model": "gpt-4o", "provider": "openai"}
        runner._pending_skill_model_overrides["s1"] = "anthropic/claude-haiku-4-5"
        model, _ = runner._apply_session_model_override("s1", "default-model", {})
        assert model == "anthropic/claude-haiku-4-5"

    def test_session_override_still_works_without_skill_override(self):
        """/model session override unaffected when no pending skill override."""
        runner = _make_runner()
        runner._session_model_overrides["s1"] = {
            "model": "gpt-4o",
            "provider": "openai",
            "api_key": "sk-test",
            "base_url": None,
            "api_mode": None,
        }
        model, kwargs = runner._apply_session_model_override("s1", "default-model", {})
        assert model == "gpt-4o"
        assert kwargs.get("provider") == "openai"

    def test_skill_override_does_not_clobber_runtime_kwargs(self):
        """Skill model swap leaves provider/api_key/etc. from config intact."""
        runner = _make_runner()
        runner._pending_skill_model_overrides["s1"] = "anthropic/claude-haiku-4-5"
        runtime = {"provider": "anthropic", "api_key": "sk-ant"}
        model, kwargs = runner._apply_session_model_override("s1", "default-model", runtime)
        assert model == "anthropic/claude-haiku-4-5"
        assert kwargs["provider"] == "anthropic"
        assert kwargs["api_key"] == "sk-ant"

    def test_no_override_for_unknown_session(self):
        runner = _make_runner()
        runner._pending_skill_model_overrides["other"] = "anthropic/claude-haiku-4-5"
        model, _ = runner._apply_session_model_override(None, "default-model", {})
        assert model == "default-model"
