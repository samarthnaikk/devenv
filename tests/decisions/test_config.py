from __future__ import annotations

import unittest

from core.decisions.config import GATE_INTENT, GATE_MEMORY_WRITE, DecisionConfig
from core.decisions.models import DecisionMode


class DecisionConfigTest(unittest.TestCase):
    def test_defaults_preserve_current_behavior(self) -> None:
        config = DecisionConfig.from_env(env={})
        self.assertEqual(config.provider, "off")
        self.assertIs(config.intent_mode, DecisionMode.OFF)
        self.assertIs(config.memory_write_mode, DecisionMode.OFF)
        self.assertFalse(config.enabled_for(GATE_INTENT))
        self.assertFalse(config.enabled_for(GATE_MEMORY_WRITE))

    def test_provider_and_gate_modes(self) -> None:
        config = DecisionConfig.from_env(
            env={
                "DEVENV_DECISION_PROVIDER": "typesafe",
                "DEVENV_DECISION_MODE": "shadow",
                "DEVENV_DECISION_MEMORY_WRITE": "enforce",
                "TYPESAFE_API_KEY": "secret-key",
            }
        )
        self.assertEqual(config.provider, "typesafe")
        self.assertIs(config.intent_mode, DecisionMode.SHADOW)
        self.assertIs(config.memory_write_mode, DecisionMode.ENFORCE)
        self.assertEqual(config.api_key, "secret-key")
        self.assertTrue(config.enabled_for(GATE_INTENT))
        self.assertTrue(config.enabled_for(GATE_MEMORY_WRITE))

    def test_unknown_provider_falls_back_to_off(self) -> None:
        config = DecisionConfig.from_env(env={"DEVENV_DECISION_PROVIDER": "wat"})
        self.assertEqual(config.provider, "off")

    def test_numeric_clamping_and_parsing(self) -> None:
        config = DecisionConfig.from_env(
            env={
                "DEVENV_DECISION_TIMEOUT": "-4",
                "DEVENV_DECISION_RETRIES": "3.9",
                "DEVENV_DECISION_MIN_CONFIDENCE": "2.0",
                "DEVENV_DECISION_REDACT": "0",
            }
        )
        self.assertEqual(config.timeout, 0.0)
        self.assertEqual(config.retries, 3)
        self.assertEqual(config.min_confidence, 1.0)
        self.assertFalse(config.redact)

    def test_is_remote(self) -> None:
        self.assertTrue(DecisionConfig.from_env(env={"DEVENV_DECISION_PROVIDER": "typesafe"}).is_remote)
        self.assertFalse(
            DecisionConfig.from_env(
                env={
                    "DEVENV_DECISION_PROVIDER": "local",
                    "DEVENV_SYSTEMONE_BASE_URL": "http://127.0.0.1:8000",
                }
            ).is_remote
        )
        self.assertFalse(
            DecisionConfig.from_env(
                env={
                    "DEVENV_DECISION_PROVIDER": "auto",
                    "DEVENV_SYSTEMONE_BASE_URL": "http://localhost:8000",
                }
            ).is_remote
        )

    def test_summary(self) -> None:
        summary = DecisionConfig.from_env(env={}).summary()
        self.assertEqual(summary["provider"], "off")
        self.assertEqual(summary["intent_mode"], "off")


if __name__ == "__main__":
    unittest.main()
