from __future__ import annotations

import os
import unittest
from unittest import mock

from core.runtime import connectivity


class ConnectivityTest(unittest.TestCase):
    def setUp(self) -> None:
        connectivity.clear_cache()

    def tearDown(self) -> None:
        connectivity.clear_cache()

    def test_force_offline_and_online_overrides(self) -> None:
        with mock.patch.dict(os.environ, {"DEVENV_FORCE_OFFLINE": "1"}):
            self.assertFalse(connectivity.internet_available(refresh=True))
            self.assertTrue(connectivity.is_offline(refresh=True))
        with mock.patch.dict(os.environ, {"DEVENV_FORCE_ONLINE": "1"}):
            self.assertTrue(connectivity.internet_available(refresh=True))

    def test_probe_uses_injected_hosts(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"DEVENV_FORCE_OFFLINE": "", "DEVENV_FORCE_ONLINE": "", "DEVENV_CONNECTIVITY_HOSTS": "example.test"},
        ):
            with mock.patch.object(connectivity, "_reachable", return_value=True) as probe:
                self.assertTrue(connectivity.internet_available(refresh=True))
            probe.assert_called_once_with("example.test")

    def test_offline_when_all_probes_fail(self) -> None:
        with mock.patch.dict(os.environ, {"DEVENV_FORCE_OFFLINE": "", "DEVENV_FORCE_ONLINE": ""}):
            with mock.patch.object(connectivity, "_reachable", return_value=False):
                self.assertFalse(connectivity.internet_available(refresh=True))

    def test_result_is_cached(self) -> None:
        with mock.patch.dict(os.environ, {"DEVENV_FORCE_OFFLINE": "", "DEVENV_FORCE_ONLINE": "", "DEVENV_CONNECTIVITY_TTL": "60"}):
            with mock.patch.object(connectivity, "_reachable", return_value=True) as probe:
                connectivity.internet_available(refresh=True)
                connectivity.internet_available()
                self.assertEqual(probe.call_count, 1)

    def test_local_backend_priority_default_and_override(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEVENV_OFFLINE_BACKENDS", None)
            self.assertEqual(connectivity.local_backend_priority(), ("ollama", "llama_cpp"))
        with mock.patch.dict(os.environ, {"DEVENV_OFFLINE_BACKENDS": "llama_cpp"}):
            self.assertEqual(connectivity.local_backend_priority(), ("llama_cpp",))


class RoutingOfflineTest(unittest.TestCase):
    def _core(self):
        from core.ai.routing import RoutingAICore

        class _Backend:
            model = "m"

            def register_tool(self, tool):  # pragma: no cover
                return None

            def status(self):  # pragma: no cover
                return None

        return RoutingAICore(
            workspace_path=".",
            opencode_ai=_Backend(),
            ollama_ai=_Backend(),
            llama_cpp_ai=_Backend(),
            codex_ai=_Backend(),
        )

    def test_apply_offline_disables_remote_and_picks_local(self) -> None:
        core = self._core()
        core.opencode_enabled = True
        core.codex_enabled = True
        core.ollama_enabled = True
        core.llama_cpp_enabled = True
        core.preferred_backend = "opencode"

        receipt = core.apply_offline_mode(offline=True)

        self.assertFalse(core.opencode_enabled)
        self.assertFalse(core.codex_enabled)
        self.assertEqual(core.preferred_backend, "ollama")
        self.assertEqual(receipt["previous_backend"], "opencode")
        self.assertTrue(receipt["local_backend_available"])

    def test_apply_offline_without_local_backend(self) -> None:
        core = self._core()
        core.opencode_enabled = True
        core.ollama_enabled = False
        core.llama_cpp_enabled = False
        receipt = core.apply_offline_mode(offline=True)
        self.assertFalse(receipt["local_backend_available"])
        self.assertIn("no local backend", core.last_backend_reason.lower())

    def test_apply_offline_false_is_noop(self) -> None:
        core = self._core()
        core.preferred_backend = "opencode"
        receipt = core.apply_offline_mode(offline=False)
        self.assertFalse(receipt["changed"])


if __name__ == "__main__":
    unittest.main()
