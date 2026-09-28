from __future__ import annotations

import unittest

from textual.app import App

from core.runtime.tui_widgets import Choice, SelectionScreen


class _Host(App[None]):
    def __init__(self, screen: SelectionScreen) -> None:
        super().__init__()
        self._screen = screen
        self.result: object = "unset"

    def on_mount(self) -> None:
        self.push_screen(self._screen, self._store)

    def _store(self, value: object) -> None:
        self.result = value


def _choices() -> list[Choice[str]]:
    return [
        Choice("alpha", "Alpha", detail="first"),
        Choice("beta", "Beta", detail="second"),
        Choice("gamma", "Gamma", detail="third"),
        Choice("delta", "Delta", detail="fourth", disabled=True),
    ]


class SelectionScreenTest(unittest.IsolatedAsyncioTestCase):
    async def test_filter_then_enter_selects(self) -> None:
        app = _Host(SelectionScreen(_choices(), title="Pick"))
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertIsInstance(app.screen, SelectionScreen)
            await pilot.press("g", "a", "m")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
        self.assertEqual(app.result, "gamma")

    async def test_escape_cancels(self) -> None:
        app = _Host(SelectionScreen(_choices(), title="Pick"))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
        self.assertIsNone(app.result)

    async def test_number_accelerator_selects(self) -> None:
        app = _Host(SelectionScreen(_choices(), title="Pick"))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("2")
            await pilot.pause()
        self.assertEqual(app.result, "beta")

    async def test_arrow_navigation_and_enter(self) -> None:
        app = _Host(SelectionScreen(_choices(), title="Pick"))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
        self.assertEqual(app.result, "beta")

    async def test_disabled_choice_is_not_selectable(self) -> None:
        app = _Host(SelectionScreen(_choices(), title="Pick"))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("d", "e", "l", "t", "a")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
        self.assertEqual(app.result, "unset")

    async def test_current_marker_and_custom_entry(self) -> None:
        app = _Host(
            SelectionScreen(
                _choices(),
                title="Pick",
                current="alpha",
                allow_custom=True,
                custom_placeholder="Type custom",
            )
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            # Filter so only the custom row is reachable via Enter.
            await pilot.press("z", "z", "z")
            await pilot.pause()
            await pilot.press("enter")  # highlights custom fallback
            await pilot.pause()
            await pilot.press("c", "u", "s", "t", "o", "m")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
        self.assertEqual(app.result, "custom")


if __name__ == "__main__":
    unittest.main()
