from __future__ import annotations

import importlib
from pathlib import Path


class FakeNode:
    def __init__(self, value: str = "", text: str = "") -> None:
        self._value = value
        self._text = text

    def get_attribute(self, name: str) -> str:
        return self._value if name == "value" else ""

    def inner_text(self) -> str:
        return self._text

    def is_disabled(self) -> bool:
        return False


class FakePage:
    """Minimal Playwright stand-in that records every click and screenshot."""

    def __init__(self, groups: list[str]) -> None:
        self.groups = groups
        self.clicks: list[str] = []
        self.selected: str | None = None
        self.url = "http://127.0.0.1:0/evaluation"

    def goto(self, url: str, **_kwargs) -> None:
        self.url = url

    def wait_for_selector(self, *_args, **_kwargs) -> None:
        return None

    def wait_for_timeout(self, *_args) -> None:
        return None

    def wait_for_function(self, *_args, **_kwargs) -> None:
        return None

    def wait_for_load_state(self, *_args, **_kwargs) -> None:
        return None

    def screenshot(self, **_kwargs) -> None:
        return None

    def query_selector_all(self, selector: str) -> list[FakeNode]:
        if selector == "#groupSelect option":
            return [FakeNode(value=name, text=name) for name in self.groups]
        return []

    def query_selector(self, _selector: str) -> FakeNode:
        return FakeNode()

    def eval_on_selector(self, _selector: str, _script: str) -> str:
        return self.selected or ""

    def select_option(self, _selector: str, value: str) -> None:
        self.selected = value

    def click(self, selector: str) -> None:
        self.clicks.append(selector)

    def fill(self, *_args) -> None:
        return None


def load_verifier():
    return importlib.import_module("scripts.verify_training_gui")


def test_verifier_never_writes_into_a_non_scratch_run_group(monkeypatch) -> None:
    verifier = load_verifier()
    monkeypatch.setattr(verifier, "CHECKS", [])
    page = FakePage(["real_group_a", "real_group_b"])

    verifier.verify_step7(page, "http://127.0.0.1:0", trained=False)

    # No report is written and no promotion is attempted when the scratch run is missing.
    assert "#writeReport" not in page.clicks
    assert "#promoteWeight" not in page.clicks
    names = {name: ok for name, ok, _ in verifier.CHECKS}
    assert names["step7: the scratch run group is available to evaluate"] is False
    assert names["step7: Write Evaluation Report writes the markdown report"] is False
    details = {name: detail for name, _, detail in verifier.CHECKS}
    assert "non-scratch run group" in details["step7: Write Evaluation Report writes the markdown report"]


def test_verifier_writes_only_when_the_scratch_group_is_present(monkeypatch) -> None:
    verifier = load_verifier()
    monkeypatch.setattr(verifier, "CHECKS", [])
    page = FakePage(["real_group_a", verifier.SCRATCH_RUN_GROUP])

    verifier.verify_step7(page, "http://127.0.0.1:0", trained=True)

    assert "#writeReport" in page.clicks
    assert page.selected == verifier.SCRATCH_RUN_GROUP
    names = {name: ok for name, ok, _ in verifier.CHECKS}
    assert names["step7: the scratch run group is available to evaluate"] is True
