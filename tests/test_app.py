from streamlit.testing.v1 import AppTest

from src.config import ROOT


def test_app_runs_without_exception():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 6
    assert len(at.metric) >= 5


def test_filter_changes_view():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    before = at.metric[0].value
    at.sidebar.multiselect[0].set_value(["CBD"]).run()
    assert not at.exception
    assert at.metric[0].value != before
