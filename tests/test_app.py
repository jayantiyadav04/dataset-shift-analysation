"""Check the real placeholder renders and its sole selection works."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_placeholder_validates_both_configs_without_results():
    path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(path)).run()
    assert not app.exception
    assert "No completed experiments" in app.info[0].value
    assert "main_v1" in app.success[0].value
    app.selectbox[0].select("smoke").run()
    assert not app.exception
    assert "development_smoke" in app.success[0].value
    assert len(app.button) == 0
    assert len(app.metric) == 0
