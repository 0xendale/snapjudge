from app.summaries import summarize


def test_summarize_short_text():
    assert summarize("The package arrived late and the box was damaged.")
