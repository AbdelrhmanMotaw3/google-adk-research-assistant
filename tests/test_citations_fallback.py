from research_assistant.agents.citations import _derive_from_sources, _looks_placeholder


def test_placeholder_detection():
    assert _looks_placeholder("")
    assert _looks_placeholder("N/A")
    assert _looks_placeholder("Smith, J. (2020). Placeholder.")
    assert not _looks_placeholder("Some Org (n.d.). Title. https://example.com")


def test_derive_citations_has_urls():
    sources = [
        {"title": "Example Source", "url": "https://example.com", "source": "Example Org"},
        {"title": "Another Source", "url": "https://example.org", "publisher": "Example Publisher"},
    ]
    derived = _derive_from_sources(sources)
    assert derived["total_citations"] == 2
    assert "https://example.com" in derived["bibliography"]

