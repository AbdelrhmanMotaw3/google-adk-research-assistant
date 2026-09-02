from research_assistant.agents.sources import _derive_top_sources


def test_top_sources_derived_and_deduped():
    search_results = [
        {
            "source_type": "web_like",
            "results": [
                {"title": "A", "url": "https://a.com", "snippet": "x", "relevance": 0.9},
                {"title": "A dup", "url": "https://a.com", "snippet": "y", "relevance": 0.8},
            ],
            "total_found": 2,
        },
        {
            "source_type": "papers_like",
            "results": [
                {"title": "B", "url": "https://b.org", "abstract": "z", "relevance": 0.95},
            ],
            "total_found": 1,
        },
    ]
    top = _derive_top_sources(search_results, limit=10)
    urls = [s["url"] for s in top]
    assert urls.count("https://a.com") == 1
    assert "https://b.org" in urls

