from pdf_chunking import chunk_pages, split_sentences, strip_repeated_lines


def test_split_sentences_collapses_whitespace_and_splits_on_punctuation():
    text = "Hello   world.\nThis is\na test! Is it?"
    assert split_sentences(text) == ["Hello world.", "This is a test!", "Is it?"]


def test_split_sentences_empty_text():
    assert split_sentences("") == []


def test_strip_repeated_lines_drops_headers_present_on_most_pages():
    pages = [
        "Company Confidential Report\nPage content one.",
        "Company Confidential Report\nPage content two.",
        "Company Confidential Report\nPage content three.",
    ]
    result = strip_repeated_lines(pages)
    assert all("Company Confidential Report" not in p for p in result)
    assert "Page content one." in result[0]


def test_strip_repeated_lines_keeps_short_repeated_tokens():
    pages = ["and\nReal content here.", "and\nMore real content."]
    result = strip_repeated_lines(pages)
    assert "and" in result[0]


def test_strip_repeated_lines_single_page_is_noop():
    pages = ["Only one page."]
    assert strip_repeated_lines(pages) == pages


def test_chunk_pages_tracks_source_page():
    pages = ["First page sentence one. First page sentence two.", "Second page sentence."]
    chunks = chunk_pages(pages, chunk_size=8, overlap=0)
    assert chunks[0][1] == 1
    assert any(page == 2 for _, page in chunks)


def test_chunk_pages_splits_on_size_with_overlap():
    long_sentence_words = 20
    pages = [" ".join(f"word{i}." for i in range(10))]
    chunks = chunk_pages(pages, chunk_size=long_sentence_words, overlap=5)
    assert len(chunks) >= 1
    assert all(isinstance(text, str) and isinstance(page, int) for text, page in chunks)
