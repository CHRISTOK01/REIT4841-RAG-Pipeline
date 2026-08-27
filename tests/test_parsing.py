from generation.parsing import parse_generation_output

RAW_TEXT = """## Summary
Adult ALL is a cancer of the blood and bone marrow that requires prompt treatment.

## Key Points
1. Adult ALL causes the bone marrow to make too many lymphocytes. [1]
2. Common symptoms include fever, fatigue, and easy bruising. [2][3]
3. There is no standard staging system for adult ALL.

## Disclaimer
This summary is not a substitute for professional medical advice.
"""

PASSAGE_MAP = {1: "CancerGov:doc1:q1#0", 2: "CancerGov:doc1:q2#0", 3: "CancerGov:doc1:q2#1"}


def test_parses_all_three_sections():
    output = parse_generation_output(RAW_TEXT, PASSAGE_MAP)
    assert "cancer of the blood and bone marrow" in output.summary
    assert "not a substitute for professional medical advice" in output.disclaimer
    assert len(output.key_points) == 3


def test_key_point_citations_resolve_to_chunk_ids():
    output = parse_generation_output(RAW_TEXT, PASSAGE_MAP)
    first, second, third = output.key_points

    assert first.cited_passage_numbers == [1]
    assert first.cited_chunk_ids == ["CancerGov:doc1:q1#0"]

    assert second.cited_passage_numbers == [2, 3]
    assert second.cited_chunk_ids == ["CancerGov:doc1:q2#0", "CancerGov:doc1:q2#1"]

    assert third.cited_passage_numbers == []
    assert third.cited_chunk_ids == []


def test_hallucinated_citation_number_is_dropped_from_chunk_ids_but_kept_as_a_flag():
    raw = """## Summary
s

## Key Points
1. A claim citing a passage that was never provided. [7]

## Disclaimer
d
"""
    output = parse_generation_output(raw, PASSAGE_MAP)  # PASSAGE_MAP only has 1, 2, 3
    point = output.key_points[0]
    assert point.cited_passage_numbers == [7]  # visible for a "did it hallucinate a citation?" check
    assert point.cited_chunk_ids == []  # but it can't be resolved to a real chunk


def test_missing_sections_default_to_empty():
    output = parse_generation_output("no headings here at all", {})
    assert output.summary == ""
    assert output.disclaimer == ""
    assert output.key_points == []


def test_accepts_both_dot_and_paren_numbering():
    raw = """## Summary
s

## Key Points
1. dot style [1]
2) paren style [2]

## Disclaimer
d
"""
    output = parse_generation_output(raw, {1: "a", 2: "b"})
    assert len(output.key_points) == 2
    assert output.key_points[0].text.startswith("dot style")
    assert output.key_points[1].text.startswith("paren style")


def test_raw_text_preserved_verbatim():
    output = parse_generation_output(RAW_TEXT, PASSAGE_MAP)
    assert output.raw_text == RAW_TEXT
