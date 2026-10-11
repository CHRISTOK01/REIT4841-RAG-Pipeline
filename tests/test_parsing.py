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


def test_parses_all_three_sections_exactly():
    output = parse_generation_output(RAW_TEXT, PASSAGE_MAP)
    assert output.summary == "Adult ALL is a cancer of the blood and bone marrow that requires prompt treatment."
    assert output.disclaimer == "This summary is not a substitute for professional medical advice."
    assert [kp.text for kp in output.key_points] == [
        "Adult ALL causes the bone marrow to make too many lymphocytes. [1]",
        "Common symptoms include fever, fatigue, and easy bruising. [2][3]",
        "There is no standard staging system for adult ALL.",
    ]


def test_key_point_citations_resolve_to_chunk_ids():
    first, second, third = parse_generation_output(RAW_TEXT, PASSAGE_MAP).key_points
    assert (first.cited_passage_numbers, first.cited_chunk_ids) == ([1], ["CancerGov:doc1:q1#0"])
    assert (second.cited_passage_numbers, second.cited_chunk_ids) == (
        [2, 3],
        ["CancerGov:doc1:q2#0", "CancerGov:doc1:q2#1"],
    )
    assert (third.cited_passage_numbers, third.cited_chunk_ids) == ([], [])


def test_hallucinated_citation_number_is_dropped_from_chunk_ids_but_kept_as_a_flag():
    raw = "## Summary\ns\n\n## Key Points\n1. A claim citing a passage that was never provided. [7]\n\n## Disclaimer\nd\n"
    point = parse_generation_output(raw, PASSAGE_MAP).key_points[0]
    assert point.cited_passage_numbers == [7]
    assert point.cited_chunk_ids == []


def test_missing_sections_come_back_empty_for_the_caller_to_reject():
    """The parser never invents content; generation/generator.py turns an
    empty Summary or Key Points into an error."""
    output = parse_generation_output("no headings here at all", {})
    assert (output.summary, output.disclaimer, output.key_points) == ("", "", [])


def test_accepts_numbered_and_bulleted_key_points():
    raw = "## Summary\ns\n\n## Key Points\n1. dot style [1]\n2) paren style [2]\n- dash style\n* star style\n\n## Disclaimer\nd\n"
    output = parse_generation_output(raw, {1: "a", 2: "b"})
    assert [kp.text for kp in output.key_points] == ["dot style [1]", "paren style [2]", "dash style", "star style"]


def test_accepts_other_heading_levels_and_case():
    raw = "### SUMMARY\nthe summary\n\n# key points\n1. a point\n\n#### Disclaimer\nd\n"
    output = parse_generation_output(raw, {})
    assert output.summary == "the summary"
    assert [kp.text for kp in output.key_points] == ["a point"]
    assert output.disclaimer == "d"


def test_sections_do_not_bleed_into_each_other():
    output = parse_generation_output(RAW_TEXT, PASSAGE_MAP)
    assert "Key Points" not in output.summary and "Disclaimer" not in output.summary
    assert all("Disclaimer" not in kp.text and "substitute" not in kp.text for kp in output.key_points)


def test_a_numbered_line_in_the_summary_is_not_a_key_point():
    raw = "## Summary\n1. this is the summary, oddly numbered\n\n## Key Points\n1. the only key point\n\n## Disclaimer\nd\n"
    output = parse_generation_output(raw, {})
    assert [kp.text for kp in output.key_points] == ["the only key point"]


def test_raw_text_preserved_verbatim():
    assert parse_generation_output(RAW_TEXT, PASSAGE_MAP).raw_text == RAW_TEXT


def test_an_empty_section_is_empty_and_does_not_swallow_the_next_heading():
    """Regression: an empty Summary used to capture the text "## Key Points"
    as the summary, so a missing summary looked like a present one."""
    raw = "## Summary\n\n## Key Points\n1. a point\n\n## Disclaimer\nd\n"
    output = parse_generation_output(raw, {})
    assert output.summary == ""
    assert [kp.text for kp in output.key_points] == ["a point"]

    all_empty = parse_generation_output("## Summary\n\n## Key Points\n\n## Disclaimer\nd\n", {})
    assert (all_empty.summary, all_empty.key_points, all_empty.disclaimer) == ("", [], "d")
