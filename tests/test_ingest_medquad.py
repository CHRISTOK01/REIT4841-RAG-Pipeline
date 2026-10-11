from pathlib import Path

from data.ingest_medquad import (
    load_medquad,
    load_source_documents_jsonl,
    parse_document_xml,
    save_source_documents_jsonl,
)
from data.schemas import SourceDocument

USABLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Document id="0000001_1" source="CancerGov" url="https://www.cancer.gov/example">
<Focus>Adult Acute Lymphoblastic Leukemia</Focus>
<FocusAnnotations>
\t<UMLS>
\t\t<CUIs>
\t\t\t<CUI>C0751606</CUI>
\t\t</CUIs>
\t</UMLS>
</FocusAnnotations>
<QAPairs>
\t<QAPair pid="1">
\t\t\t<Question qid="0000001_1-1" qtype="information">What is (are) Adult Acute Lymphoblastic Leukemia ?</Question>
\t\t\t<Answer>Adult ALL is a type of cancer in which the bone marrow makes too many lymphocytes.</Answer>
\t</QAPair>
\t<QAPair pid="2">
\t\t\t<Question qid="0000001_1-2" qtype="symptoms">What are the symptoms of Adult Acute Lymphoblastic Leukemia ?</Question>
\t\t\t<Answer>Signs and symptoms include fever, feeling tired, and easy bruising or bleeding.</Answer>
\t</QAPair>
</QAPairs>
</Document>
"""

# Mimics the excluded subsets (e.g. MPlus ADAM), where <Answer/> is present but empty.
EXCLUDED_STYLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Document id="0000002_1" source="ADAM" url="https://medlineplus.gov/example">
<Focus>Some Drug</Focus>
<FocusAnnotations>
\t<UMLS>
\t\t<CUIs>
\t\t\t<CUI>C0000000</CUI>
\t\t</CUIs>
\t</UMLS>
</FocusAnnotations>
<QAPairs>
\t<QAPair pid="1">
\t\t\t<Question qid="0000002_1-1" qtype="information">What is Some Drug ?</Question>
\t\t\t<Answer></Answer>
\t</QAPair>
</QAPairs>
</Document>
"""


def test_parse_document_xml_extracts_all_qa_pairs(tmp_path: Path):
    xml_path = tmp_path / "0000001_1.xml"
    xml_path.write_text(USABLE_XML, encoding="utf-8")

    docs = parse_document_xml(xml_path, subset="CancerGov")

    assert len(docs) == 2
    first = docs[0]
    assert first.doc_id == "CancerGov:0000001_1:0000001_1-1"
    assert first.subset == "CancerGov"
    assert first.focus == "Adult Acute Lymphoblastic Leukemia"
    assert first.question_type == "information"
    assert first.question == "What is (are) Adult Acute Lymphoblastic Leukemia ?"
    assert "bone marrow" in first.answer
    assert first.url == "https://www.cancer.gov/example"
    assert first.umls_cui == "C0751606"

    assert docs[1].question_type == "symptoms"


def test_parse_document_xml_skips_empty_answers(tmp_path: Path):
    xml_path = tmp_path / "0000002_1.xml"
    xml_path.write_text(EXCLUDED_STYLE_XML, encoding="utf-8")

    docs = parse_document_xml(xml_path, subset="MPlusADAM")

    assert docs == []


def test_load_medquad_walks_subset_directories(tmp_path: Path):
    repo_dir = tmp_path / "MedQuAD"
    subset_dir = repo_dir / "1_CancerGov_QA"
    subset_dir.mkdir(parents=True)
    (subset_dir / "0000001_1.xml").write_text(USABLE_XML, encoding="utf-8")

    docs = load_medquad(repo_dir, subsets={"CancerGov": "1_CancerGov_QA"})

    assert len(docs) == 2
    assert all(doc.subset == "CancerGov" for doc in docs)


def test_load_medquad_raises_on_missing_subset_folder(tmp_path: Path):
    repo_dir = tmp_path / "MedQuAD"
    repo_dir.mkdir()
    try:
        load_medquad(repo_dir, subsets={"CancerGov": "1_CancerGov_QA"})
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_jsonl_roundtrip(tmp_path: Path):
    docs = [
        SourceDocument(
            doc_id="CancerGov:doc1:q1",
            subset="CancerGov",
            document_id="doc1",
            focus="Some Focus",
            question_id="q1",
            question_type="information",
            question="What is X?",
            answer="X is a thing.",
            url="https://example.com",
            umls_cui="C123",
        )
    ]
    path = tmp_path / "docs.jsonl"
    save_source_documents_jsonl(docs, path)
    loaded = load_source_documents_jsonl(path)

    assert loaded == docs


def test_files_sharing_a_document_id_and_qid_get_distinct_doc_ids(tmp_path: Path):
    """Regression: MedQuAD ships CancerGov files such as 0000013_2.xml and
    0000013_2_1.xml that reuse one <Document id> and the same qids for
    different questions. Their doc_ids used to collide (34 rows in the real
    corpus), which would have merged different documents' relevance labels."""
    subset_dir = tmp_path / "MedQuAD" / "1_CancerGov_QA"
    subset_dir.mkdir(parents=True)
    (subset_dir / "0000001_1.xml").write_text(USABLE_XML, encoding="utf-8")
    second = USABLE_XML.replace("Adult Acute Lymphoblastic Leukemia", "Childhood Acute Lymphoblastic Leukemia")
    (subset_dir / "0000001_1_1.xml").write_text(second, encoding="utf-8")  # same <Document id="0000001_1">, same qids

    docs = load_medquad(tmp_path / "MedQuAD", subsets={"CancerGov": "1_CancerGov_QA"})

    assert len(docs) == 4
    assert [d.doc_id for d in docs] == [
        "CancerGov:0000001_1:0000001_1-1",  # file stem == document id: id format unchanged
        "CancerGov:0000001_1:0000001_1-2",
        "CancerGov:0000001_1_1:0000001_1-1",  # disambiguated by file stem
        "CancerGov:0000001_1_1:0000001_1-2",
    ]
    assert docs[0].focus != docs[2].focus


def test_duplicate_doc_ids_are_a_hard_error():
    import pytest

    from data.ingest_medquad import assert_unique_doc_ids

    doc = SourceDocument(
        doc_id="S:1:1", subset="S", document_id="1", focus="F", question_id="1", question_type="information",
        question="q", answer="a", url=None, umls_cui=None,
    )  # fmt: skip
    assert_unique_doc_ids([doc])
    with pytest.raises(ValueError, match="duplicate doc_id"):
        assert_unique_doc_ids([doc, doc])


def test_internal_whitespace_in_answers_is_collapsed(tmp_path: Path):
    xml = USABLE_XML.replace(
        "Adult ALL is a type of cancer in which the bone marrow makes too many lymphocytes.",
        "Adult ALL   is a type\n\t\tof cancer.",
    )
    xml_path = tmp_path / "0000001_1.xml"
    xml_path.write_text(xml, encoding="utf-8")
    assert parse_document_xml(xml_path, subset="CancerGov")[0].answer == "Adult ALL is a type of cancer."
