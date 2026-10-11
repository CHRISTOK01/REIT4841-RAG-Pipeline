"""Download and parse the MedQuAD dataset.

MedQuAD (Ben Abacha & Demner-Fushman, 2019, "A Question-Entailment Approach to
Question Answering", BMC Bioinformatics) is CC BY 4.0 licensed and distributed at
https://github.com/abachaa/MedQuAD as one XML file per <Document> (a topic/"focus"),
each containing one or more <QAPair> question+answer entries.

Only the 9 of 12 NIH source subsets that retain answer text are used here. The
other 3 (MPlus ADAM, MPlus Drugs, MPlus Herbs/Supplements) have their <Answer>
elements stripped for copyright reasons -- QAPairs there parse with an empty
answer and are skipped by `parse_document_xml`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from data.schemas import SourceDocument

MEDQUAD_REPO_URL = "https://github.com/abachaa/MedQuAD.git"

# subset_name -> folder name in the MedQuAD repo. These 9 retain full answer text.
USABLE_SUBSETS = {
    "CancerGov": "1_CancerGov_QA",
    "GARD": "2_GARD_QA",
    "GHR": "3_GHR_QA",
    "MPlusHealthTopics": "4_MPlus_Health_Topics_QA",
    "NIDDK": "5_NIDDK_QA",
    "NINDS": "6_NINDS_QA",
    "SeniorHealth": "7_SeniorHealth_QA",
    "NHLBI": "8_NHLBI_QA_XML",
    "CDC": "9_CDC_QA",
}

# Excluded: answer text stripped from the source repo for copyright reasons.
EXCLUDED_SUBSETS = {
    "MPlusADAM": "10_MPlus_ADAM_QA",
    "MPlusDrugs": "11_MPlusDrugs_QA",
    "MPlusHerbsSupplements": "12_MPlusHerbsSupplements_QA",
}


def download_medquad(dest_dir: Path, *, force: bool = False) -> Path:
    """Clone the MedQuAD repo into dest_dir/MedQuAD if not already present.

    Returns the path to the cloned repo root.
    """
    repo_dir = dest_dir / "MedQuAD"
    if repo_dir.exists() and force:
        shutil.rmtree(repo_dir)
    if repo_dir.exists():
        return repo_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", MEDQUAD_REPO_URL, str(repo_dir)],
        check=True,
    )
    return repo_dir


def _text_or_none(elem: ET.Element | None) -> str | None:
    if elem is None or elem.text is None:
        return None
    text = " ".join(elem.text.split())  # collapse internal whitespace/newlines
    return text or None


def parse_document_xml(xml_path: Path, subset: str) -> list[SourceDocument]:
    """Parse one MedQuAD <Document> XML file into one SourceDocument per QAPair
    that has non-empty question and answer text."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    document_id = root.get("id", xml_path.stem)
    url = root.get("url")
    focus = _text_or_none(root.find("Focus")) or "Unknown"
    cui_elem = root.find("./FocusAnnotations/UMLS/CUIs/CUI")
    umls_cui = _text_or_none(cui_elem)

    docs: list[SourceDocument] = []
    qa_pairs_elem = root.find("QAPairs")
    if qa_pairs_elem is None:
        return docs

    for qa_pair in qa_pairs_elem.findall("QAPair"):
        question_elem = qa_pair.find("Question")
        answer_elem = qa_pair.find("Answer")
        question = _text_or_none(question_elem)
        answer = _text_or_none(answer_elem)
        if not question or not answer:
            continue  # stripped answer (excluded subset) or malformed pair

        question_id = (
            question_elem.get("qid") if question_elem is not None else None
        ) or f"{document_id}-{qa_pair.get('pid', '?')}"
        question_type = (
            question_elem.get("qtype", "unknown") if question_elem is not None else "unknown"
        )

        # Keyed on the file stem, not <Document id>: MedQuAD ships several CancerGov
        # files (e.g. 0000013_2.xml, 0000013_2_1.xml, ...) that reuse one Document id
        # AND the same qids for different questions, so id+qid is not unique.
        if xml_path.stem == document_id:
            doc_id = f"{subset}:{document_id}:{question_id}"
        else:
            doc_id = f"{subset}:{xml_path.stem}:{question_id}"
        docs.append(
            SourceDocument(
                doc_id=doc_id,
                subset=subset,
                document_id=document_id,
                focus=focus,
                question_id=question_id,
                question_type=question_type,
                question=question,
                answer=answer,
                url=url,
                umls_cui=umls_cui,
            )
        )
    return docs


def load_medquad(
    repo_dir: Path, subsets: dict[str, str] | None = None
) -> list[SourceDocument]:
    """Parse all usable subsets under repo_dir into a flat list of SourceDocuments."""
    subsets = subsets if subsets is not None else USABLE_SUBSETS
    all_docs: list[SourceDocument] = []
    for subset_name, folder_name in subsets.items():
        subset_dir = repo_dir / folder_name
        if not subset_dir.is_dir():
            raise FileNotFoundError(f"Expected MedQuAD subset folder not found: {subset_dir}")
        for xml_path in sorted(subset_dir.glob("*.xml")):
            all_docs.extend(parse_document_xml(xml_path, subset_name))
    assert_unique_doc_ids(all_docs)
    return all_docs


def assert_unique_doc_ids(docs: list[SourceDocument]) -> None:
    """Relevance judgments key on doc_id, so two documents sharing one would
    silently merge their relevant-chunk sets. Fail loudly instead."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for doc in docs:
        if doc.doc_id in seen:
            duplicates.add(doc.doc_id)
        seen.add(doc.doc_id)
    if duplicates:
        raise ValueError(
            f"{len(duplicates)} duplicate doc_id(s) in corpus, e.g. {sorted(duplicates)[:3]}"
        )


def save_source_documents_jsonl(docs: list[SourceDocument], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for doc in docs:
            f.write(json.dumps(doc.__dict__, ensure_ascii=False) + "\n")


def load_source_documents_jsonl(path: Path) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(SourceDocument(**json.loads(line)))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and parse MedQuAD into a JSONL corpus.")
    parser.add_argument("--raw-dir", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--out", type=Path, default=Path("datasets/processed/medquad_documents.jsonl"))
    parser.add_argument("--force-redownload", action="store_true")
    args = parser.parse_args()

    repo_dir = download_medquad(args.raw_dir, force=args.force_redownload)
    docs = load_medquad(repo_dir)
    save_source_documents_jsonl(docs, args.out)

    subset_counts: dict[str, int] = {}
    for doc in docs:
        subset_counts[doc.subset] = subset_counts.get(doc.subset, 0) + 1

    print(f"Parsed {len(docs)} question-answer source documents from {len(subset_counts)} subsets.")
    for subset, count in sorted(subset_counts.items()):
        print(f"  {subset}: {count}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
