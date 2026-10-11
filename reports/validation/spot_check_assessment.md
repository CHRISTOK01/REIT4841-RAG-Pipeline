# Do the automated scores match what a reader sees?

A case-by-case reading of the 15 rows in `spot_check_pack.md`.

**Who judged:** the AI assistant that audited the pipeline read each case (question, reference
answer, retrieved chunks, generated answer, per-claim labels, scores). This is one careful
reader's opinion, not a clinician's, and 15 cases is a small sample: 6 of them were selected
*because* the scores disagreed, so the error rates below overstate how often the scores are wrong
across the whole sweep. Use the blank boxes in the pack to record your own reading.

## Verdict per metric

| Metric | Agrees with the reader in | What goes wrong when it does not |
|---|---|---|
| Retrieval metrics (given the labels) | 15 / 15 | Nothing in the arithmetic. The *labels* are the weak point: in cases 4 and 6 the retriever returned passages on the right condition that are labelled not relevant because they come from a different QA pair; in case 7 Recall@5 is 0.25 for a retrieval that found the one informative gold chunk, because the other three "gold" chunks are boilerplate |
| Abstention label | 15 / 15 | — |
| Faithfulness | 10 / 15 | **Too low** in cases 1, 2 and 14: the model correctly said the context did not answer, and the judge labelled sentences such as "the resources listed do not specify treatments" as *unsupported*. **Too high** in cases 4 and 5: a symptom of a different eye disease is presented as a symptom of low vision, and a gene for a different neuropathy is offered as "related"; each sentence can be found in the context, so the judge calls it supported, but the answer misleads |
| Reference agreement | 12 / 15 in direction | Understates in cases 6 and 13, where the answer is reasonable but says more than a short or differently-focused reference. In cases 1 and 2 it was 0 for a correct refusal (same cause as faithfulness) |
| Context relevance | 14 / 15 | Generous in case 3: four of five passages that merely mention coughing are marked relevant to "What is cough?" |
| Answer relevance | not informative | It sits between 0.62 and 0.88 in every case, including case 9 (asked for the outlook, answered with prevalence: 0.78) and case 14 (0.86 for an answer that says the context has no treatment information) |

## What was changed because of this reading

- Faithfulness and reference agreement are now reported only for answers that were not labelled
  `abstained` (`analysis/aggregate.py`), with regression tests. This removes the artefact in cases
  1 and 2. The raw per-claim labels remain in the log. Effect on the sweep: 30 of 1,350 rows from
  the nine real configurations had a faithfulness value on an abstained answer, with a mean of
  0.25; excluding them raises configuration means by 0.00 to 0.04.
- Case 14 (label `partial`) is not fixed by that rule and remains an example of the judge being
  too harsh on hedged answers. Partial answers have a mean faithfulness of 0.74 against 0.98 for
  fully answered ones; some of that gap is real and some is this artefact, and the data cannot
  separate the two.

## Case notes

| Case | Config | Reader's view of the answer | Scores vs reader |
|---|---|---|---|
| 1 | reduced / lexical | Correct refusal. The MedQuAD "treatment" answer for this query is a list of resource links, so there is nothing to retrieve | Faithfulness 0.00 and reference agreement 0.00 are wrong (now excluded: abstained) |
| 2 | mixed / dense | Same query, same correct refusal | Same; the judge labelled near-identical sentences differently from case 1, which shows its inconsistency on such sentences |
| 3 | reduced / lexical | Answer topic removed from the KB. The model assembled an answer about coughs from cold, whooping-cough and COPD passages. Every sentence is in the context; it does not say what a cough is | Faithfulness 1.00 is right. Reference agreement 0.50 and answer relevance 0.50 are fair. Context relevance 0.80 is generous |
| 4 | curated / lexical | BM25 missed the gold chunks but found two other passages about low vision and one about a different retinal disease. The answer mixes that disease's symptoms into "symptoms of low vision" | Faithfulness 1.00 overstates. Reference agreement 0.25 catches the problem |
| 5 | reduced / lexical | Answer not in the KB. The model says so, then offers genes for a different neuropathy as "related" | Faithfulness 1.00 overstates ("may share some characteristics" is not in the context). Abstention `partial` is right. Reference agreement 0.00 is right |
| 6 | reduced / lexical | Reasonable answer about research treatments, from the same source document as the gold chunk | Faithfulness 1.00 right. Reference agreement 0.25 understates: the reference is one sentence about treatment goals |
| 7 | curated / lexical | Good, specific answer | All agree. Recall 0.25 reflects boilerplate gold chunks, not a retrieval failure |
| 8 | curated / dense | Good | All agree |
| 9 | curated / hybrid | Retrieval missed. The answer gives prevalence instead of outlook, and "generally poor" is not in the context | Faithfulness 0.50 and reference agreement 0.25 are right. Answer relevance 0.78 is too kind |
| 10 | mixed / lexical | Retrieval missed. Half the answer is from the context, half is the model's general knowledge | Faithfulness 0.50 is exactly right: the judge marked the outside-knowledge sentences unsupported |
| 11 | mixed / dense | Good | All agree |
| 12 | mixed / hybrid | Good | All agree |
| 13 | reduced / lexical | Good; includes correct detail not in the reference | Reference agreement 0.67 understates slightly |
| 14 | reduced / dense | Answer not in the KB. The model says so and notes that the passages concern other syndromes | Faithfulness 0.00 is too harsh; `partial` is defensible |
| 15 | reduced / hybrid | Good | All agree |
