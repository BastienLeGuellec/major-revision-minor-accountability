# Data licenses

## Corpus texts (`papers/*.txt`)

Each file is derived from an article published under a Creative Commons Attribution (CC BY) license: CC BY 4.0 for most articles, CC BY 3.0 or 3.0 IGO for three Lancet Global Health articles, and an unversioned "CC BY" statement in the publisher's XML for the remaining JAMA Network, PLOS, eLife and Circulation articles. The original authors keep copyright. `papers/corpus.csv` gives each article's DOI, title, journal, copyright statement and license as recorded in the publisher's XML.

Changes made: the JATS XML full text was converted to plain text containing the title, abstract, body text, figure and table captions, and table content. Rendered images and supplementary files were not included. These files are not the version of record; cite the original articles through their DOIs.

## Flaw-injected texts (`papers/_exp2_injected/*.txt`)

These are adaptations of the same CC BY-licensed articles. In each file one passage was replaced by a language model so that the text contains a deliberate, serious methodological or statistical flaw, for the purpose of testing whether reviewers detect it. The replaced passage and its replacement are recorded in the `flaw_location` field of `results/exp2_injections.jsonl`.

**These texts do not describe the studies as they were conducted or published, and the original authors did not write the altered passages.** They are distributed only so that Experiment 2 can be reproduced and audited.

## Model outputs (`results/`)

Reviews, decisions, flaw descriptions and quality judgments generated for this study are released under CC BY 4.0 by the study authors.
