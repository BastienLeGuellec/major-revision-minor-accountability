# Data licenses

## Corpus articles

The article texts are not distributed in this repository. `papers/corpus.csv` identifies each article by DOI, with its title, journal, copyright statement and license as recorded in the publisher's XML. Every article was published under a Creative Commons Attribution (CC BY) license: CC BY 4.0 for most, CC BY 3.0 or 3.0 IGO for three Lancet Global Health articles, and an unversioned "CC BY" statement in the publisher's XML for the remaining JAMA Network, PLOS, eLife and Circulation articles. The original authors keep copyright.

`experiments/common/rebuild_corpus.py` re-creates the plain-text versions used in the study from the publishers' open-access APIs. Those texts are derived from the articles: the JATS XML was converted to plain text containing the title, abstract, body text, figure and table captions, and table content; rendered images and supplementary files were not included. They are not the version of record; cite the original articles through their DOIs.

The flaw-injected texts that the script also builds are adaptations in which one passage was replaced by a language model so that the text contains a deliberate, serious methodological or statistical flaw. **They do not describe the studies as they were conducted or published, and the original authors did not write the altered passages.** The replaced passage and its replacement are recorded in the `flaw_location` field of `results/exp2_injections.jsonl`.

## Model outputs (`results/`)

Reviews, decisions, flaw descriptions and quality judgments generated for this study are released under CC BY 4.0 by the study authors. The `flaw_location` field quotes short passages of the original articles, under their CC BY licenses.
