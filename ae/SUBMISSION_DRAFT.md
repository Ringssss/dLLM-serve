# PPoPP'27 AE — submission draft

**Field-by-field form text:** see `AE_FORM_GUIDE.md` (same directory). That file
has the exact text for every field of the HotCRP form in form order. This file
only records decisions and open items.

## Decisions

- Artifact URL: `https://github.com/Ringssss/dLLM-serve/tree/<commit>` — a commit,
  never the branch name. Fill the hash after pushing and paste the same hash at
  the top of `ae/README.md`.
- README upload: `ae/README.md` as PDF or plain text.
- Archive (optional): `python ae/package.py`. About 24 MB; the 32.5 GB checkpoint
  is downloaded at run time and verified against `ae/model-files.json`.
- Badges: Artifacts Evaluated — Functional and Results Reproduced. Not Available
  unless a Zenodo-style DOI exists; not Reusable alongside Functional.

## Open items (need the authors)

- Remaining author names, emails and affiliations, in ACM DL order.
- ACM corresponding author (open-access fees are charged there).
- PC conflicts, entered from the author list.
- Whether the submitted paper PDF already contains the operating-point change
  described in `CHANGES.md`; if not, say so in the README header.

## Scope reminders for the form

- Default PAS config is `pas-final.yaml` (floor 0.8 + stability gate), not the
  paper's 0.6. The paper's floor scores 155/200 on GSM8K versus 182/200 for
  SGLang; the default point scores 177–183/200 against 182–183/200.
- The reproduced trace is Azure; Kimi is confidential and excluded.
- MT-Bench, HumanEval and MGSM are load-sweep inputs, not a quality matrix.
- Peak throughput at 64 req/s is queueing-free capacity only up to saturation
  (about 14–15 req/s for PAS, 5–8 req/s for SGLang on 2×A100).
