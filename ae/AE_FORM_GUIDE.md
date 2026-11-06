# PPoPP'27 Artifact Evaluation — submission form, field by field

Every field of the HotCRP "New submission" form, in form order, with text that
can be pasted directly. Numbered notes flag anything you must confirm yourself.

Deadline: **2026-11-09 23:59 AoE**. The submission must be marked *ready for
review*; saving a draft is not enough.

---

## 1. Title

> PAS: An Efficient Block-Wise Scheduling Framework for Diffusion LLM Serving

Must match the ACM Digital Library title exactly. Compare with the camera-ready
before submitting; if the shepherded title changed, use the new one here **and**
in the README.

---

## 2. Accepted paper (PDF, max 600 MB)

Upload the current paper PDF (revised/intermediate version is allowed). In the
README, list any change made since the reviewed version. If the paper PDF and
the artifact disagree on a number, the README is the place to say which one the
artifact reproduces.

---

## 3. Artifact abstract

> This artifact reproduces the end-to-end serving comparison of PAS
> (`CW_SRPT_V3`) against the SGLang `LowConfidence` baseline on the same
> LLaDA2.0-mini 16B BF16 checkpoint. A single entry point provisions the
> environment from a locked dependency set, downloads the checkpoint from the
> public model repository (Hugging Face, with a ModelScope mirror fallback),
> verifies every weight file against recorded SHA-256 values, and then runs four
> stages: a controlled-load sweep, a four-dataset load sweep, a replay of a
> bundled 30-minute Azure conversation trace, and a GSM8K quality check. All
> public inputs are bundled with source, licence and checksum (GSM8K, MGSM,
> HumanEval, MT-Bench prompts, Azure trace window); no private data is needed.
> The artifact reproduces the *direction and magnitude* of the paper's
> latency/throughput trend on 2×A100, not the paper's H100 absolute maxima. The
> Kimi traces used in the paper are confidential and cannot be released; the
> Azure trace is the public trace example. Hardware requirement: 2 NVIDIA GPUs
> with ≥40 GB each (validated on A100 SXM4 40 GB, TP=2), ~128 GB host RAM and
> ~120 GB free disk.

---

## 4. Authors (email, name, affiliation — 5 slots)

Slot 1 is prefilled:

| # | Email | Name | Affiliation |
|---|---|---|---|
| 1 | idc_zjn@hust.edu.cn | Jianian Zhu | Huazhong University of Science and Technology |
| 2 | *(fill)* | *(fill)* | *(fill)* |
| 3–5 | *(fill or leave empty)* | | |

2. Enter the remaining authors exactly as they will appear in the ACM DL, in
   paper order. The paper PDF submitted to AE is anonymised, so copy the order
   from the camera-ready or the HotCRP author list — do not retype from memory.

---

## 5. ACM corresponding author

Select one author (normally the first author or the senior author). This person
is charged institutional open-access fees if the paper is accepted, so it must
be the person who agreed to that.

---

## 6. Contacts

Everyone in the author list who has a site account is added automatically. Add
anyone else who needs edit access (e.g. a student running the evaluations).

---

## 7. README and evaluation instructions (max 600 MB)

Upload `ae/README.md` converted to PDF or plain text. The README in this
repository is written for exactly this field: getting-started test (~30 min),
full reproduction steps, expected outputs and supported/unsupported claims.
Add one line at the top with the commit hash you are submitting.

---

## 8. Artifact download URL

> https://github.com/Ringssss/dLLM-serve/tree/5211b67f3cdeb32b4417a1c88b4118a8cba0e4ee

3. Use a **commit hash**, never the branch name. `ppopp-artifacts` moves; a
   commit does not. Fill in the hash printed by `git rev-parse HEAD` after the
   push, and paste the same hash into the README.
4. Do not use URL shorteners or any link that records who clicked — evaluation
   is single-blind.
5. The repository is public, so no credentials are needed. If it is ever made
   private, send access privately to the chairs, not in the form.

---

## 9. PC conflicts

Select every PC member who has a conflict: past advisors and students, anyone
at Huazhong University of Science and Technology, and any coauthor or
collaborator within roughly the last two years. **Do this from the author list,
not from the artifact** — the artifact cannot tell you who your coauthors are.
If in doubt about a name, ask the senior author rather than guessing.

---

## 10. Artifact archive (max 600 MB)

Optional if the URL in field 8 works. To also upload a self-contained snapshot:

```bash
python ae/package.py     # writes ../PAS-PPoPP27-AE-public.tar.gz + .sha256
```

The archive is ~24 MB (source + bundled public inputs; no model weights).
The 32.5 GB checkpoint is downloaded at run time from the public model
repository and verified against `ae/model-files.json`, so it does not belong in
the archive.

---

## 11. Requested badges

Recommended:

- **Artifacts Evaluated — Functional** — the artifact builds, runs, and
  reproduces the trend on the evaluators' hardware with no author intervention.
- **Results Reproduced** — the documented end-to-end experiments are re-run and
  produce results consistent with the paper for the stated scope.

Do **not** request **Artifacts Available** unless a permanent archival record
exists (see field 14). Do not request **Reusable** as well as Functional; they
are alternatives and only one red badge is awarded.

---

## 12. Hardware, software and resource requirements

> Two NVIDIA GPUs with at least 40 GB each are required; the artifact was
> validated on 2×A100 SXM4 40 GB with tensor parallelism 2, running the two
> methods sequentially on the same pair. The paper's own experiments used
> 8×H100 80 GB; absolute H100 numbers and the 8B/100B model panels are not
> reproduced here. Host: Linux x86-64, Python 3.12, ~128 GB RAM, ~120 GB free
> disk (32.5 GB checkpoint plus environment, caches and results). Software:
> NVIDIA driver supporting CUDA 13, installed automatically from
> `ae/requirements.lock` into a local virtual environment; no system-wide
> Python packages, no root, no private credentials, no paid service. Network
> access to PyPI and to the Hugging Face model repository (ModelScope mirror as
> fallback) is needed once for installation; all datasets and the trace are
> bundled in the repository, so evaluation runs are offline-capable. Ordinary
> hardware is not sufficient — the checkpoint does not fit on a single 40 GB
> GPU with tensor parallelism 1 at this batch size. No remote access to author
> machines is needed.

---

## 13. Expected runtime and claims to reproduce

> Getting-started test (`bash ae/reproduce.sh smoke`): about 30 minutes on
> 2×A100 after installation and checkpoint download (5–10 minutes for install,
> 10–20 minutes for the 32.5 GB download, then ~5 minutes of measurement).
>
> Full reproduction (`bash ae/reproduce.sh all`): about 60–90 minutes on the
> validated pair, plus installation and download once. It runs four stages:
> a controlled-load sweep (5 load points × 2 methods), a four-dataset load sweep
> (4 datasets × 7 load points × 2 methods), an Azure trace replay, and a
> 200-example GSM8K quality check.
>
> Claims reproduced: (i) P90 end-to-end latency rises with offered load and PAS
> shifts the saturation knee to a higher request rate; (ii) PAS sustains higher
> completed-request throughput under load; (iii) the Azure conversation trace
> shows a latency reduction at matched offered load; (iv) GSM8K accuracy is
> preserved at the default operating point. Expected direction: PAS P90 below
> SGLang at every load point, with the largest gap near saturation; quality
> within a few examples out of 200.
>
> Claims **not** reproduced: exact H100 maxima (8.3× P90, 166% throughput), the
> LLaDA-8B and LLaDA2.0-flash-100B panels, the four-task quality matrix
> (HumanEval pass@1, MGSM exact match, MT-Bench judged score — the bundled
> HumanEval/MGSM/MT-Bench files are use as load-sweep inputs only, and the
> MT-Bench score is judged by a model that is not shipped), the component
> ablation in Fig. 13, the hindsight-signal analysis in Fig. 10, and all Kimi
> experiments, whose trace is confidential and cannot be released. The public
> trace experiment therefore uses Azure.
>
> Variation: results depend on GPU model and on whether CUDA graphs are enabled
> (the entry point enables them by default). Single-seed runs are expected to
> vary by a few percent; the three-seed full sweep is provided for stronger
> claims. Any run with a failed request is invalid by construction — the
> harness exits non-zero.

---

## 14. Public archival DOI or URL

Only needed for **Artifacts Available**. A GitHub branch or commit is *not* a
permanent archival record. If you want this badge, deposit the archive from
field 10 on Zenodo (or an institutional repository), get a DOI, and paste it
here. Leave the field empty if you are not requesting Available.

---

## Pre-submission checklist

- [ ] Commit pushed to `ppopp-artifacts`; `git rev-parse HEAD` recorded
- [ ] Field 8 URL contains that commit, not a branch name
- [ ] README header states the same commit
- [ ] `bash ae/reproduce.sh smoke` passes on a clean machine
- [ ] `python ae/package.py` produced an archive under 600 MB
- [ ] Author list and corresponding author match the ACM DL record
- [ ] PC conflicts entered from the author list
- [ ] Form marked *ready for review* (drafts are not evaluated)
