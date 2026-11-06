# Archiving this artifact on Zenodo (Artifacts Available badge)

The PPoPP'27 AE form asks for a permanent archival record for the *Artifacts
Available* badge; a development repository or a branch link does not qualify.
This page records how to deposit this artifact and what was prepared for it.

The deposit itself must be performed by an author, from an author's Zenodo
account: publishing a DOI is an irreversible public act in the authors' name.
Nothing below has been executed on behalf of the authors.

## What to upload

```bash
python ae/package.py
```

writes `../PAS-PPoPP27-AE-public.tar.gz` (about 24 MB) plus a `.sha256` file.
The archive contains the tracked source, the AE harness and the bundled public
inputs; it contains **no model weights, no virtual environment and no Kimi
trace data**. Do not upload any archive whose name contains `PRIVATE`.

Metadata for the deposit is in `ae/ZENODO-metadata.json`; Zenodo's "Import
metadata" step accepts it directly. Fill in the full author list, in paper
order, before publishing.

## Steps

1. Sign in at https://zenodo.org and open https://zenodo.org/uploads/new .
2. Upload `PAS-PPoPP27-AE-public.tar.gz`.
3. Import `ae/ZENODO-metadata.json` (or copy its fields by hand): title, upload
   type `software`, access `open`, licence `Apache-2.0`, version `ppopp27-ae`.
4. Add every author with their affiliation and ORCID if available, in the same
   order as the paper.
5. Set the publication date and publish. Zenodo mints a DOI such as
   `10.5281/zenodo.1234567`.
6. Paste `Zenodo. https://doi.org/10.5281/zenodo.1234567` into the form's
   *Public archival DOI or URL* field and select **Artifacts Available**.

## Before publishing

- [ ] The archive was built from the same commit that the submission form names
- [ ] `ae/ZENODO-metadata.json` lists the complete, correctly ordered author list
- [ ] The archive name does **not** contain `PRIVATE`
- [ ] `sha256sum PAS-PPoPP27-AE-public.tar.gz` matches the `.sha256` file
- [ ] You understand a published Zenodo record can only be superseded by a new
      version, not deleted

## Relationship to the GitHub tag

Zenodo and the Git tag should point at the same content. Build the archive after
tagging, from the tagged commit, so the DOI and the URL describe one state.
