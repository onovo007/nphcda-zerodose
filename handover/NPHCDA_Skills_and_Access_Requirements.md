# NPHCDA Zero-Dose Platform - Skills and access required to deliver the updates

A reference note on the capabilities, skills, software and access needed on the new computer so Claude Code can deliver premium, top-notch updates across the four deliverables: the report (.docx), the slide deck (.pptx), the notebooks (.ipynb) and the web application. This lists what must be in place; it is not a set of fix instructions.

---

## 1. How Claude Code must be running
Claude Code needs an agent environment with a real terminal and filesystem - either the Claude Code CLI, or the Code / Cowork side of the Claude Desktop app (not the web chat connector). It must be able to:
- **Run terminal commands** (Git, Python, pip) with permission to execute.
- **Read, create and edit files** in the project folder.
- **Run Python** against the installed scientific stack.
- **Use Git** to pull and push to both remotes.
- **Open a browser / preview** to verify the live web app (built-in browser or preview tools).

## 2. Skills to have enabled
- **docx** - to read and edit the Word report (`.docx`) with proper styles, tables, figures and typography.
- **pptx** - to read and edit the PowerPoint deck (`.pptx`), matching its template.
- **xlsx** - to read and edit the Excel workbooks (priority lists, archetype list, data-quality files).
- **pdf** - if any deliverable is exported to or supplied as PDF.
- **Notebook editing** - to edit and validate the Colab notebooks (`.ipynb`) via the notebook-edit tool or `nbformat`.
- (A plain file-write path is sufficient for the Markdown docs.)

## 3. Software and environment on the PC
- **Python 3.12** and **Git** installed; **Node.js** (for the Claude Code CLI and any MCP servers).
- The full project stack installed: `pip install -r requirements.txt`. This includes Streamlit, PyMC, nutpie, PyTensor, numba, Prophet (cmdstanpy), geopandas, libpysal, esda, shapely, pyproj, matplotlib, plotly, scikit-learn, statsmodels, pandas 2.2.3, numpy 1.26.4, pyarrow 16.1.0, python-docx, python-pptx, openpyxl.
- Document editing libraries present (python-docx, python-pptx, openpyxl come with the stack above) so the report, deck and workbooks can be edited programmatically and reproducibly.
- Plotting libraries (matplotlib, plotly) for regenerating figures.

## 4. Accounts and access required
- **GitHub** - the `onovo007` account and a **Personal Access Token** (fine-grained, scoped to `onovo007/nphcda-zerodose`, Contents: read and write). Remote `origin` points here; pushing to `main` auto-deploys Render.
- **Hugging Face** - the `amobionovo` account and a **Write token** scoped to the Space. Remote `hf` points to `huggingface.co/spaces/amobionovo/nphcda-zerodose`; it is pushed separately. The development-era token should be revoked and a fresh token generated on the new PC.
- **Render** - dashboard access to the service, to watch deploy logs and restart if needed (deploys trigger automatically from the GitHub push).
- **Google Drive** - access to the datasets folder (the data of record), either through the Google Drive connector in Claude Desktop or by downloading the domain-organized CSVs locally. Folder: https://drive.google.com/drive/folders/107pWT0m4_A9Sk4aVji-t8LV3m7gfoLzx
- **OpenAI API key** - optional, only needed to exercise the ZARA assistant / AI interpretation while testing.
- **App email allow-list** - the `ALLOWED_EMAILS` Space secret controls who can sign in; editing it needs owner access to the Hugging Face Space settings.

## 5. What each deliverable needs
- **Report (`.docx`):** the docx skill or python-docx; access to the current report in `handover/`; the datasets and regenerated figures so any numbers and figures can be refreshed consistently.
- **Slide deck (`.pptx`):** the pptx skill or python-pptx; the current deck in `handover/`; the figure image files (in `02_PPT_Slide-Deck/` and `05_Outputs/Figures/`).
- **Notebooks (`.ipynb`):** the notebook-edit tool or nbformat; the datasets (local or Drive); Google Colab or a local Jupyter runtime to run cells and regenerate outputs.
- **Web application:** the full Python stack to run and test it (`python -m streamlit run app.py`), `_smoke_test.py` to check Domains 1, 2, 5 and the spatial layer end to end, and `precompute_d5.py` to regenerate the bundled Domain 5 precomputed results after any data or model change; matplotlib/plotly for figures; a browser/preview to verify the live app; and Git access to push to `origin` (GitHub, which drives Render) and `hf` (Hugging Face).

## 6. Verification tooling available in the repo
- `_smoke_test.py` - end-to-end run of the core models and figure builders on the bundled data.
- `precompute_d5.py` - regenerates `data/sample/precomputed/` (the app serves these for the bundled data).
- The archetype / equity pipeline scripts under `Archtyping at LGA level/`.
- The README (`README.md`) and `handover/NPHCDA_System_Methods_Context.md` describe the architecture, methods, datasets and deployment.

## 7. Security
- No credentials are stored in the repository. The GitHub and Hugging Face tokens are supplied by the user at push time and cached locally; the OpenAI key is entered per session; sign-in access is granted by adding authorized emails to the allow-list. Any token exposed during development should be revoked and replaced.

Consortium: CIDRE and Quantium Insights LLC, in technical support of NPHCDA; funders and reviewers GAVI and UNICEF.
