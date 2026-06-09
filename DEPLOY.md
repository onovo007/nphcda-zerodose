# Run and deploy

## Run locally (primary demo path)

```
cd claude_model
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Click **Use bundled sample data** on the Home page, then open a
domain from the sidebar. The first Domain 5 model run takes about 60-90 seconds (it samples the
Bayesian model live); later runs are cached and instant.

No system C++ compiler is required: the Bayesian model samples with nutpie (numba/LLVM).

## Deploy to Render (shareable link)

The app is a standard Streamlit service. `render.yaml` is included.

1. Put the contents of this `claude_model/` folder in a GitHub repository (repo root = this
   folder, so `app.py` and `requirements.txt` are at the top level).
2. In the Render dashboard: New > Blueprint, point it at the repo. Render reads `render.yaml`
   and creates a Python web service:
   - Build: `pip install -r requirements.txt`
   - Start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - Plan: starter (paid) so there is enough RAM for live PyMC/Prophet and no spin-down.
3. First build takes several minutes (pymc, prophet, geopandas). After it goes live, open the
   URL once to warm it before the demo.

Notes:
- Versions in `requirements.txt` are pinned to the exact set verified locally. Do not loosen
  `numpy<2` or `arviz<1.0`: arviz 1.x removes the `concat` symbol pymc imports, and a numpy-1.x
  ABI module breaks under numpy 2.x.
- Boundary geometry ships as GeoJSON in `data/sample/geo/`, so no shapefiles are needed at runtime.

## Deploy to Hugging Face (Docker Space) - recommended for memory headroom

No GPU is needed (PyMC/Prophet/spatial are CPU work). The reason to prefer HF is RAM: the free
**CPU basic** Space provides about **16 GB RAM / 2 vCPU**, far more than Render's instances, which
removes the out-of-memory restarts. The repo ships a `Dockerfile` and the README has the Docker
Space front-matter (`sdk: docker`, `app_port: 7860`).

1. Create a token at https://huggingface.co/settings/tokens (Write access).
2. Create a Space: https://huggingface.co/new-space -> SDK = **Docker** (blank), Hardware = CPU
   basic (free). Name it e.g. `nphcda-zerodose`.
3. Push this repo to the Space (run from the app folder):
   ```
   git remote add hf https://huggingface.co/spaces/<your-hf-username>/nphcda-zerodose
   git push hf main
   ```
   When prompted for a password, paste the **HF write token** (username is your HF handle).
4. HF builds the Dockerfile automatically (about 5-10 minutes) and serves the app on port 7860.
   The Space URL becomes the shareable link. Add your OpenAI key in the sidebar as usual.

Notes:
- The same `Dockerfile` also runs on Render (set the service to the Docker runtime) or any Docker
  host, so you are not locked in.
- If a free CPU Space ever feels slow, upgrade to a paid CPU tier in the Space settings. Still no GPU.

## If you prefer to stay on Render - quick diagnosis

The 502 is an out-of-memory restart, not a code bug (it never happens locally). Check:
1. **Deployed commit**: the service must be on the latest commit (the one that gates the Gi* maps
   behind a button). An older commit auto-runs the spatial step after the Bayesian fit and OOMs.
2. **Instance type**: Render -> service -> Settings -> Instance Type must actually be **Pro (4 GB)**
   (a blueprint plan change sometimes needs manual confirmation).
3. **Logs at crash time**: look for `Out of memory` / `SIGKILL` / `Ran out of memory` to confirm OOM.

## Demo-day checklist

- Run locally as the primary path (no network or cold-start risk).
- Record a 2-minute screen capture of the three domains as a backup.
- Pre-warm the Render link if you intend to show it.
