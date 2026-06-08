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

## Demo-day checklist

- Run locally as the primary path (no network or cold-start risk).
- Record a 2-minute screen capture of the three domains as a backup.
- Pre-warm the Render link if you intend to show it.
