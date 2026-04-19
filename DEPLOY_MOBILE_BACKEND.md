# Deploying the Mobile Backend

This app should not depend on your laptop staying online.

The deployment shape we want is:

- Flutter app on iOS + Android
- Python inference backend hosted on Render first
- `AVIATIONSTACK_KEY` stored only on the server
- model checkpoint downloaded at startup from a public file URL

## Why We Need a Slim Deploy Repo

The main project repo is great for research and training, but it is not a good
first Render source repo because:

- it contains large local datasets and generated artifacts
- the final checkpoint is about `126 MB`
- GitHub blocks normal Git pushes for files larger than `100 MB`

GitHub's docs confirm that regular repositories block files larger than
`100 MiB`, and large binaries should be distributed with Git LFS or releases:

- https://docs.github.com/articles/distributing-large-binaries?platform=windows

So the clean Render path is:

1. create a slim deployment bundle
2. host the checkpoint file separately
3. point Render at that bundle repo

## What the Hosted Backend Needs

Required code files:

- `12_mobile_api.py`
- `07_dashboard.py`
- `requirements-mobile-api.txt`
- `Dockerfile.mobile-api`
- `render.yaml`

Required graph artifacts:

- `graph_data/static_edges.pt`
- `graph_data/airport_index.parquet`
- `graph_data/route_stats.parquet`
- `graph_data/route_stats_global.parquet`

Required env vars:

- `AVIATIONSTACK_KEY`
- `MODEL_URL`

The backend now supports `MODEL_URL`. If the checkpoint is missing locally, it
downloads it on startup into `checkpoints/`.

## Fastest Render Workflow

### 1. Build the slim deploy bundle

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\prepare_render_bundle.ps1
```

This creates:

- `deploy/render_mobile_api_bundle/`

That bundle intentionally excludes:

- `.env`
- raw datasets
- large snapshot tensors
- the `126 MB` checkpoint

### 2. Host the checkpoint file

Upload `checkpoints/best_model_dep_12k_ordinal_ep25.pt` somewhere public.

Good simple options:

- GitHub Release asset
- Hugging Face model repo
- cloud object storage with a public download URL

Then copy the public direct-download URL.

### 3. Create a small GitHub repo from the bundle

Create a separate GitHub repo just for the Render backend, then upload the
contents of:

- `deploy/render_mobile_api_bundle/`

This keeps the deployment repo small and avoids pushing research artifacts.

### 4. Create the Render Web Service

Render docs for GitHub-backed deploys:

- https://render.com/docs/github
- https://render.com/docs/deploys/

In Render:

1. Create a new **Web Service**
2. Connect the slim backend repo
3. Let Render detect `render.yaml`
4. Add these env vars:

```text
AVIATIONSTACK_KEY=your_key_here
MODEL_URL=https://your-public-checkpoint-url
```

### 5. Deploy and test

After deploy, test:

- `/`
- `/health`

## Render Free Caveat

Render free services spin down after inactivity and can take a bit to wake up
again. That is fine for:

- first deployment
- personal testing
- validating the Flutter app end to end

It is not ideal for a polished shared app. Once the app is working, the next
upgrade is usually:

- small VPS
- or an always-on paid container service

## Local Run Without Docker

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-mobile-api.txt
set MODEL_URL=https://your-public-checkpoint-url
uvicorn 12_mobile_api:app --host 0.0.0.0 --port 8000
```

On PowerShell:

```powershell
$env:MODEL_URL="https://your-public-checkpoint-url"
uvicorn 12_mobile_api:app --host 0.0.0.0 --port 8000
```

## Security Notes

- Never ship the AviationStack key in the mobile app
- Never commit `.env`
- Start without public auth if this is just for you and a friend
- Add auth or rate limiting before making the API broadly public

## Mobile Client Base URL

Once deployed, point the Flutter app at:

```text
https://your-render-service.onrender.com
```
