# mathemAIcs

Comprehensive end-to-end pipeline for generating synthetic math datasets, training a multimodal Transformer, and serving an interactive FastAPI + web UI for reasoning tasks.

## Features
- **Synthetic dataset generation**: geometry diagrams, charts, OCR text panels, line-reasoning exercises, and a rich text-only reasoning set.
- **Multimodal model**: vision encoder + text decoder trained from scratch, supports image-only, text-only, and multimodal prompts.
- **FastAPI server**: `/predict` endpoint plus dataset build/training triggers, model import/export, and static web client.
- **Web UI**: password-gated interface to upload images/text, trigger dataset builds (geometry, charts, OCR, line reasoning, text counts), launch training with custom epochs, and manage model files.

## Quick Start
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate datasets
```bash
cd src
python build_dataset.py \
  --geometry 200 \
  --charts 200 \
  --ocr 100 \
  --lines 80 \
  --text 600
```
- `--ocr` controls how many OCR text panels (multi-font snippets) are generated.
- `--lines` controls how many parallel/perpendicular reasoning diagrams are created.
- Outputs go to `dataset/images`, `dataset/annotations`, and `dataset/text`.

### 3. Train the model
```bash
python src/train.py \
  --manifest dataset/annotations/dataset_manifest.jsonl \
  --text dataset/text/math_text.jsonl \
  --epochs 30 \
  --batch_size 16 \
  --model_file models/active.mathai
```
- Training streams data from disk (no need to load entire JSONL files into RAM).
- Checkpoints are saved in `checkpoints/`, and the best model copies to `models/active.mathai`.
- Use `--save_every` to change periodic checkpoint frequency (default 20 epochs).

### 4. Serve the API + Web UI
```bash
python run.py
# or
uvicorn api:app --app-dir src --reload
```
- Default password for the UI lock screen: `shark` (see `src/web/script.js`).
- Web app allows dataset building, training, model upload/download/delete, and inference.

### 5. Git workflow
Repo ignores generated assets via `.gitignore` (models, dataset, checkpoints, caches). After the first commit:
```bash
git push -u origin main
```

## Project Structure
```
project/
+-- dataset/
¦   +-- annotations/
¦   +-- images/
¦   +-- text/
+-- src/
¦   +-- api.py            # FastAPI server
¦   +-- build_dataset.py  # orchestrates synthetic generation
¦   +-- generate_*.py     # individual generators
¦   +-- model/
¦   ¦   +-- vision_encoder.py
¦   ¦   +-- text_decoder.py
¦   ¦   +-- multimodal_model.py
¦   +-- train.py          # streaming dataloaders + training loop
¦   +-- inference.py      # Predictor wrapper
¦   +-- web/              # static frontend
+-- models/               # exported .mathai checkpoints
+-- checkpoints/
+-- run.py                # helper to launch uvicorn
+-- requirements.txt
+-- README.md
```

## Recommended Workflow
1. Tune dataset counts in the web UI or CLI.
2. Regenerate datasets when new templates are added.
3. Retrain with desired epochs; monitor `/train/status` and `/train/logs` or the UI cards.
4. Export/import models via the UI buttons or by copying `.mathai` files.

Happy hacking!
