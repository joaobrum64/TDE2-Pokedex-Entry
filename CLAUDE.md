# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TDE2 — a Gen 1 Pokédex that identifies Pokémon in an uploaded image. FastAPI + PyTorch backend, React/Vite frontend. Code, identifiers, comments and user-facing strings are in Portuguese; keep new code consistent with that.

## Commands

All commands run from the repo root unless noted. The Python venv (`.venv`) must be active for anything that starts the backend.

```powershell
.venv\Scripts\activate
npm run start-all          # backend (uvicorn --reload, :8000) + frontend (vite, :5173) via concurrently
npm run backend            # uvicorn backend.main:app --reload
npm run frontend           # vite dev server only
```

First-time setup: `pip install -r backend/reqs.txt`, then `npm install` in both the root and `frontend/`.

Frontend (run inside `frontend/`):

```powershell
npm run lint               # eslint
npm run build              # tsc -b && vite build (this is the only type-check)
```

ML scripts (run from the repo root, always as modules):

```powershell
python -m backend.ml.teste_pipeline "path\to\image"   # run the API's pipeline on one image; writes an annotated copy to saidas/
python -m backend.ml.avaliar_modelo_v2                # classifier accuracy on the saved test split
python -m backend.ml.treinar_modelo_v3 <arch>         # v3 recipe on the cleaned split; arch = resnet18 | efficientnet_b0 | convnext_tiny -> modelos/pokemon_<arch>_v3.pth
python -m backend.ml.avaliar_classificador <file.pth>  # top-1/top-5 on original and cleaned test (v2: 78.92% / 84.72%)
python -m backend.ml.avaliar_pipeline <label> [classifier.pth|-] [detector.pth]  # pipeline metrics with other model files
python -m backend.ml.calibrar_limiares <label> [classifier.pth|-] [detector.pth]  # grid-search the pipeline thresholds (~3 min)
python -m backend.ml.gerar_dataset_deteccao_v2         # synthetic detection scenes from the cleaned split -> datasets/deteccao_v2 (~6 min)
python -m backend.ml.treinar_detector_objetos_v2 [continuar]  # ~3.5 min/epoch, 8 epochs -> modelos/pokemon_object_detector_v2.pth
python -m backend.ml.avaliar_pipeline <label>         # end-to-end pipeline metrics (~3 min on GPU) -> saidas/avaliacao_pipeline_<label>.{json,csv}
```

`avaliar_pipeline` reads `datasets/avaliacao_pipeline/gabarito.json`, built once by `python -m backend.ml.gerar_avaliacao_pipeline` (fixed seed; 2855 images: classifier test split as single-Pokémon cases, Corel-5k as no-Pokémon cases, side-by-side mosaics of real test images, and the synthetic sprite test scenes). Results are compared by Pokémon names per image, not by box position. This is the yardstick for any change to models or thresholds: the classifier's own test accuracy (78.92% for v2) does not predict pipeline behaviour, which scored 49.29% exact on single-Pokémon images at the v2 baseline. Current API (classifier v3 + detector v2 + same-species merge + calibrated thresholds): unico 80.97%, nenhum false positives 3.20% (v2: 15.40%), mosaico_2 65.33%, mosaico_3 50.00%, sprites 89.33% — saved as `saidas/avaliacao_pipeline_final.json`. The old detector had trained on all Corel images (including the evaluation's no-Pokémon images), so its false-positive figure was optimistic; `gerar_dataset_deteccao_v2` excludes the evaluation and calibration Corel photos and their `_aug` variants.

There is no automated test suite and no Python linter configured. The `teste_*.py` files in `backend/ml/` are manual scripts (see below), not pytest tests.

Swagger UI is at http://127.0.0.1:8000/docs.

## Architecture

### Request flow

`frontend/src/App.tsx` (the whole UI is this one component) POSTs the image as multipart field `arquivo` to `POST /pokemon/identificar`. `backend/main.py` opens it with PIL and hands it to `PipelinePokemon.processar()`, then joins each result to a row of `datasets/pokemon_gen1.csv` and returns `{quantidade, pokemons: [{pokemon, confianca, confianca_detector, box, fallback, top_5}]}`. The frontend's `ResultadoPokemon` / `Pokemon` types mirror that response and the CSV columns by hand — change them together.

The backend URL is hardcoded in `App.tsx` (`API_URL`), and the allowed CORS origins are hardcoded in `main.py` (ports 5173 only). Sprites are served from `datasets/sprites/{pokedex_number}.png` via the `/sprites` static mount.

### Two-model pipeline (`backend/ml/pipeline.py`)

1. **Detector** — Faster R-CNN (ResNet50-FPN), 2 classes (background / pokemon), loaded from `modelos/pokemon_object_detector_v2.pth` (trained on `datasets/deteccao_v2`; the old sprite-only `pokemon_object_detector.pth` is kept for comparison). Boxes below `SCORE_MINIMO_DETECTOR` are dropped, then `remover_boxes_redundantes` removes nested boxes using intersection-over-*smaller*-area (`IOA_THRESHOLD`), not IoU.
2. **Classifier** — 151 classes (`backend/ml/modelo.py`, `modelos/pokemon_resnet18_v3.pth`; v2 kept for comparison). `criar_arquitetura` builds the network from the checkpoint's `architecture` key, so swapping in a v3 checkpoint of another architecture only means changing `MODELO_PATH`. Each detected box is cropped and classified; results under `CONFIANCA_MINIMA_CLASSIFICADOR` are discarded.
3. **Same-species merge** — `unir_mesma_especie` keeps only the most confident of two same-species results whose boxes overlap (`IOA_MESMA_ESPECIE`); the detector often puts two overlapping boxes on one Pokémon. Non-overlapping duplicates (two Pikachus) survive.
4. **Fallback** — if nothing survives, the whole image is classified and accepted only above the stricter `CONFIANCA_MINIMA_FALLBACK`; the result is flagged `fallback: True` with `confianca_detector: None`. An empty list means "no Gen 1 Pokémon found".

Confidences are percentages (0–100), not probabilities. v3 was trained with label smoothing, so its confidences are lower than v2's (a correct answer at 60% is normal).

**Thresholds are per classifier.** `SCORE_MINIMO_DETECTOR` / `CONFIANCA_MINIMA_CLASSIFICADOR` / `CONFIANCA_MINIMA_FALLBACK` (now 0.90 / 10 / 85) were chosen by `python -m backend.ml.calibrar_limiares <label> [file.pth]`, which runs the models once over a calibration set disjoint from the evaluation set (validation split, other Corel images, validation mosaics and sprite scenes) and grid-searches the three thresholds, keeping false positives <= 3%. Re-run it and update `pipeline.py` whenever the classifier or detector changes.

The classifier's class names come from the checkpoint (`checkpoint["classes"]`), which are the folder names of `datasets/pokemons/` at training time (e.g. `Farfetchd`, `MrMime`). `main.py` maps them to CSV rows through `normalizar_nome` (drops spaces, dots and apostrophes; keeps ♀/♂ so the two Nidoran stay distinct) and refuses to start if any class has no CSV row.

Both models are loaded at import time of `backend.main`, so the API will not start if either `.pth` file is missing.

### Data and model files are not in git

`modelos/*` and `datasets/` are gitignored. Only `datasets/pokemon_gen1.csv` and `datasets/sprites/` are tracked; anything else under `datasets/` needs `git add -f` to be shared. A fresh clone cannot run the API until the `.pth` files are placed in `modelos/` (copied in or retrained). Training additionally needs, under `datasets/`:

- `pokemons/<Name>/*.{png,jpg,...}` — classifier images, one folder per class
- `naopokemons/Corel-5k/` — non-Pokémon images, used as backgrounds for synthetic detection data and as negatives
- `deteccao/{train,val,test}/{images,labels}` — generated, not downloaded (see below)

### ML scripts (`backend/ml/`)

The folder is flat on purpose; the filename prefix is the organisation. Apart from `pipeline.py` and `modelo.py` (the only modules the API imports), everything is a standalone script with a `__main__` block and module-level constants for paths and hyperparameters — there is no CLI parsing beyond an optional image path in `sys.argv[1]`.

**Import convention:** every script imports siblings package-style (`from backend.ml.preparar_dados import ...`) and must be run from the repo root as a module (`python -m backend.ml.<script>`). Running a file directly (`python backend/ml/x.py`) fails. There are no `__init__.py` files; `backend` works as a namespace package. Each file computes its own `ROOT_DIR` from `__file__`, so moving a script between `ml/` and `ml/antigos/` means adjusting the number of `.parent` hops.

Script families by prefix:

- `preparar_dados*.py` — Datasets/DataLoaders, imported by the train/eval scripts.
- `gerar_*.py` — build the synthetic detection dataset by pasting sprites from `datasets/sprites` onto Corel-5k backgrounds, writing one JSON label per image (`objects[].bbox` with `x1,y1,x2,y2`).
- `treinar_*.py` — train and write a checkpoint dict (`model_state_dict`, `classes`, `num_classes`, ...) into `modelos/`.
- `avaliar_*.py`, `analis*.py` — metrics on the test split, rejection-threshold analysis.
- `teste_*.py`, `visualizar_*.py` — manual, single-image checks. `teste_pipeline.py` exercises the real `PipelinePokemon`, so it always reflects the thresholds the API uses.

**Fixed split:** `preparar_dados.py` reads the classifier's train/val/test split from `datasets/divisao_pokemons.json` (relative paths, 7481/1544/1755 images) and only draws a new 70/15/15 split when that file is missing. Images deleted from the dataset drop out of their set; images not listed in the file are ignored with a warning. Do not delete or regenerate this file when comparing models — v2 scores 78.92% on its test set and new models are judged against that.

**Cleaning list:** `datasets/limpeza_pokemons.csv` (columns `imagem, conjunto, motivo, copia_de, acao`) is hand-reviewed by the user. `carregar_amostras()` / `testar_dataloaders()` drop rows whose `acao` is `remover` from all three sets by default; pass `aplicar_exclusoes=False` to get the original split. `acao` values `manter` and `revisar` keep the image. `motivo` is `copia_mesma_classe` (`copia_de` = the kept copy, which is not itself listed), `classe_duvidosa` (same image in two class folders; both listed, each pointing at the other) `varios_pokemons` (`copia_de` = other species seen) or `classe_divergente` (`copia_de` = species and confidence predicted by an out-of-fold ConvNeXt-Tiny linear probe; v2 itself is useless for this because it memorised its training images). Never overwrite this file — it holds the user's decisions. `avaliar_modelo_v2` reports both the original test (78.92%, used for "better than the previous model") and the cleaned test (80.99% for v2). `gerar_avaliacao_pipeline` and `analisar_duplicatas` deliberately use the original split.

To reproduce the two models the API uses:

```powershell
python -m backend.ml.treinar_modelo_v2          # -> modelos/pokemon_resnet18_v2.pth
python -m backend.ml.gerar_dataset_deteccao     # -> datasets/deteccao/
python -m backend.ml.treinar_detector_objetos   # -> modelos/pokemon_object_detector.pth
```

`backend/ml/antigos/` holds superseded experiments that the API does not use: the v1 classifier (`treinar_modelo.py`, `avaliar_modelo.py`, `prever_imagem.py`, `pokemon_resnet18.pth`), a binary pokemon-vs-not ResNet18 (`preparar_dados_binario.py`, `treinar_detector.py`, `avaliar_detector.py`, `teste_detector.py`, `pokemon_detector_v2.pth`) that the Faster R-CNN detector replaced, and older test scripts that carry their own copies of the pipeline thresholds. They still import and run as `python -m backend.ml.antigos.<script>`.
