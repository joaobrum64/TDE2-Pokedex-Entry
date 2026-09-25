# TDE2 - Pokédex Entry Gen 1

## Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

## macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

## Teste

API:
http://127.0.0.1:8000

Swagger:
http://127.0.0.1:8000/docs