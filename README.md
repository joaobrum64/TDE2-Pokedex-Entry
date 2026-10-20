# TDE2 - Pokédex Entry Gen 1

## USAR PÓS SETUP

```powershell
.venv\Scripts\activate
npm run start-all
```

## SETUP Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/reqs.txt
npm install
cd frontend
npm install
cd ..
npm run start-all
```

## SETUP macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/reqs.txt
npm install
cd frontend
npm install
cd ..
npm run start-all
```

## LLM local (Ollama)

As perguntas no chat são respondidas por um LLM aberto rodando na própria máquina, via [Ollama](https://ollama.com). Nada é enviado para a internet.

Modelo usado: **`llama3.1:8b`** (~4,9 GB).

### SETUP

1. Instale o Ollama:
   - Windows: baixe e rode o instalador em https://ollama.com/download ou rode `winget install --id Ollama.Ollama -e`
   - macOS: baixe em https://ollama.com/download ou rode `curl -fsSL https://ollama.com/install.sh | sh`

2. Baixe o modelo:

```bash
ollama pull llama3.1:8b
```

3. Confira se ele aparece em `ollama list`.

O Ollama precisa estar aberto enquanto o projeto roda. Sem ele, o resto do site funciona normalmente e o chat indica o que fazer (abrir o Ollama ou baixar o modelo).

## Acessos

Frontend:
http://localhost:5173

API:
http://127.0.0.1:8000

Swagger:
http://127.0.0.1:8000/docs

Ollama (usado pelo backend):
http://localhost:11434