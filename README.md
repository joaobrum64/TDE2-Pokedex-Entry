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

As perguntas sobre um Pokémon ("Pergunte sobre este Pokémon" no site) são respondidas por um LLM aberto rodando na própria máquina, via [Ollama](https://ollama.com). Nada é enviado para a internet.

Modelo usado: **`llama3.2:3b`** (~2 GB, roda na CPU).

### Instalação (uma vez)

1. Instale o Ollama:
   - Windows: baixe e rode o instalador em https://ollama.com/download
   - macOS: baixe em https://ollama.com/download ou rode `curl -fsSL https://ollama.com/install.sh | sh`
2. Baixe o modelo:

```bash
ollama pull llama3.2:3b
```

3. Confira se ele aparece em `ollama list`.

O Ollama precisa estar aberto (ícone na bandeja/barra de menus) enquanto o projeto roda. Sem ele, o resto do site funciona normalmente e só as perguntas mostram um aviso de que o LLM está indisponível.

### Como funciona (RAG)

1. O site envia a pergunta para `POST /pokemon/{numero}/perguntar`.
2. O backend busca a ficha do Pokémon em `datasets/pokemon_gen1.csv` e a transforma em texto (`montar_contexto` em `backend/pokedex.py`).
3. `backend/llm.py` envia ao Ollama as regras + a ficha + a pergunta, com temperatura 0.
4. O LLM responde só com os dados da ficha; se a informação não estiver nela (rotas, fraquezas de tipo, outras gerações), responde que a Pokédex não tem essa informação.

Para trocar de modelo, baixe-o com `ollama pull` e altere `MODELO_OLLAMA` em `backend/llm.py`. Detalhes em [LLM.md](LLM.md).

### Perguntas para testar

| Pokémon | Pergunta | Resposta esperada |
|---|---|---|
| 25 (Pikachu) | Quais golpes o Pikachu aprende e em que nível? | Os 7 golpes, com os níveis |
| 133 (Eevee) | Para quem o Eevee evolui? | Vaporeon, Jolteon e Flareon |
| 150 (Mewtwo) | Qual o Ataque Especial do Mewtwo? | Explica que na Gen 1 há um único stat Especial (154) |
| 25 (Pikachu) | Em que rota eu encontro o Pikachu? | Que a Pokédex não tem essa informação |

## Acessos

Frontend:
http://localhost:5173

API:
http://127.0.0.1:8000

Swagger:
http://127.0.0.1:8000/docs

Ollama (usado pelo backend):
http://localhost:11434