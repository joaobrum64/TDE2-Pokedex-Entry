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

## Scripts de Machine Learning

Os scripts ficam em `backend/ml/` e são executados a partir da raiz do projeto, com o ambiente virtual ativo, sempre no formato `python -m`:

```powershell
python -m backend.ml.teste_pipeline "caminho_da_imagem"   # roda o pipeline da API em uma imagem
python -m backend.ml.avaliar_modelo_v2                    # acurácia do classificador no teste
python -m backend.ml.gerar_avaliacao_pipeline             # monta o conjunto de avaliação do pipeline (uma vez)
python -m backend.ml.avaliar_pipeline v2                  # mede o pipeline completo; "v2" é o rótulo dos arquivos em saidas/
python -m backend.ml.treinar_modelo_v2                    # treina o classificador
python -m backend.ml.gerar_dataset_deteccao               # gera o dataset sintético do detector
python -m backend.ml.treinar_detector_objetos             # treina o detector
```

A API usa apenas `modelo.py` e `pipeline.py`. Experimentos substituídos (classificador v1, detector binário e testes antigos) ficam em `backend/ml/antigos/` e rodam como `python -m backend.ml.antigos.<script>`.

A divisão treino/validação/teste do classificador fica gravada em `datasets/divisao_pokemons.json`, para que todos os modelos sejam comparados nas mesmas imagens. Ela só é sorteada de novo se esse arquivo for apagado.

## Acessos

Frontend:
http://localhost:5173

API:
http://127.0.0.1:8000

Swagger:
http://127.0.0.1:8000/docs