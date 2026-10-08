# Integração do LLM local (Ollama + RAG)

Guia para implementar o componente de LLM do TDE2. Foi escrito para ser lido tanto por uma pessoa quanto pelo Claude: se você usa o Claude Code, peça a ele para ler este arquivo e o `CLAUDE.md` antes de começar.

## Objetivo (requisito do projeto)

Depois que o sistema identifica um Pokémon em uma imagem, o usuário pode fazer perguntas sobre ele. As respostas vêm de um **LLM rodando localmente via Ollama**, com **RAG**: o backend busca os dados do Pokémon no dataset (`datasets/pokemon_gen1.csv`) e injeta esses dados no prompt. O sistema precisa continuar **100% offline**: nenhuma chamada a APIs na nuvem.

## O que já está pronto

| Parte | Onde | Estado |
|---|---|---|
| Rota da API | `POST /pokemon/{numero}/perguntar` em `backend/main.py` | Pronta. Recebe a pergunta, monta o contexto e chama `responder()` |
| Contexto do RAG | `montar_contexto(ficha)` em `backend/pokedex.py` | Pronto. Transforma a ficha em texto em português |
| Ficha do Pokémon | `Pokedex.obter_ficha(nome_ou_numero)` em `backend/pokedex.py` | Pronta |
| Campo de pergunta no site | Componente `PerguntaPokemon` em `frontend/src/App.tsx` | Pronto. Aparece embaixo de cada Pokémon e mostra a resposta ou o erro |
| **Geração da resposta** | **`responder()` em `backend/llm.py`** | **Falta implementar.** Hoje levanta `LLMNaoConfigurado` e a rota devolve 501 |

**A única coisa que precisa ser escrita é o corpo de `responder()` em `backend/llm.py`.** O resto do caminho (site → API → contexto → resposta na tela) já funciona.

## Fluxo de uma pergunta

```
Site (PerguntaPokemon)
  │  POST /pokemon/25/perguntar   {"pergunta": "Quais são os pontos fortes dele?"}
  ▼
backend/main.py · perguntar_sobre_pokemon()
  │  ficha    = pokedex.obter_ficha(25)
  │  contexto = montar_contexto(ficha)          ← o "R" do RAG
  ▼
backend/llm.py · responder(pergunta, contexto, ficha)   ← IMPLEMENTAR AQUI
  │  monta o prompt e chama o Ollama local
  ▼
{"numero": 25, "nome": "Pikachu", "pergunta": "...", "resposta": "..."}
```

## Contrato de `responder()`

```python
def responder(pergunta: str, contexto: str, ficha: dict) -> str
```

- **`pergunta`**: o texto do usuário, já sem espaços nas pontas, com 1 a 500 caracteres.
- **`contexto`**: texto em português com todos os dados do Pokémon (exemplo abaixo).
- **`ficha`**: o mesmo dicionário de `GET /pokemon/{numero}`, para quando for útil ler um campo direto.
- **Retorno**: a resposta em texto puro, que o site mostra num parágrafo.
- **Erros**: se o Ollama não responder (desligado, modelo não baixado, tempo esgotado), levante `LLMIndisponivel("mensagem em português para o usuário")`. A rota transforma isso em 503 e o site mostra a mensagem. **Não deixe outras exceções escaparem.**

Não mude a assinatura da função, a rota nem o formato da resposta: o frontend depende deles.

### Exemplo de `contexto` (Pikachu)

```
Pokémon: Pikachu (nº 25 da Pokédex, Primeira Geração)
Classificação: Mouse Pokémon
Tipo: Elétrico
Altura: 0.4 m
Peso: 6.0 kg
Lendário: não
Stats base: HP 35, Ataque 55, Defesa 30, Especial 50, Velocidade 90 (na Primeira Geração existe um único stat Especial)
Evolui de: não evolui de nenhum Pokémon
Evolui para: Raichu (nº 26)
Golpes aprendidos por nível (Pokémon Red/Blue): Thunder Shock (nível 1), Growl (nível 1), Thunder Wave (nível 9), Quick Attack (nível 16), Swift (nível 26), Agility (nível 33), Thunder (nível 43)
Descrição da Pokédex (em inglês): When several of these POKéMON gather, their electricity could build and cause lightning storms.
```

Para ver o contexto de qualquer Pokémon:

```powershell
python -c "from backend.pokedex import Pokedex, montar_contexto; print(montar_contexto(Pokedex().obter_ficha(6)))"
```

## O que o dataset tem e o que não tem

Isso define o que o LLM consegue responder com segurança:

- **Tem**: nome, número, classificação, tipos, altura, peso, se é lendário, stats base, evoluções (de/para), golpes aprendidos por nível no Red/Blue e a descrição da Pokédex (em inglês, como no jogo).
- **Não tem**: habilidades (não existem na Primeira Geração), vantagens/fraquezas de tipo, locais onde o Pokémon aparece, golpes por TM/HM, dados de outras gerações, e "jogos em que aparece", que foi tirado do escopo.
- **Particularidades da Geração 1**: existe um único stat **Especial** (não há Ataque Especial e Defesa Especial separados), e não existem os tipos Fada, Aço e Sombrio. O Clefairy é Normal; o Magnemite é só Elétrico.

Se uma pergunta pedir algo que não está no contexto, o LLM deve dizer que não tem essa informação, em vez de inventar. Isso é o que torna o RAG "consistente com o dataset", como pede o documento do projeto.

## Implementação sugerida

O Ollama expõe uma API HTTP em `http://localhost:11434`. Dá para chamá-la só com a biblioteca padrão do Python, sem acrescentar dependências:

```python
import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODELO_OLLAMA = "llama3.2:3b"   # troque pelo modelo escolhido pelo grupo
TEMPO_LIMITE_SEGUNDOS = 120

INSTRUCOES = (
    "Você é a Pokédex da Primeira Geração. Responda em português do Brasil, "
    "de forma curta e direta, usando APENAS os dados do Pokémon fornecidos "
    "abaixo. Se a resposta não estiver nesses dados, diga que a Pokédex não "
    "tem essa informação. Não invente dados de outras gerações."
)


def responder(pergunta, contexto, ficha):
    corpo = {
        "model": MODELO_OLLAMA,
        "stream": False,
        "messages": [
            {"role": "system", "content": f"{INSTRUCOES}\n\nDados do Pokémon:\n{contexto}"},
            {"role": "user", "content": pergunta},
        ],
    }

    requisicao = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(corpo).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(requisicao, timeout=TEMPO_LIMITE_SEGUNDOS) as resposta:
            dados = json.loads(resposta.read().decode("utf-8"))

    except (urllib.error.URLError, TimeoutError) as erro:
        raise LLMIndisponivel(
            "Não foi possível falar com o Ollama. Verifique se ele está aberto "
            f"e se o modelo {MODELO_OLLAMA} foi baixado."
        ) from erro

    return dados["message"]["content"].strip()
```

Mantenha as classes `LLMNaoConfigurado` e `LLMIndisponivel` em `backend/llm.py`, porque `main.py` as importa. Depois de implementar, `LLMNaoConfigurado` deixa de ser usada, mas pode continuar no arquivo.

### Escolha do modelo

- O grupo vai rodar **sem GPU**. Modelos pequenos (de 1 a 4 bilhões de parâmetros, como `llama3.2:3b`, `qwen2.5:3b` ou `gemma3:4b`) respondem em alguns segundos no processador. Modelos de 7–8B funcionam, mas podem levar mais de 30 segundos por resposta.
- Prefira um modelo que escreva bem em português.
- Coloque o nome do modelo numa constante no topo de `llm.py` e documente no README qual foi escolhido e o comando `ollama pull` correspondente.

### Melhorias opcionais

- **Mais contexto**: incluir também a ficha das evoluções (`ficha["evolui_para"]` traz os números; use `Pokedex().obter_ficha(numero)` e `montar_contexto`) permite responder "o Raichu é mais rápido que o Pikachu?".
- **Pergunta comparando dois Pokémon**: exigiria mudar a rota e o frontend; combine com o grupo antes.
- **Histórico de conversa**: hoje cada pergunta é independente. Guardar o histórico exige mudar a rota e o componente `PerguntaPokemon`.

## Como testar

1. Instale o Ollama (https://ollama.com), abra-o e baixe o modelo, por exemplo `ollama pull llama3.2:3b`.
2. Suba o projeto: `npm run start-all` (com o ambiente virtual ativo).
3. **Pela API**: abra http://127.0.0.1:8000/docs, encontre `POST /pokemon/{id_pokemon}/perguntar`, use `25` e o corpo `{"pergunta": "Quais golpes ele aprende?"}`.
4. **Pelo site**: abra http://localhost:5173, busque o número 25 na busca manual (ou identifique uma imagem) e use o campo "Pergunte sobre este Pokémon".

Respostas esperadas da API:

| Situação | Código | Resposta |
|---|---|---|
| Antes de implementar | 501 | `{"detail": "O LLM local ainda não foi configurado."}` |
| Ollama fechado ou sem o modelo | 503 | `{"detail": "<mensagem de LLMIndisponivel>"}` |
| Funcionando | 200 | `{"numero": 25, "nome": "Pikachu", "pergunta": "...", "resposta": "..."}` |
| Número que não existe | 404 | `{"detail": "Pokémon não encontrado."}` |
| Pergunta vazia | 400 ou 422 | Mensagem de validação |

Perguntas para conferir se o RAG está funcionando:

- "Quais golpes o Pikachu aprende e em que nível?" → deve listar os 7 golpes do contexto, com os níveis certos.
- "Para quem o Eevee evolui?" (número 133) → Vaporeon, Jolteon e Flareon.
- "Qual o Ataque Especial do Mewtwo?" (número 150) → deve explicar que na Geração 1 existe um único stat Especial, de valor 154.
- "Em que rota eu encontro o Pikachu?" → deve dizer que a Pokédex não tem essa informação.

## Regras do projeto para quem for mexer no código

- Código, comentários e mensagens ao usuário em **português**, no mesmo estilo do resto do projeto.
- Sistema **offline**: só o Ollama local. Nada de chaves de API, OpenAI, Anthropic etc.
- **Privacidade**: não grave perguntas, respostas nem imagens em disco.
- A rota é síncrona (`def`, não `async def`) de propósito: o FastAPI a executa em outra thread e o site continua respondendo enquanto o LLM pensa. Mantenha assim, ou use um cliente assíncrono de verdade se mudar para `async def`.
- Para mudar o formato da ficha, edite `backend/pokedex.py` e os tipos TypeScript em `frontend/src/App.tsx` juntos.
