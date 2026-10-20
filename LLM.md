# LLM local (Ollama + RAG)

Guia do componente de LLM do TDE2: como está implementado, por que cada decisão foi tomada e como mexer sem quebrar. Foi escrito para ser lido tanto por uma pessoa quanto pelo Claude: se você usa o Claude Code, peça a ele para ler este arquivo e o `CLAUDE.md` antes de alterar qualquer coisa da IA.

## Objetivo

Depois que o sistema identifica Pokémon numa imagem, o usuário conversa com a "Pokédex" sobre eles. As respostas vêm de um **LLM rodando localmente via Ollama**, com **RAG** sobre o dataset do projeto. O sistema continua **100% offline**.

Comportamento pedido pelo grupo:

1. **Tom de Pokédex da Primeira Geração**, criativo e descritivo, em português do Brasil.
2. **Dados do dataset como fonte principal**; conhecimento próprio do modelo é permitido, mas sempre sinalizado num parágrafo que começa com `Registro externo:`.
3. **Universo restrito à Gen 1** (Red, Green, Blue e Yellow); diferenças relevantes entre versões podem ser comentadas.
4. **Perguntas sobre outros Pokémon da Gen 1**, relacionando com o(s) detectado(s) (ex.: "qual a altura do Blastoise?" com um Pikachu na imagem → responde e compara).
5. **Histórico por conversa**, guardado enquanto o backend estiver rodando.

## Arquitetura

```
Site: componente ChatImagem (frontend/src/App.tsx), um chat por imagem enviada
  │  POST /pokemon/identificar   → resposta traz "conversa_id" (Pokémon detectados já registrados)
  │  POST /conversas             → cria conversa para a busca manual ({"numeros": [25]})
  │  POST /conversas/{id}/perguntar  {"pergunta": "..."}
  ▼
backend/main.py · perguntar()
  │  conversa  = conversas.obter(id)                       backend/conversas.py (só memória)
  │  contexto  = pokedex.montar_contexto_conversa(detectados, [pergunta, perguntas anteriores])
  │  lembrete  = pokedex.lembrete_para_pergunta(detectados, pergunta)
  ▼
backend/llm.py · responder_em_partes(pergunta, contexto, histórico, lembrete=...)
  │  POST http://localhost:11434/api/chat  (stream)
  ▼
StreamingResponse (application/x-ndjson), uma linha por evento:
  {"tipo": "parte", "texto": "..."}   pedaços de texto enquanto o modelo escreve
  {"tipo": "fim", "resposta": "..."}  resposta completa, já sem markdown (vai para o histórico)
  {"tipo": "erro", "detail": "..."}   falha no meio da geração
```

Erros antes de a resposta começar são HTTP normais: 404 (conversa não existe, por exemplo depois de reiniciar o backend), 400/422 (pergunta vazia ou longa demais) e 503 (Ollama fechado, modelo não baixado, tempo esgotado, com mensagem específica para cada caso).

### Arquivos

| Arquivo | Papel |
|---|---|
| `backend/llm.py` | Prompt (`INSTRUCOES`), opções do modelo, chamada ao Ollama em streaming, limpeza de markdown, mensagens de erro, pré-carregamento do modelo |
| `backend/pokedex.py` | Fichas do CSV; `encontrar_citados()` (nomes e números na pergunta); `montar_contexto_conversa()` (o "R" do RAG); `resumo_calculado()`; `lembrete_para_pergunta()`; `montar_contexto()` (texto de uma ficha) |
| `backend/tipos.py` | Tabela de efetividade de tipos da Gen 1 |
| `backend/conversas.py` | Conversas em memória (Pokémon detectados + histórico), até 500 |
| `backend/main.py` | Rotas `/conversas`, `/conversas/{id}`, `/conversas/{id}/perguntar`; cria a conversa em `/pokemon/identificar` |
| `backend/testar_llm.py` | Bateria fixa de perguntas para comparar modelos e mudanças no prompt |
| `frontend/src/App.tsx` | `ChatImagem`: lê o streaming e preenche a bolha da resposta |

## O contexto (RAG)

`montar_contexto_conversa()` monta, nesta ordem:

Primeiro a parte **fixa da conversa**, depois a parte que **muda a cada pergunta**. Essa ordem é proposital: o Ollama reaproveita o começo do prompt que se repete entre perguntas, então nas perguntas seguintes da mesma conversa ele só lê o final (veja "Tempo de resposta").

1. **Pokémon detectados** na imagem e as **fichas completas** deles: dados, golpes de Red/Blue, golpes de outras versões quando diferem (27 Pokémon, como o Pikachu no Yellow), fraquezas, resistências e imunidades calculadas pela tabela de tipos, e a descrição da Pokédex (em inglês).
2. **Fichas resumidas das evoluções** dos detectados (só dados básicos e stats), com cabeçalho explícito: "FICHA RESUMIDA DE RAICHU (forma evolutiva de Pikachu; NÃO foi detectado na imagem)". Sem isso, o modelo às vezes descrevia o Raichu quando perguntavam sobre o Pikachu.
3. **Pokémon dos registros citados na pergunta atual**. Quando nenhum é citado, a linha diz "nenhum (se a pergunta fala de um POKéMON pelo nome, ele não existe nos registros de Kanto)": é assim que o modelo sabe recusar Pokémon de outras gerações.
4. **Fichas completas dos citados** (na pergunta e nas 2 perguntas anteriores, para "e a altura dele?" continuar no mesmo Pokémon). Detectados + citados: até 6 fichas completas.
5. **RESUMO CALCULADO** (com 2 ou mais Pokémon envolvidos): ordem de cada stat ("Velocidade, do maior para o menor: Charmander (65) > ..."), quantas vezes um é mais alto ou pesado que o outro e, para cada Pokémon, duas frases curtas de batalha nas duas direções: "Batalha - Squirtle leva vantagem de tipo sobre: Charmander (...)" e "Batalha - quem leva vantagem de tipo sobre Charmander: Squirtle (...)". A direção inversa existe porque "quem vence o Charmander?" era respondido errado quando só havia "X leva vantagem sobre Y"; frases longas com vários multiplicadores também confundiam o modelo com 3 Pokémon.

Para ver o contexto que o modelo recebe:

```powershell
python -c "from backend.pokedex import Pokedex; print(Pokedex().montar_contexto_conversa([6], ['Ele é bom contra um Blastoise?']))"
```

## Por que está assim (lições dos testes)

Foram testados `llama3.2:3b`, `gemma3:4b`, `qwen2.5:7b`, `qwen3:4b` e `llama3.1:8b` com a bateria de `testar_llm.py`. O que se aprendeu, e que explica decisões que podem parecer exageradas:

- **Modelos pequenos erram comparações de números** ("Squirtle, com 43, é mais rápido que Charmander, com 65") **e vantagens de tipo**. `qwen2.5:7b` e `llama3.1:8b` respondem "sim" para "Fogo é super efetivo contra Água?" mesmo sem prompt nenhum. Por isso o backend **calcula** comparações e vereditos e entrega prontos no RESUMO CALCULADO. Não remova isso achando que o modelo dá conta.
- **`qwen2.5:7b` ignora os dados quando eles contrariam o que ele "acredita"**, mesmo com o resultado colado na pergunta. `llama3.1:8b` segue os dados. Esse foi o principal motivo da escolha.
- **Exemplos no prompt vazam.** Frases genéricas dos exemplos ("vulnerável a ataques de energia", "bandos de X costumam...") reaparecem em respostas de outros Pokémon, e um exemplo comparando Onix e Geodude fez o modelo inventar que "Geodude é evolução do Onix". O único exemplo restante (Onix) usa frases que só servem para o Onix. Se acrescentar exemplos, faça o mesmo e rode a bateria.
- **Regras longe da pergunta são esquecidas.** "Compare com o detectado" só passou a funcionar com o `lembrete` colado na própria pergunta enviada ao modelo (o histórico guarda a pergunta original, sem o lembrete).
- **`qwen3` não serve nesta versão do Ollama**: o modo de raciocínio não desliga (`think: false` faz o raciocínio vazar para a resposta), o que deixa tudo lento demais na CPU.
- **Perguntas de batalha com 3 Pokémon continuam sendo o ponto mais fraco**: mesmo com as frases prontas, "qual dos três leva vantagem contra o Charmander?" ainda sai errado em cerca de metade das vezes. Perguntas de batalha entre dois Pokémon e comparações de stats saem certas.
- **Temperatura**: 0,6 deixava as respostas instáveis (o mesmo modelo acertava e errava a mesma pergunta); 0,3 ficava correta, mas seca. Ficou em 0,4. Subir aumenta a criatividade e também detalhes inventados sem sinalização.

## Configuração atual (`backend/llm.py`)

| Constante | Valor | Observação |
|---|---|---|
| `MODELO_OLLAMA` | `llama3.1:8b` | ~4,9 GB |
| `temperature` | 0,4 | ver acima |
| `num_ctx` | 8192 | o padrão do Ollama é pequeno e cortaria fichas e histórico sem avisar |
| `num_predict` | 450 | teto de tamanho da resposta |
| `MENSAGENS_DE_HISTORICO` | 12 | perguntas + respostas anteriores enviadas ao modelo |
| `MANTER_CARREGADO` | 30 min | o modelo fica na memória entre perguntas; `aquecer()` o carrega quando a API sobe |
| `think` | `False` | desliga o modo de raciocínio em modelos que têm |

**Tempo de resposta** (medido numa CPU sem placa de vídeo):
- A **primeira pergunta de cada conversa** é a mais lenta: o modelo precisa ler o prompt inteiro (~2.100 tokens com 1 Pokémon, ~3.700 com 3), o que leva de 40 a 45 s, mais a escrita da resposta (~7 tokens/s, de 5 a 25 s).
- As **perguntas seguintes na mesma conversa** são bem mais rápidas, porque o começo do prompt é reaproveitado: a leitura cai para ~1 s e sobra só a escrita (5 a 25 s, conforme o tamanho da resposta).
- O streaming faz o texto aparecer assim que o modelo começa a escrever. Com GPU, tudo leva cerca de 3 s.
- Para isso, o prompt foi enxugado (sem a lista dos 151 nomes, sem frases repetidas nas fichas, evoluções resumidas), o que cortou 25% dos tokens e 27 a 31% do tempo de leitura, sem perder qualidade na bateria. Antes de aumentar o prompt, meça o efeito no tempo.
- O `llama3.2:3b` leva cerca de 25 s na CPU, mas erra muito mais (Eevee "não evolui", Fogo vence Água, inventa rotas).

## Resultado atual da bateria (`llama3.1:8b`)

Nas duas últimas rodadas, 11 e 12 de 12 respostas corretas: golpes completos com nível, evoluções do Eevee, Especial único do Mewtwo (154), mais rápido entre os iniciais, Charizard em desvantagem contra Blastoise, comparação de altura Blastoise × Pikachu (4 vezes), Lucario recusado, pergunta fora do tema recusada e comparação de peso usando o histórico.

Limitações conhecidas:
- Em descrições livres, às vezes inventa detalhes pequenos sem marcar como `Registro externo` (ex.: cor dos olhos do Pikachu).
- Sobre lugares onde o Pokémon aparece, costuma só dizer que não consta, sem `Registro externo`.
- O tom de Pokédex aparece, mas nem sempre é tão rico quanto nos jogos.

## Como mexer com segurança

1. Altere o prompt, as opções ou o contexto.
2. Rode a bateria pelo menos duas vezes (a resposta varia entre execuções):

```powershell
python -m backend.testar_llm                       # modelo configurado em llm.py
python -m backend.testar_llm llama3.1:8b qwen2.5:7b # comparar modelos
python -m backend.testar_llm --cpu --rapido        # tempo na CPU, só 3 perguntas
```

3. Leia `saidas/teste_llm_<modelo>.md`: cada caso traz a resposta esperada. Uma mudança só vale se não piorar nenhum caso que antes acertava.
4. Para casos novos, acrescente em `CASOS` no `testar_llm.py`.

## Regras do projeto

- Código, comentários e mensagens ao usuário em **português**.
- **Offline**: só o Ollama local. Nada de APIs na nuvem.
- **Privacidade**: perguntas, respostas e imagens nunca vão para o disco; o histórico some quando o backend para.
- Mudou o formato da ficha? Edite `backend/pokedex.py` e os tipos TypeScript em `frontend/src/App.tsx` juntos.
