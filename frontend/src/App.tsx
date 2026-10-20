import { useEffect, useRef, useState, type ChangeEvent, type CSSProperties } from "react"
import "./App.css"


type Referencia = {
  numero: number
  nome: string
}


type Golpe = {
  nivel: number
  nome: string
}


type Pokemon = {
  numero: number
  nome: string
  classificacao: string
  tipos: string[]
  altura_m: number | null
  peso_kg: number | null
  stats: {
    hp: number
    ataque: number
    defesa: number
    especial: number
    velocidade: number
  }
  lendario: boolean
  descricao: string
  evolui_de: Referencia[]
  evolui_para: Referencia[]
  golpes: Golpe[]
  sprite: string
}


type Previsao = Referencia & {
  confianca: number
}


type ResultadoPokemon = {
  pokemon: Pokemon
  confianca: number
  confianca_detector: number | null
  box: number[]
  fallback: boolean
  top_5: Previsao[]
}


type RespostaIdentificacao = {
  quantidade: number
  conversa_id: string | null
  largura: number
  altura: number
  pokemons: ResultadoPokemon[]
}


function formatarMedida(valor: number | null, unidade: string) {
  return valor === null ? "Não informado" : `${valor} ${unidade}`
}


const API_URL = "http://127.0.0.1:8000"



// Mostra as caixas só para resultados do detector; no fallback a
// caixa é a imagem inteira e não acrescenta informação.
function temCaixa(resultado: ResultadoPokemon) {
  return !resultado.fallback && resultado.box.length === 4
}


// Faixinha decorativa no topo do cabeçalho e do painel de resultados,
// preenchida com o peixinho girando repetido lado a lado.
function PadraoDecorativo() {
  return <div className="padrao-decorativo" aria-hidden="true" />
}


// Ícones simples usados nos avisos (traço, sem preenchimento).
const ICONES = {
  cadeado: "M7 11V8a5 5 0 0 1 10 0v3M6 11h12v9H6z",
  computador: "M4 5h16v11H4zM9 20h6M12 16v4",
  alerta: "M12 4l9 16H3zM12 10v4M12 17v.5",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 11v5M12 8v.5",
}


function Icone({ nome }: { nome: keyof typeof ICONES }) {
  return (
    <svg
      className="icone-aviso"
      viewBox="0 0 24 24"
      width="18"
      height="18"
      aria-hidden="true"
    >
      <path d={ICONES[nome]} />
    </svg>
  )
}


const MARCADOR_EXTERNO = "Registro externo:"


// Resposta da IA com o trecho "Registro externo" destacado: ele vem do
// conhecimento do próprio modelo, e não do dataset, e pode ter erros.
function TextoResposta({ texto }: { texto: string }) {
  const posicao = texto.indexOf(MARCADOR_EXTERNO)

  if (posicao < 0) {
    return <>{texto}</>
  }

  const principal = texto.slice(0, posicao).trim()
  const externo = texto.slice(posicao + MARCADOR_EXTERNO.length).trim()

  return (
    <>
      {principal}

      <span className="registro-externo">
        <span className="registro-externo-rotulo">
          Fora do dataset · pode conter imprecisões
        </span>
        {externo}
      </span>
    </>
  )
}


// Faixa "Sobre esta Pokédex", antes do rodapé, com os pontos de ética do
// projeto: privacidade, uso local, possíveis erros e limitações.
function AvisosEticos() {
  const avisos: { icone: keyof typeof ICONES, titulo: string, texto: string }[] = [
    {
      icone: "cadeado",
      titulo: "Privacidade",
      texto:
        "As imagens ficam só na memória durante a identificação e são " +
        "descartadas em seguida. Nada é gravado.",
    },
    {
      icone: "computador",
      titulo: "Tudo roda localmente",
      texto:
        "Os modelos e a IA rodam neste computador. Imagens e perguntas " +
        "não vão para a internet nem são usadas para treinar nada.",
    },
    {
      icone: "alerta",
      titulo: "A IA pode errar",
      texto:
        "Pokémon parecidos, como evoluções da mesma linha, podem ser " +
        "confundidos, e o chat pode se enganar. Confira a confiança e as " +
        "outras possibilidades.",
    },
    {
      icone: "info",
      titulo: "Só a Primeira Geração",
      texto:
        "Reconhece apenas os 151 Pokémon de Kanto, com dados de Red, Green, " +
        "Blue e Yellow. Pokémon de outras gerações não são identificados.",
    },
  ]

  return (
    <section className="avisos-eticos" aria-label="Sobre esta Pokédex">
      <h2>Sobre esta Pokédex</h2>

      <div className="avisos-eticos-lista">
        {avisos.map((aviso) => (
          <div className="aviso-etico" key={aviso.titulo}>
            <Icone nome={aviso.icone} />

            <div>
              <strong>{aviso.titulo}</strong>
              <p>{aviso.texto}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}


type MensagemChat = {
  autor: "usuario" | "ia" | "erro"
  texto: string
}


// Chat com o LLM local sobre a imagem enviada (uma conversa por imagem
// ou por busca manual; rota POST /conversas/{id}/perguntar, ver LLM.md).
// O backend guarda o histórico e já sabe quais Pokémon foram detectados.
function ChatImagem({
  conversaId,
  nomes,
}: {
  conversaId: string
  nomes: string[]
}) {
  const [pergunta, setPergunta] = useState("")
  const [mensagens, setMensagens] = useState<MensagemChat[]>([])
  const [carregando, setCarregando] = useState(false)

  // Verdadeiro depois que o primeiro pedaço da resposta chegou: a partir
  // daí o texto aparece na própria bolha e a animação de "digitando" sai.
  const [recebendo, setRecebendo] = useState(false)

  // Mantém o histórico rolado até a última mensagem. Rola só a caixa do
  // chat, sem mexer na página.
  const mensagensRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const caixa = mensagensRef.current

    if (caixa) {
      caixa.scrollTo({ top: caixa.scrollHeight, behavior: "smooth" })
    }
  }, [mensagens, carregando])

  async function enviarPergunta(textoSugerido?: string) {
    const texto = (textoSugerido ?? pergunta).trim()

    // Enter durante o carregamento não envia de novo.
    if (!texto || carregando) {
      return
    }

    setMensagens((atuais) => [
      ...atuais,
      { autor: "usuario", texto },
    ])

    setPergunta("")
    setCarregando(true)

    try {
      const retorno = await fetch(
        `${API_URL}/conversas/${conversaId}/perguntar`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pergunta: texto }),
        }
      )

      if (!retorno.ok || !retorno.body) {
        let detalhe = "Não foi possível obter uma resposta."

        try {
          const dados = await retorno.json()

          if (typeof dados.detail === "string") {
            detalhe = dados.detail
          }
        } catch {
          // A API pode retornar uma resposta que não seja JSON.
        }

        setMensagens((atuais) => [
          ...atuais,
          { autor: "erro", texto: detalhe },
        ])
        return
      }

      // A resposta chega em streaming: uma linha JSON por pedaço de
      // texto ({"tipo": "parte"}), e no fim a resposta completa
      // ({"tipo": "fim"}) ou um erro ({"tipo": "erro"}).
      const leitor = retorno.body.getReader()
      const decodificador = new TextDecoder()

      let pendente = ""
      let textoIa = ""
      let bolhaCriada = false

      const mostrarTextoIa = (novoTexto: string) => {
        textoIa = novoTexto

        if (!bolhaCriada) {
          bolhaCriada = true
          setRecebendo(true)
          setMensagens((atuais) => [
            ...atuais,
            { autor: "ia", texto: novoTexto },
          ])
          return
        }

        setMensagens((atuais) => [
          ...atuais.slice(0, -1),
          { autor: "ia", texto: novoTexto },
        ])
      }

      for (;;) {
        const { value, done } = await leitor.read()

        if (done) {
          break
        }

        pendente += decodificador.decode(value, { stream: true })

        const linhas = pendente.split("\n")
        pendente = linhas.pop() ?? ""

        for (const linha of linhas) {
          if (!linha.trim()) {
            continue
          }

          const evento = JSON.parse(linha)

          if (evento.tipo === "parte") {
            mostrarTextoIa(textoIa + evento.texto)
          } else if (evento.tipo === "fim") {
            mostrarTextoIa(evento.resposta)
          } else if (evento.tipo === "erro") {
            setMensagens((atuais) => [
              ...atuais,
              { autor: "erro", texto: evento.detail },
            ])
          }
        }
      }
    } catch (erroConexao) {
      console.error(erroConexao)

      setMensagens((atuais) => [
        ...atuais,
        { autor: "erro", texto: "Falha ao conectar no backend." },
      ])
    } finally {
      setCarregando(false)
      setRecebendo(false)
    }
  }

  // Perguntas de exemplo para o chat vazio, conforme a quantidade de
  // Pokémon na imagem.
  const sugestoes =
    nomes.length > 1
      ? [
        "Qual deles é o mais rápido?",
        "Quem leva vantagem numa batalha entre eles?",
        `Quais golpes o ${nomes[0]} aprende?`,
      ]
      : [
        "O que você sabe sobre ele?",
        "Quais golpes ele aprende e em que nível?",
        "Ele é mais alto que um Pikachu?",
      ]

  return (
    <aside className="painel-chat">
      <PadraoDecorativo />

      <div className="painel-chat-corpo">
        <header className="painel-chat-cabecalho">
          <h3>
            Pergunte à Pokédex
          </h3>

          <p>
            Sobre {nomes.join(", ")} ou qualquer Pokémon de Kanto. A IA
            roda localmente.
          </p>
        </header>

        {/* O histórico rola dentro desta caixa: com o mouse em cima, a
            roda do mouse move só o chat. */}
        <div className="chat-mensagens" ref={mensagensRef}>
          {mensagens.length === 0 && (
            <div className="chat-vazio">
              <p>Experimente perguntar:</p>

              {sugestoes.map((sugestao) => (
                <button
                  key={sugestao}
                  className="chat-sugestao"
                  onClick={() => enviarPergunta(sugestao)}
                  disabled={carregando}
                >
                  {sugestao}
                </button>
              ))}
            </div>
          )}

          {mensagens.map((mensagem, indice) => (
            <div
              key={indice}
              className={`chat-bolha chat-bolha--${mensagem.autor}`}
            >
              {mensagem.autor === "ia"
                ? <TextoResposta texto={mensagem.texto} />
                : mensagem.texto}
            </div>
          ))}

          {carregando && !recebendo && (
            <div className="chat-bolha chat-bolha--ia chat-bolha--digitando">
              <span />
              <span />
              <span />
            </div>
          )}
        </div>

        <div className="pergunta-pokemon-linha">
          <input
            type="text"
            value={pergunta}
            maxLength={500}
            placeholder="Escreva sua pergunta..."
            onChange={(evento) => setPergunta(evento.target.value)}
            onKeyDown={(evento) => {
              if (evento.key === "Enter") {
                enviarPergunta()
              }
            }}
          />

          <button
            className="botao-primario"
            onClick={() => enviarPergunta()}
            disabled={carregando}
          >
            {carregando ? "Pensando..." : "Perguntar"}
          </button>
        </div>

        <p className="nota-aviso">
          <Icone nome="alerta" />
          A IA pode errar. Os dados vêm das fichas do projeto; trechos
          marcados como fora do dataset vêm do conhecimento do modelo.
        </p>
      </div>
    </aside>
  )
}


// Tipos em inglês, como nos jogos originais, com as cores das etiquetas
// dos jogos (o estilo pixelado fica no CSS, em .etiqueta-tipo).
const TIPOS: Record<string, { nome: string, cor: string }> = {
  normal: { nome: "NORMAL", cor: "#a8a878" },
  fire: { nome: "FIRE", cor: "#f08030" },
  water: { nome: "WATER", cor: "#6890f0" },
  electric: { nome: "ELECTR", cor: "#f8d030" },
  grass: { nome: "GRASS", cor: "#78c850" },
  ice: { nome: "ICE", cor: "#98d8d8" },
  fighting: { nome: "FIGHT", cor: "#c03028" },
  poison: { nome: "POISON", cor: "#a040a0" },
  ground: { nome: "GROUND", cor: "#e0c068" },
  flying: { nome: "FLYING", cor: "#a890f0" },
  psychic: { nome: "PSYCHC", cor: "#f85888" },
  bug: { nome: "BUG", cor: "#a8b820" },
  rock: { nome: "ROCK", cor: "#b8a038" },
  ghost: { nome: "GHOST", cor: "#705898" },
  dragon: { nome: "DRAGON", cor: "#7038f8" },
}


function EtiquetaTipo({ tipo }: { tipo: string }) {
  const dados = TIPOS[tipo] ?? { nome: tipo.toUpperCase(), cor: "#888888" }

  return (
    <span
      className="etiqueta-tipo"
      style={{ "--cor-tipo": dados.cor } as CSSProperties}
    >
      {dados.nome}
    </span>
  )
}


const STATS: { chave: keyof Pokemon["stats"], nome: string }[] = [
  { chave: "hp", nome: "HP" },
  { chave: "ataque", nome: "Ataque" },
  { chave: "defesa", nome: "Defesa" },
  { chave: "especial", nome: "Especial" },
  { chave: "velocidade", nome: "Velocidade" },
]

// Abaixo desta confiança (%), a identificação é marcada como incerta.
// O classificador foi treinado com label smoothing, então acertos entre
// 50% e 90% são normais.
const LIMIAR_INCERTO = 50

// Outra espécie com pelo menos esta confiança (%) também gera o aviso.
const LIMIAR_ALTERNATIVA = 15

// Maior stat base possível na Gen 1 (HP do Chansey é 250).
const STAT_MAXIMO = 255


// Card dos Pokémon encontrados: cabeçalho com a quantidade e os sprites
// clicáveis, setas para o anterior e o próximo e a ficha do selecionado.
// O chat fica num card separado, ao lado (ChatImagem).
function PainelResultados({
  resultados,
  selecionado,
  selecionar,
  encontrados,
}: {
  resultados: ResultadoPokemon[]
  selecionado: number
  selecionar: (indice: number) => void
  // null quando a ficha veio da busca por número.
  encontrados: number | null
}) {
  const atual = Math.min(selecionado, resultados.length - 1)
  const resultado = resultados[atual]
  const pokemon = resultado.pokemon

  const anterior = atual > 0 ? resultados[atual - 1] : null
  const proximo = atual < resultados.length - 1 ? resultados[atual + 1] : null

  const varios = resultados.length > 1
  const temConfianca = resultado.top_5.length > 0

  // Avisa quando a identificação é duvidosa: confiança baixa ou outra
  // espécie com chance considerável.
  const alternativas = resultado.top_5
    .slice(1)
    .filter((previsao) => previsao.confianca >= LIMIAR_ALTERNATIVA)
    .slice(0, 2)

  const incerto =
    temConfianca &&
    (resultado.confianca < LIMIAR_INCERTO || alternativas.length > 0)

  return (
    <div className="painel-resultados">
      <PadraoDecorativo />

      <div className="painel-resultados-corpo">
        {/* Cabeçalho do card: quantos Pokémon foram encontrados, com os
            nomes e os sprites clicáveis para escolher qual mostrar. */}
        <header className="cabecalho-resultados">
          <span className="resumo-numero">
            {encontrados === null ? `#${pokemon.numero}` : encontrados}
          </span>

          <div className="resumo-texto">
            <strong>
              {encontrados === null
                ? "Busca por número"
                : encontrados === 1
                  ? "Pokémon encontrado"
                  : "Pokémon encontrados"}
            </strong>

            <span className="resumo-nomes">
              {resultados.map((item, indice) => (
                <button
                  key={`${item.pokemon.numero}-${indice}`}
                  className={
                    "resumo-nome"
                    + (indice === atual ? " resumo-nome--ativo" : "")
                  }
                  onClick={() => selecionar(indice)}
                  disabled={!varios}
                >
                  {item.pokemon.nome}
                </button>
              ))}
            </span>
          </div>

          <div className="resumo-sprites">
            {resultados.map((item, indice) => (
              <button
                key={`${item.pokemon.numero}-${indice}`}
                className={
                  "resumo-sprite"
                  + (indice === atual ? " resumo-sprite--ativo" : "")
                }
                onClick={() => selecionar(indice)}
                disabled={!varios}
                title={item.pokemon.nome}
                aria-label={`Ver ${item.pokemon.nome}`}
                style={{ animationDelay: `${indice * 90}ms` }}
              >
                <img
                  src={`${API_URL}${item.pokemon.sprite}`}
                  alt=""
                />
              </button>
            ))}
          </div>
        </header>

        <div className="carrossel">
          {varios && (
            <button
              className="seta"
              disabled={!anterior}
              onClick={() => selecionar(atual - 1)}
              aria-label={anterior ? `Ver ${anterior.pokemon.nome}` : undefined}
            >
              <span className="seta-icone">‹</span>
              <span className="seta-nome">{anterior?.pokemon.nome}</span>
            </button>
          )}

          <div className="destaque" key={`destaque-${atual}`}>
            <div className="destaque-sprite">
              <img
                src={`${API_URL}${pokemon.sprite}`}
                alt={`Sprite de ${pokemon.nome}`}
              />
            </div>

            <div className="destaque-nome">
              <h2>
                <span className="destaque-numero">#{pokemon.numero}</span>
                {" "}
                {pokemon.nome}
              </h2>

              <div className="etiquetas">
                {pokemon.tipos.map((tipo) => (
                  <EtiquetaTipo key={tipo} tipo={tipo} />
                ))}

                {pokemon.lendario && (
                  <span className="etiqueta-tipo etiqueta-lendario">
                    LEGEND
                  </span>
                )}
              </div>
            </div>
          </div>

          {varios && (
            <button
              className="seta seta--direita"
              disabled={!proximo}
              onClick={() => selecionar(atual + 1)}
              aria-label={proximo ? `Ver ${proximo.pokemon.nome}` : undefined}
            >
              <span className="seta-nome">{proximo?.pokemon.nome}</span>
              <span className="seta-icone">›</span>
            </button>
          )}
        </div>

        <div className="ficha" key={`ficha-${atual}`}>
          <p className="ficha-descricao">
            <strong className="ficha-classificacao">
              {pokemon.classificacao}.
            </strong>
            {" "}
            {pokemon.descricao}
          </p>

          <dl className="ficha-medidas">
            <div>
              <dt>Altura</dt>
              <dd>{formatarMedida(pokemon.altura_m, "m")}</dd>
            </div>

            <div>
              <dt>Peso</dt>
              <dd>{formatarMedida(pokemon.peso_kg, "kg")}</dd>
            </div>

            {temConfianca && (
              <div>
                <dt>Identificação</dt>
                <dd>{resultado.confianca.toFixed(1)}%</dd>
              </div>
            )}

            {resultado.confianca_detector !== null && (
              <div>
                <dt>Detecção</dt>
                <dd>{resultado.confianca_detector.toFixed(1)}%</dd>
              </div>
            )}
          </dl>

          {resultado.fallback && temConfianca && (
            <p className="ficha-nota">
              Identificação feita usando a imagem completa.
            </p>
          )}

          {incerto && (
            <div className="aviso-incerto" role="note">
              <Icone nome="alerta" />

              <p>
                <strong>Identificação incerta.</strong>{" "}
                {alternativas.length > 0
                  ? "O modelo também considerou " + alternativas
                    .map((previsao) => `${previsao.nome} (${previsao.confianca.toFixed(0)}%)`)
                    .join(" e ") + "."
                  : `A confiança ficou abaixo de ${LIMIAR_INCERTO}%.`}
                {" "}Pokémon parecidos, como evoluções da mesma linha, podem
                ser confundidos.
              </p>
            </div>
          )}

          <div className="ficha-colunas">
            <section>
              <h3>Stats base</h3>

              <ul className="stats">
                {STATS.map(({ chave, nome }) => (
                  <li key={chave}>
                    <span className="stats-nome">{nome}</span>
                    <span className="stats-barra">
                      <span
                        style={{
                          width: `${(pokemon.stats[chave] / STAT_MAXIMO) * 100}%`,
                        }}
                      />
                    </span>
                    <span className="stats-valor">{pokemon.stats[chave]}</span>
                  </li>
                ))}
              </ul>
            </section>

            <section>
              <h3>Golpes por nível (Red/Blue)</h3>

              <ul className="golpes">
                {pokemon.golpes.map((golpe, posicao) => (
                  <li key={`${golpe.nome}-${posicao}`}>
                    <span className="golpes-nivel">Nv {golpe.nivel}</span>
                    {golpe.nome}
                  </li>
                ))}
              </ul>
            </section>
          </div>

          {(pokemon.evolui_de.length > 0 || pokemon.evolui_para.length > 0) && (
            <section className="evolucao">
              <h3>Evolução</h3>

              <div className="evolucao-linha">
                {pokemon.evolui_de.map((referencia) => (
                  <span className="evolucao-item" key={referencia.numero}>
                    <img src={`${API_URL}/sprites/${referencia.numero}.png`} alt="" />
                    {referencia.nome}
                  </span>
                ))}

                {pokemon.evolui_de.length > 0 && (
                  <span className="evolucao-seta">→</span>
                )}

                <span className="evolucao-item evolucao-item--atual">
                  <img src={`${API_URL}${pokemon.sprite}`} alt="" />
                  {pokemon.nome}
                </span>

                {pokemon.evolui_para.length > 0 && (
                  <span className="evolucao-seta">→</span>
                )}

                {pokemon.evolui_para.map((referencia) => (
                  <span className="evolucao-item" key={referencia.numero}>
                    <img src={`${API_URL}/sprites/${referencia.numero}.png`} alt="" />
                    {referencia.nome}
                  </span>
                ))}
              </div>
            </section>
          )}

          {temConfianca && (
            <section>
              <h3>Outras possibilidades do modelo</h3>

              <p className="ficha-legenda">
                Pokémon parecidos podem ser confundidos; estas foram as
                outras espécies consideradas.
              </p>

              <ol className="top5">
                {resultado.top_5.map((previsao) => (
                  <li key={previsao.numero}>
                    {previsao.nome}
                    <span>{previsao.confianca.toFixed(1)}%</span>
                  </li>
                ))}
              </ol>
            </section>
          )}
        </div>
      </div>
    </div>
  )
}


function formatarTamanhoArquivo(bytes: number) {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`
  }

  return `${(bytes / (1024 * 1024)).toFixed(1).replace(".", ",")} MB`
}


// Aviso para quando a imagem não tem nenhum Pokémon de Kanto (quando há
// Pokémon, o resumo fica no cabeçalho do card de resultados).
function NenhumEncontrado() {
  return (
    <div className="resumo-busca">
      <span className="resumo-numero">0</span>

      <div className="resumo-texto">
        <strong>Nenhum Pokémon encontrado</strong>
        <span>
          Nenhuma das 151 espécies de Kanto aparece nesta imagem. Esta
          Pokédex só reconhece a Primeira Geração: Pokémon de outras
          gerações não são identificados.
        </span>
      </div>
    </div>
  )
}


function App() {
  const [idPokemon, setIdPokemon] = useState("")

  // Incrementa/decrementa o número do Pokémon digitado, sempre
  // mantendo o valor dentro do intervalo válido (1 a 151).
  function ajustarIdPokemon(delta: number) {
    const atual = Number(idPokemon) || 0
    const novo = Math.min(151, Math.max(1, atual + delta))
    setIdPokemon(String(novo))
  }

  const [resultados, setResultados] = useState<ResultadoPokemon[]>([])

  const [arquivo, setArquivo] = useState<File | null>(null)
  const [preview, setPreview] = useState("")

  // Tamanho da imagem processada, para posicionar as caixas.
  const [dimensoes, setDimensoes] = useState<{
    largura: number
    altura: number
  } | null>(null)

  const [erro, setErro] = useState("")

  // Quantos Pokémon a última identificação encontrou (null = nenhuma
  // identificação para mostrar).
  const [encontrados, setEncontrados] = useState<number | null>(null)

  // Busca por imagem ou por número, que dividem a área do topo.
  const [modo, setModo] = useState<"imagem" | "numero">("imagem")

  // Tamanho real da imagem escolhida, mostrado abaixo da prévia.
  const [tamanhoImagem, setTamanhoImagem] = useState<{
    largura: number
    altura: number
  } | null>(null)

  const [carregando, setCarregando] = useState(false)

  // Pokémon selecionado no painel de resultados.
  const [itemAberto, setItemAberto] = useState<number | null>(0)

  // Conversa com o LLM da imagem (ou busca manual) atual.
  const [conversaId, setConversaId] = useState<string | null>(null)

  // Depois de uma identificação (ou busca por número), a página rola até
  // os cards do Pokémon e do chat. O contador muda a cada busca concluída.
  const resultadosRef = useRef<HTMLDivElement>(null)
  const [buscasConcluidas, setBuscasConcluidas] = useState(0)

  useEffect(() => {
    if (buscasConcluidas > 0) {
      resultadosRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      })
    }
  }, [buscasConcluidas])


  async function buscarPokemon() {
    setErro("")
    setEncontrados(null)
    setResultados([])
    setDimensoes(null)
    setConversaId(null)

    const id_pokemon = Number(idPokemon)

    if (
      !Number.isInteger(id_pokemon) ||
      id_pokemon < 1 ||
      id_pokemon > 151
    ) {
      setErro(
        "Digite um número inteiro entre 1 e 151."
      )

      return
    }

    try {
      const resposta = await fetch(
        `${API_URL}/pokemon/${id_pokemon}`
      )

      if (!resposta.ok) {
        setErro(
          "Pokémon não encontrado."
        )

        return
      }

      const pokemon: Pokemon = await resposta.json()

      const resultadoManual: ResultadoPokemon = {
        pokemon: pokemon,
        confianca: 100,
        confianca_detector: null,
        box: [],
        fallback: true,
        top_5: [],
      }

      setResultados([
        resultadoManual
      ])

      setItemAberto(0)
      setBuscasConcluidas((total) => total + 1)

      // A busca manual também ganha uma conversa com o LLM.
      const retornoConversa = await fetch(
        `${API_URL}/conversas`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ numeros: [pokemon.numero] }),
        }
      )

      if (retornoConversa.ok) {
        const conversa = await retornoConversa.json()
        setConversaId(conversa.conversa_id)
      }
    } catch (erro) {
      console.error(erro)

      setErro(
        "Não foi possível conectar ao backend."
      )
    }
  }


  function selecionarArquivo(
    evento: ChangeEvent<HTMLInputElement>
  ) {
    const arquivoSelecionado =
      evento.target.files?.[0]

    if (!arquivoSelecionado) {
      setArquivo(null)
      setPreview("")
      return
    }

    if (preview) {
      URL.revokeObjectURL(
        preview
      )
    }

    const urlPreview = URL.createObjectURL(
      arquivoSelecionado
    )

    setArquivo(
      arquivoSelecionado
    )

    setPreview(
      urlPreview
    )

    setTamanhoImagem(null)

    setResultados([])
    setDimensoes(null)
    setConversaId(null)
    setErro("")
    setEncontrados(null)
  }


  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(
          preview
        )
      }
    }
  }, [preview])


  async function identificarPokemon() {
    if (!arquivo) {
      setErro(
        "Selecione uma imagem primeiro."
      )

      return
    }

    setErro("")
    setEncontrados(null)
    setResultados([])
    setDimensoes(null)
    setConversaId(null)
    setCarregando(true)

    const formulario = new FormData()

    formulario.append(
      "arquivo",
      arquivo
    )

    try {
      const resposta = await fetch(
        `${API_URL}/pokemon/identificar`,
        {
          method: "POST",
          body: formulario,
        }
      )

      if (!resposta.ok) {
        let mensagemErro = "Falha em identificar a imagem."

        try {
          const dadosErro = await resposta.json()
          mensagemErro =
            typeof dadosErro.detail === "string"
              ? dadosErro.detail
              : mensagemErro
        } catch {
          // A API pode retornar uma resposta que não seja JSON.
        }

        setErro(mensagemErro)
        return
      }

      const dados: RespostaIdentificacao =
        await resposta.json()

      setDimensoes({
        largura: dados.largura,
        altura: dados.altura,
      })

      setEncontrados(dados.quantidade)

      if (dados.quantidade === 0) {
        setResultados([])

        return
      }

      setResultados(
        dados.pokemons
      )

      setConversaId(dados.conversa_id)

      setItemAberto(0)
      setBuscasConcluidas((total) => total + 1)
    } catch (erro) {
      console.error(erro)

      setErro(
        "Falha ao conectar no backend."
      )
    } finally {
      setCarregando(false)
    }
  }


  const numeroDigitado = Number(idPokemon)

  const numeroValido =
    Number.isInteger(numeroDigitado) &&
    numeroDigitado >= 1 &&
    numeroDigitado <= 151


  return (
    <main className="pagina">
      <header className="topo">
        <PadraoDecorativo />

        <div className="topo-conteudo">
          <div className="topo-visual-coluna">
            <div className="topo-visual">
              {modo === "imagem" && preview && (
                <>
                  <img
                    className="preview"
                    src={preview}
                    alt="Imagem enviada para identificação"
                    onLoad={(evento) =>
                      setTamanhoImagem({
                        largura: evento.currentTarget.naturalWidth,
                        altura: evento.currentTarget.naturalHeight,
                      })
                    }
                  />

                  {dimensoes && resultados.map((resultado, indice) => {
                    if (!temCaixa(resultado)) {
                      return null
                    }

                    const [x1, y1, x2, y2] = resultado.box

                    return (
                      <div
                        key={`caixa-${indice}`}
                        className="caixa-deteccao"
                        style={{
                          left: `${(x1 / dimensoes.largura) * 100}%`,
                          top: `${(y1 / dimensoes.altura) * 100}%`,
                          width: `${((x2 - x1) / dimensoes.largura) * 100}%`,
                          height: `${((y2 - y1) / dimensoes.altura) * 100}%`,
                          animationDelay: `${indice * 120}ms`,
                        }}
                      >
                        <span>
                          {indice + 1}. {resultado.pokemon.nome}
                        </span>
                      </div>
                    )
                  })}
                </>
              )}

              {/* Linha que percorre a prévia enquanto a imagem é analisada. */}
              {modo === "imagem" && preview && carregando && (
                <div className="scanner" aria-hidden="true" />
              )}

              {modo === "imagem" && !preview && (
                <p className="topo-visual-vazio">
                  Nenhuma imagem selecionada
                </p>
              )}

              {/* Na busca por número, o quadro mostra o sprite do número
                  digitado enquanto a pessoa escolhe. */}
              {modo === "numero" && (
                numeroValido ? (
                  <img
                    className="sprite-previa"
                    src={`${API_URL}/sprites/${numeroDigitado}.png`}
                    alt={`Sprite do Pokémon número ${numeroDigitado}`}
                  />
                ) : (
                  <p className="topo-visual-vazio">
                    Digite um número de 1 a 151
                  </p>
                )
              )}
            </div>

            {modo === "imagem" && arquivo && (
              <p className="arquivo-info" title={arquivo.name}>
                <span className="arquivo-nome">{arquivo.name}</span>
                <span>
                  {tamanhoImagem && `${tamanhoImagem.largura} × ${tamanhoImagem.altura} px · `}
                  {formatarTamanhoArquivo(arquivo.size)}
                </span>
              </p>
            )}

            {modo === "numero" && numeroValido && (
              <p className="arquivo-info">
                <span className="arquivo-nome">Nº {numeroDigitado}</span>
              </p>
            )}
          </div>

          <div className="topo-texto">
            <h1>
              Pokédex Gen 1.
            </h1>

            <p className="descricao-topo">
              Envie uma foto e eu identifico quais das 151 espécies de Kanto
              aparecem nela, usando um detector e um classificador treinados
              localmente, com as fichas de Pokémon Red e Blue. Depois é só
              perguntar o que quiser sobre cada um.
            </p>

            <div className="modos" role="tablist" aria-label="Tipo de busca">
              <button
                role="tab"
                aria-selected={modo === "imagem"}
                className={"modo" + (modo === "imagem" ? " modo--ativo" : "")}
                onClick={() => setModo("imagem")}
              >
                Por imagem
              </button>

              <button
                role="tab"
                aria-selected={modo === "numero"}
                className={"modo" + (modo === "numero" ? " modo--ativo" : "")}
                onClick={() => setModo("numero")}
              >
                Por número
              </button>
            </div>

            {modo === "imagem" ? (
              <div className="acao-upload">
                <label className="botao-selecionar-imagem" htmlFor="entrada-arquivo">
                  {arquivo ? "Trocar imagem" : "Escolher imagem"}
                </label>

                <input
                  id="entrada-arquivo"
                  className="entrada-arquivo-oculta"
                  type="file"
                  accept="image/*"
                  onChange={selecionarArquivo}
                />

                <button
                  className="botao-primario"
                  onClick={identificarPokemon}
                  disabled={carregando || !arquivo}
                >
                  {
                    carregando
                      ? "Identificando..."
                      : "Identificar Pokémon"
                  }
                </button>

                <p className="nota-privacidade">
                  <Icone nome="cadeado" />
                  A imagem é analisada só neste computador e descartada em
                  seguida. Nada é gravado nem enviado para a internet.
                </p>
              </div>
            ) : (
              <div className="acao-upload">
                <div className="campo-numero">
                  <input
                    id="pokemon-id"
                    type="number"
                    min="1"
                    max="151"
                    placeholder="25"
                    aria-label="Número do Pokémon"
                    value={idPokemon}
                    onChange={(evento) =>
                      setIdPokemon(
                        evento.target.value
                      )
                    }
                    onKeyDown={(evento) => {
                      if (evento.key === "Enter") {
                        buscarPokemon()
                      }
                    }}
                  />

                  <div className="campo-numero-setas">
                    <button
                      type="button"
                      aria-label="Aumentar número"
                      onClick={() => ajustarIdPokemon(1)}
                    >
                      <svg viewBox="0 0 10 6" width="10" height="6">
                        <path d="M1 5L5 1L9 5" />
                      </svg>
                    </button>

                    <button
                      type="button"
                      aria-label="Diminuir número"
                      onClick={() => ajustarIdPokemon(-1)}
                    >
                      <svg viewBox="0 0 10 6" width="10" height="6">
                        <path d="M1 1L5 5L9 1" />
                      </svg>
                    </button>
                  </div>
                </div>

                <button
                  className="botao-primario"
                  onClick={buscarPokemon}
                >
                  Ver ficha
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      <section className="secao-escura">
        {erro && (
          <p className="aviso aviso--erro">
            {erro}
          </p>
        )}

        {encontrados === 0 && <NenhumEncontrado />}

        {/* Dois cards lado a lado: o Pokémon à esquerda e o chat à
            direita. A chave reinicia o chat quando chega uma imagem nova. */}
        {resultados.length > 0 && (
          <div className="resultados-layout" ref={resultadosRef}>
            <PainelResultados
              resultados={resultados}
              selecionado={itemAberto ?? 0}
              selecionar={setItemAberto}
              encontrados={encontrados}
            />

            {conversaId && (
              <ChatImagem
                key={conversaId}
                conversaId={conversaId}
                nomes={resultados.map((item) => item.pokemon.nome)}
              />
            )}
          </div>
        )}
      </section>

      <AvisosEticos />

      <footer className="rodape">
        <p className="rodape-texto">
          Trabalho da disciplina de Sistemas de Informação Inteligentes —
          desenvolvido por João Brum e Kauane Santana.
        </p>
      </footer>
    </main>
  )
}


export default App
