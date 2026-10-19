import { useEffect, useRef, useState, type ChangeEvent } from "react"
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
  largura: number
  altura: number
  pokemons: ResultadoPokemon[]
}


function listarNomes(referencias: Referencia[]) {
  return referencias.map((referencia) => referencia.nome).join(", ")
}


function formatarMedida(valor: number | null, unidade: string) {
  return valor === null ? "Não informado" : `${valor} ${unidade}`
}


const API_URL = "http://127.0.0.1:8000"

// Cores que se repetem nos itens do acordeon de resultados.
const CORES_ACORDEON = [
  "var(--roxo-600)",
  "var(--rosa)",
  "var(--indigo)",
  "var(--roxo-700)",
]


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


type MensagemChat = {
  autor: "usuario" | "ia" | "erro"
  texto: string
}


// Campo de pergunta ao LLM local sobre um Pokémon (rota
// POST /pokemon/{numero}/perguntar; ver LLM.md), em formato de chat.
function PerguntaPokemon({ numero }: { numero: number }) {
  const [pergunta, setPergunta] = useState("")
  const [mensagens, setMensagens] = useState<MensagemChat[]>([])
  const [carregando, setCarregando] = useState(false)

  // Rola tanto o histórico do chat quanto, se preciso, a própria página
  // até a última mensagem ficar visível (block: "nearest" só mexe no
  // que realmente está fora da vista, sem forçar um salto brusco).
  const fimChatRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fimChatRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    })
  }, [mensagens, carregando])

  async function enviarPergunta() {
    const texto = pergunta.trim()

    if (!texto) {
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
        `${API_URL}/pokemon/${numero}/perguntar`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pergunta: texto }),
        }
      )

      const dados = await retorno.json()

      if (!retorno.ok) {
        setMensagens((atuais) => [
          ...atuais,
          {
            autor: "erro",
            texto:
              typeof dados.detail === "string"
                ? dados.detail
                : "Não foi possível obter uma resposta.",
          },
        ])
        return
      }

      setMensagens((atuais) => [
        ...atuais,
        { autor: "ia", texto: dados.resposta },
      ])
    } catch (erroConexao) {
      console.error(erroConexao)

      setMensagens((atuais) => [
        ...atuais,
        { autor: "erro", texto: "Falha ao conectar no backend." },
      ])
    } finally {
      setCarregando(false)
    }
  }

  return (
    <div className="pergunta-pokemon">
      <h3>
        Pergunte sobre este Pokémon
      </h3>

      {mensagens.length > 0 && (
        <div className="chat-mensagens">
          {mensagens.map((mensagem, indice) => (
            <div
              key={indice}
              className={`chat-bolha chat-bolha--${mensagem.autor}`}
            >
              {mensagem.texto}
            </div>
          ))}

          {carregando && (
            <div className="chat-bolha chat-bolha--ia chat-bolha--digitando">
              <span />
              <span />
              <span />
            </div>
          )}

          <div ref={fimChatRef} />
        </div>
      )}

      <div className="pergunta-pokemon-linha">
        <input
          type="text"
          value={pergunta}
          maxLength={500}
          placeholder="Ex.: Quais são os pontos fortes dele?"
          onChange={(evento) => setPergunta(evento.target.value)}
          onKeyDown={(evento) => {
            if (evento.key === "Enter") {
              enviarPergunta()
            }
          }}
        />

        <button
          className="botao-primario"
          onClick={enviarPergunta}
          disabled={carregando}
        >
          {carregando ? "Pensando..." : "Perguntar"}
        </button>
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
  const [mensagem, setMensagem] = useState("")

  const [carregando, setCarregando] = useState(false)

  // Item aberto no acordeon de resultados (null = todos fechados).
  const [itemAberto, setItemAberto] = useState<number | null>(0)


  async function buscarPokemon() {
    setErro("")
    setMensagem("")
    setResultados([])
    setDimensoes(null)

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

    setResultados([])
    setDimensoes(null)
    setErro("")
    setMensagem("")
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
    setMensagem("")
    setResultados([])
    setDimensoes(null)
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

      if (dados.quantidade === 0) {
        setMensagem(
          "Nenhum Pokémon da Primeira Geração encontrado."
        )

        setResultados([])

        return
      }

      setResultados(
        dados.pokemons
      )

      setItemAberto(0)

      if (dados.quantidade === 1) {
        setMensagem(
          "1 Pokémon encontrado."
        )
      } else {
        setMensagem(
          `${dados.quantidade} Pokémon encontrados.`
        )
      }
    } catch (erro) {
      console.error(erro)

      setErro(
        "Falha ao conectar no backend."
      )
    } finally {
      setCarregando(false)
    }
  }


  return (
    <main className="pagina">
      <header className="topo">
        <PadraoDecorativo />

        <div className="topo-conteudo">
          <div className="topo-visual-coluna">
            <div className="topo-visual">
              {preview ? (
                <>
                  <img
                    className="preview"
                    src={preview}
                    alt="Imagem enviada para identificação"
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
                        }}
                      >
                        <span>
                          {indice + 1}. {resultado.pokemon.nome}
                        </span>
                      </div>
                    )
                  })}
                </>
              ) : (
                <p className="topo-visual-vazio">
                  Nenhuma imagem selecionada
                </p>
              )}
            </div>

            <label className="botao-selecionar-imagem" htmlFor="entrada-arquivo">
              {arquivo ? arquivo.name : "Escolher imagem"}
            </label>

            <input
              id="entrada-arquivo"
              className="entrada-arquivo-oculta"
              type="file"
              accept="image/*"
              onChange={selecionarArquivo}
            />
          </div>

          <div className="topo-texto">
            <p className="tag-mono">
              // identificar pokémon
            </p>

            <h1>
              Pokédex Gen 1.
            </h1>

            <p className="descricao-topo">
              Envie uma foto e eu identifico quais Pokémon de Kanto aparecem
              nela, usando um detector e um classificador treinados
              localmente. Depois é só perguntar o que quiser sobre cada um.
            </p>

            <div className="chips">
              <span className="chip">151 espécies</span>
              <span className="chip">Kanto</span>
              <span className="chip">Red / Blue</span>
            </div>

            <div className="acao-upload">
              <button
                className="botao-primario"
                onClick={identificarPokemon}
                disabled={carregando}
              >
                {
                  carregando
                    ? "Identificando..."
                    : "Identificar Pokémon"
                }
              </button>
            </div>
          </div>
        </div>
      </header>

      <section className="secao-escura">
        <div className="coluna-esquerda">
          <p className="tag-mono tag-mono--lima">
            // buscar à vontade
          </p>

          <h2 className="titulo-grande">
            Busca manual
          </h2>

          <div className="rabisco">
            <svg width="72" height="18" viewBox="0 0 72 18" fill="none">
              <path
                d="M2 14c6 0 6-10 12-10s6 10 12 10 6-10 12-10 6 10 12 10 6-10 12-10 6 10 12 10"
                stroke="var(--lima)"
                strokeWidth="3"
                strokeLinecap="round"
              />
            </svg>
          </div>

          <p className="texto-claro-muted">
            Não tem uma imagem à mão? Digite o número do Pokémon (1 a 151) e
            veja a ficha completa.
          </p>

          <div className="busca-manual">
            <label htmlFor="pokemon-id">
              Número do Pokémon
            </label>

            <div className="campo-numero">
              <input
                id="pokemon-id"
                type="number"
                min="1"
                max="151"
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
              className="link-mono"
              onClick={buscarPokemon}
            >
              &gt; buscar
            </button>
          </div>

          <div className="blocos-decorativos">
            <span />
            <span />
          </div>
        </div>

        <div className="coluna-direita">
          {erro && (
            <p className="aviso aviso--erro">
              {erro}
            </p>
          )}

          {mensagem && (
            <p className="aviso aviso--info">
              {mensagem}
            </p>
          )}

          {resultados.length > 0 && (
            <div className="painel-resultados">
              <PadraoDecorativo />

              <div className="painel-resultados-corpo">
                {
                  resultados.map(
                    (resultado, indice) => {
                      const pokemon =
                        resultado.pokemon

                      const aberto = itemAberto === indice

                      const cor =
                        CORES_ACORDEON[indice % CORES_ACORDEON.length]

                      const temConfianca = resultado.top_5.length > 0

                      return (
                        <div
                          className="resultado-card"
                          key={`${pokemon.numero}-${indice}`}
                        >
                          <button
                            className="resultado-cabecalho"
                            onClick={() =>
                              setItemAberto(aberto ? null : indice)
                            }
                          >
                            <div className="resultado-avatar">
                              <span
                                className="resultado-numero"
                                style={{ background: cor }}
                              >
                                {String(indice + 1).padStart(2, "0")}
                              </span>

                              <img
                                src={`${API_URL}${pokemon.sprite}`}
                                alt={`Sprite de ${pokemon.nome}`}
                              />
                            </div>

                            <div className="resultado-texto">
                              <p className="tag-mono">
                                {`// ${
                                  temCaixa(resultado)
                                    ? `pokémon detectado ${indice + 1}`
                                    : "identificação"
                                }`}
                              </p>

                              <h2>
                                #{pokemon.numero} {pokemon.nome}
                              </h2>

                              <p className="resultado-descricao">
                                {pokemon.descricao}
                              </p>

                              <div className="chips">
                                <span className="chip">
                                  {pokemon.classificacao}
                                </span>

                                <span className="chip">
                                  {pokemon.tipos.join(" / ")}
                                </span>

                                <span className="chip">
                                  {formatarMedida(pokemon.altura_m, "m")}
                                  {" / "}
                                  {formatarMedida(pokemon.peso_kg, "kg")}
                                </span>

                                {temConfianca && (
                                  <span className="chip">
                                    {resultado.confianca.toFixed(1)}%
                                    {" "}confiança
                                  </span>
                                )}

                                {pokemon.lendario && (
                                  <span className="chip chip--lendario">
                                    Lendário
                                  </span>
                                )}
                              </div>
                            </div>

                            <span className="icone">
                              {aberto ? "–" : "+"}
                            </span>
                          </button>

                          {aberto && (
                            <div className="resultado-corpo">
                              {
                                resultado.confianca_detector !== null &&
                                (
                                  <p>
                                    <strong>
                                      Confiança da detecção:
                                    </strong>{" "}
                                    {
                                      resultado
                                        .confianca_detector
                                        .toFixed(2)
                                    }%
                                  </p>
                                )
                              }

                              {resultado.fallback && temConfianca && (
                                <p>
                                  Identificação feita usando a imagem
                                  completa.
                                </p>
                              )}

                              <h3>
                                Stats
                              </h3>

                              <ul>
                                <li>
                                  HP: {pokemon.stats.hp}
                                </li>

                                <li>
                                  Ataque: {pokemon.stats.ataque}
                                </li>

                                <li>
                                  Defesa: {pokemon.stats.defesa}
                                </li>

                                <li>
                                  Especial: {pokemon.stats.especial}
                                </li>

                                <li>
                                  Velocidade: {pokemon.stats.velocidade}
                                </li>
                              </ul>

                              <h3>
                                Golpes (Red/Blue)
                              </h3>

                              <ul>
                                {pokemon.golpes.map((golpe, posicao) => (
                                  <li key={`${golpe.nome}-${posicao}`}>
                                    Nível {golpe.nivel}: {golpe.nome}
                                  </li>
                                ))}
                              </ul>

                              {/* Cada linha só aparece quando existe a evolução;
                                  sem nenhuma das duas, a seção inteira some. */}
                              {(pokemon.evolui_de.length > 0 ||
                                pokemon.evolui_para.length > 0) && (
                                <>
                                  <h3>
                                    Evolução
                                  </h3>

                                  {pokemon.evolui_de.length > 0 && (
                                    <p>
                                      <strong>
                                        Evolui de:
                                      </strong>{" "}
                                      {listarNomes(pokemon.evolui_de)}
                                    </p>
                                  )}

                                  {pokemon.evolui_para.length > 0 && (
                                    <p>
                                      <strong>
                                        Evolui para:
                                      </strong>{" "}
                                      {listarNomes(pokemon.evolui_para)}
                                    </p>
                                  )}
                                </>
                              )}

                              {temConfianca && (
                                <>
                                  <h3>
                                    Top 5 do modelo
                                  </h3>

                                  <ol>
                                    {
                                      resultado.top_5.map(
                                        (previsao) => (
                                          <li
                                            key={previsao.numero}
                                          >
                                            {previsao.nome}:{" "}
                                            {
                                              previsao
                                                .confianca
                                                .toFixed(2)
                                            }%
                                          </li>
                                        )
                                      )
                                    }
                                  </ol>
                                </>
                              )}

                              <PerguntaPokemon numero={pokemon.numero} />
                            </div>
                          )}
                        </div>
                      )
                    }
                  )
                }
              </div>
            </div>
          )}
        </div>
      </section>

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
