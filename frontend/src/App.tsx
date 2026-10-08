import { useEffect, useState, type ChangeEvent } from "react"


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


// Mostra as caixas só para resultados do detector; no fallback a
// caixa é a imagem inteira e não acrescenta informação.
function temCaixa(resultado: ResultadoPokemon) {
  return !resultado.fallback && resultado.box.length === 4
}


// Campo de pergunta ao LLM local sobre um Pokémon (rota
// POST /pokemon/{numero}/perguntar; ver LLM.md).
function PerguntaPokemon({ numero }: { numero: number }) {
  const [pergunta, setPergunta] = useState("")
  const [resposta, setResposta] = useState("")
  const [erro, setErro] = useState("")
  const [carregando, setCarregando] = useState(false)

  async function enviarPergunta() {
    if (!pergunta.trim()) {
      return
    }

    setResposta("")
    setErro("")
    setCarregando(true)

    try {
      const retorno = await fetch(
        `${API_URL}/pokemon/${numero}/perguntar`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pergunta }),
        }
      )

      const dados = await retorno.json()

      if (!retorno.ok) {
        setErro(
          typeof dados.detail === "string"
            ? dados.detail
            : "Não foi possível obter uma resposta."
        )
        return
      }

      setResposta(dados.resposta)
    } catch (erroConexao) {
      console.error(erroConexao)
      setErro("Falha ao conectar no backend.")
    } finally {
      setCarregando(false)
    }
  }

  return (
    <div>
      <h3>
        Pergunte sobre este Pokémon
      </h3>

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
        style={{ width: "320px" }}
      />

      <button
        onClick={enviarPergunta}
        disabled={carregando}
      >
        {carregando ? "Pensando..." : "Perguntar"}
      </button>

      {resposta && <p>{resposta}</p>}

      {erro && <p>{erro}</p>}
    </div>
  )
}


function App() {
  const [idPokemon, setIdPokemon] = useState("")

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
    <main>
      <h1>
        Pokédex Gen 1
      </h1>


      <h2>
        Identificar por imagem
      </h2>


      <div>
        <input
          type="file"
          accept="image/*"
          onChange={selecionarArquivo}
        />

        <button
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


      {preview && (
        <div>
          <h3>
            Imagem enviada
          </h3>

          <div
            style={{
              position: "relative",
              display: "inline-block",
            }}
          >
            <img
              src={preview}
              alt="Imagem enviada para identificação"
              style={{
                display: "block",
                maxWidth: "400px",
                maxHeight: "400px",
              }}
            />

            {dimensoes && resultados.map((resultado, indice) => {
              if (!temCaixa(resultado)) {
                return null
              }

              const [x1, y1, x2, y2] = resultado.box

              return (
                <div
                  key={`caixa-${indice}`}
                  style={{
                    position: "absolute",
                    left: `${(x1 / dimensoes.largura) * 100}%`,
                    top: `${(y1 / dimensoes.altura) * 100}%`,
                    width: `${((x2 - x1) / dimensoes.largura) * 100}%`,
                    height: `${((y2 - y1) / dimensoes.altura) * 100}%`,
                    border: "2px solid red",
                    boxSizing: "border-box",
                  }}
                >
                  <span
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      background: "red",
                      color: "white",
                      fontSize: "12px",
                      padding: "0 4px",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {indice + 1}. {resultado.pokemon.nome}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}


      <hr />


      <h2>
        Busca manual para testes
      </h2>


      <div>
        <label htmlFor="pokemon-id">
          Número do Pokémon:
        </label>

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
        />

        <button
          onClick={buscarPokemon}
        >
          Buscar
        </button>
      </div>


      {erro && (
        <p>
          {erro}
        </p>
      )}


      {mensagem && (
        <p>
          <strong>
            {mensagem}
          </strong>
        </p>
      )}


      {resultados.length > 0 && (
        <div>
          {
            resultados.map(
              (resultado, indice) => {
                const pokemon =
                  resultado.pokemon

                return (
                  <section
                    key={`${pokemon.numero}-${indice}`}
                  >
                    <hr />

                    <h2>
                      {temCaixa(resultado) && `${indice + 1}. `}
                      #{pokemon.numero}{" "}
                      {pokemon.nome}
                    </h2>

                    <img
                      src={`${API_URL}${pokemon.sprite}`}
                      alt={`Sprite de ${pokemon.nome}`}
                      width={120}
                      height={120}
                      style={{
                        imageRendering: "pixelated",
                        objectFit: "contain",
                      }}
                    />


                    {resultado.top_5.length > 0 && (
                      <p>
                        <strong>
                          Confiança da identificação:
                        </strong>{" "}
                        {resultado.confianca.toFixed(2)}%
                      </p>
                    )}


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


                    {resultado.fallback && resultado.top_5.length > 0 && (
                      <p>
                        Identificação feita usando a imagem completa.
                      </p>
                    )}


                    <p>
                      <strong>
                        Classificação:
                      </strong>{" "}
                      {pokemon.classificacao}
                    </p>


                    <p>
                      <strong>
                        Tipo:
                      </strong>{" "}
                      {pokemon.tipos.join(" / ")}
                    </p>


                    <p>
                      <strong>
                        Altura:
                      </strong>{" "}
                      {formatarMedida(pokemon.altura_m, "m")}
                    </p>


                    <p>
                      <strong>
                        Peso:
                      </strong>{" "}
                      {formatarMedida(pokemon.peso_kg, "kg")}
                    </p>


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


                    <h3>
                      Descrição
                    </h3>

                    <p>
                      {pokemon.descricao}
                    </p>


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


                    {resultado.top_5.length > 0 && (
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
                  </section>
                )
              }
            )
          }
        </div>
      )}
    </main>
  )
}


export default App