import { useEffect, useState, type ChangeEvent } from "react"


type Pokemon = {
  pokedex_number: string
  name: string
  classification: string
  type1: string
  type2: string
  height_m: string
  weight_kg: string
  hp: string
  attack: string
  defense: string
  sp_attack: string
  sp_defense: string
  speed: string
  is_legendary: string
  evolves_from: string
  evolves_to: string
  description: string
  level_up_moves_red_blue: string
  level_up_moves_yellow: string
  level_up_moves_red_green_japan: string
  level_up_moves_blue_japan: string
}


type Previsao = {
  pokemon: string
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
  pokemons: ResultadoPokemon[]
}


const API_URL = "http://127.0.0.1:8000"


function App() {
  const [idPokemon, setIdPokemon] = useState("")

  const [resultados, setResultados] = useState<ResultadoPokemon[]>([])

  const [arquivo, setArquivo] = useState<File | null>(null)
  const [preview, setPreview] = useState("")

  const [erro, setErro] = useState("")
  const [mensagem, setMensagem] = useState("")

  const [carregando, setCarregando] = useState(false)


  async function buscarPokemon() {
    setErro("")
    setMensagem("")
    setResultados([])

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

          <img
            src={preview}
            alt="Imagem enviada para identificação"
            style={{
              maxWidth: "400px",
              maxHeight: "400px",
              objectFit: "contain",
            }}
          />
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
                    key={`${pokemon.pokedex_number}-${indice}`}
                  >
                    <hr />

                    <h2>
                      #{pokemon.pokedex_number}{" "}
                      {pokemon.name}
                    </h2>

                    <img
                      src={`${API_URL}/sprites/${pokemon.pokedex_number}.png`}
                      alt={`Sprite de ${pokemon.name}`}
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
                      {pokemon.classification}
                    </p>


                    <p>
                      <strong>
                        Tipo:
                      </strong>{" "}
                      {pokemon.type1}

                      {
                        pokemon.type2 &&
                        ` / ${pokemon.type2}`
                      }
                    </p>


                    <p>
                      <strong>
                        Altura:
                      </strong>{" "}
                      {pokemon.height_m} m
                    </p>


                    <p>
                      <strong>
                        Peso:
                      </strong>{" "}
                      {pokemon.weight_kg} kg
                    </p>


                    <h3>
                      Stats
                    </h3>


                    <ul>
                      <li>
                        HP: {pokemon.hp}
                      </li>

                      <li>
                        Ataque: {pokemon.attack}
                      </li>

                      <li>
                        Defesa: {pokemon.defense}
                      </li>

                      <li>
                        Ataque Especial: {pokemon.sp_attack}
                      </li>

                      <li>
                        Defesa Especial: {pokemon.sp_defense}
                      </li>

                      <li>
                        Velocidade: {pokemon.speed}
                      </li>
                    </ul>


                    <h3>
                      Descrição
                    </h3>

                    <p>
                      {pokemon.description}
                    </p>


                    <h3>
                      Evolução
                    </h3>

                    <p>
                      <strong>
                        Evolui de:
                      </strong>{" "}
                      {
                        pokemon.evolves_from ||
                        "Nenhum"
                      }
                    </p>

                    <p>
                      <strong>
                        Evolui para:
                      </strong>{" "}
                      {
                        pokemon.evolves_to ||
                        "Nenhum"
                      }
                    </p>


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
                                  key={previsao.pokemon}
                                >
                                  {previsao.pokemon}:{" "}
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