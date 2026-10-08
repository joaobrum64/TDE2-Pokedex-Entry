import csv
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = (
    ROOT_DIR
    / "datasets"
    / "pokemon_gen1.csv"
)

SPRITES_PATH = (
    ROOT_DIR
    / "datasets"
    / "sprites"
)

# Os golpes vêm da versão base da Primeira Geração (Red/Blue).
COLUNA_GOLPES = "level_up_moves_red_blue"

# Tipos em português, usados no texto de contexto do LLM.
TIPOS_PT = {
    "bug": "Inseto",
    "dragon": "Dragão",
    "electric": "Elétrico",
    "fighting": "Lutador",
    "fire": "Fogo",
    "flying": "Voador",
    "ghost": "Fantasma",
    "grass": "Planta",
    "ground": "Terra",
    "ice": "Gelo",
    "normal": "Normal",
    "poison": "Venenoso",
    "psychic": "Psíquico",
    "rock": "Pedra",
    "water": "Água",
}

# Nomes de evolução no CSV que não seguem a grafia oficial.
NOMES_EVOLUCAO = {
    "nidoran-m": "Nidoran♂",
    "nidoran-f": "Nidoran♀",
}


def normalizar_nome(nome):
    # As classes do modelo vêm dos nomes das pastas de treino
    # (ex.: "Farfetchd", "MrMime"), enquanto o CSV usa a grafia
    # oficial ("Farfetch'd", "Mr. Mime"). Os símbolos ♀ e ♂ são
    # mantidos para não confundir os dois Nidoran.
    return (
        nome
        .lower()
        .replace(" ", "")
        .replace(".", "")
        .replace("'", "")
    )


def nome_legivel_golpe(nome):
    # "thunder-shock" -> "Thunder Shock"
    return " ".join(
        parte.capitalize()
        for parte in nome.split("-")
    )


def separar_golpes(texto):
    # "1:thunder-shock|9:thunder-wave" -> lista de {nivel, nome}
    golpes = []

    for item in texto.split("|"):
        if not item:
            continue

        nivel, nome = item.split(":", 1)

        golpes.append(
            {
                "nivel": int(nivel),
                "nome": nome_legivel_golpe(nome),
            }
        )

    return golpes


def numero_ou_nada(texto):
    # Proteção para linhas novas no CSV sem altura ou peso.
    return float(texto) if texto else None


def carregar_linhas():
    with open(
        CSV_PATH,
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as arquivo:
        return list(csv.DictReader(arquivo))


class Pokedex:
    # Fichas dos 151 Pokémon montadas a partir de pokemon_gen1.csv.
    # É também a entrada para quem precisa dos dados de um Pokémon
    # identificado (ex.: a integração com o LLM): obter_ficha("Pikachu")
    # ou obter_ficha(25).

    def __init__(self):
        linhas = carregar_linhas()

        self.numero_por_nome = {
            normalizar_nome(linha["name"]): int(linha["pokedex_number"])
            for linha in linhas
        }

        self.nome_por_numero = {
            int(linha["pokedex_number"]): linha["name"]
            for linha in linhas
        }

        self.fichas = {
            int(linha["pokedex_number"]): self.montar_ficha(linha)
            for linha in linhas
        }

    def referencia(self, nome):
        # Nome escrito de qualquer jeito (classe do modelo ou campo de
        # evolução do CSV) -> {numero, nome oficial}.
        nome = NOMES_EVOLUCAO.get(nome, nome)

        numero = self.numero_por_nome.get(
            normalizar_nome(nome.replace("-", ""))
        )

        if numero is None:
            return None

        return {
            "numero": numero,
            "nome": self.nome_por_numero[numero],
        }

    def lista_de_referencias(self, texto):
        return [
            self.referencia(nome)
            for nome in texto.split("|")
            if nome
        ]

    def montar_ficha(self, linha):
        numero = int(linha["pokedex_number"])

        return {
            "numero": numero,
            "nome": linha["name"],
            "classificacao": linha["classification"],
            "tipos": [
                tipo
                for tipo in (linha["type1"], linha["type2"])
                if tipo
            ],
            "altura_m": numero_ou_nada(linha["height_m"]),
            "peso_kg": numero_ou_nada(linha["weight_kg"]),
            # Na Primeira Geração existe um único stat "Especial";
            # no CSV ele aparece repetido em sp_attack e sp_defense.
            "stats": {
                "hp": int(linha["hp"]),
                "ataque": int(linha["attack"]),
                "defesa": int(linha["defense"]),
                "especial": int(linha["sp_attack"]),
                "velocidade": int(linha["speed"]),
            },
            "lendario": linha["is_legendary"] == "1",
            "descricao": linha["description"],
            "evolui_de": self.lista_de_referencias(linha["evolves_from"]),
            "evolui_para": self.lista_de_referencias(linha["evolves_to"]),
            "golpes": separar_golpes(linha[COLUNA_GOLPES]),
            "sprite": f"/sprites/{numero}.png",
        }

    def obter_ficha(self, nome_ou_numero):
        if isinstance(nome_ou_numero, int):
            return self.fichas.get(nome_ou_numero)

        referencia = self.referencia(nome_ou_numero)

        if referencia is None:
            return None

        return self.fichas[referencia["numero"]]


def montar_contexto(ficha):
    # Texto em português com todos os dados da ficha, para ser
    # colocado no prompt do LLM (RAG). O LLM deve responder apenas
    # com base neste texto. A descrição da Pokédex e os nomes dos
    # golpes estão em inglês, como no jogo original.
    def nomes(referencias):
        return ", ".join(
            f"{referencia['nome']} (nº {referencia['numero']})"
            for referencia in referencias
        )

    stats = ficha["stats"]

    linhas = [
        f"Pokémon: {ficha['nome']} (nº {ficha['numero']} da Pokédex, "
        "Primeira Geração)",
        f"Classificação: {ficha['classificacao']}",
        "Tipo: " + " / ".join(
            TIPOS_PT.get(tipo, tipo) for tipo in ficha["tipos"]
        ),
        f"Altura: {ficha['altura_m']} m",
        f"Peso: {ficha['peso_kg']} kg",
        f"Lendário: {'sim' if ficha['lendario'] else 'não'}",
        (
            "Stats base: "
            f"HP {stats['hp']}, Ataque {stats['ataque']}, "
            f"Defesa {stats['defesa']}, Especial {stats['especial']}, "
            f"Velocidade {stats['velocidade']} "
            "(na Primeira Geração existe um único stat Especial)"
        ),
        "Evolui de: " + (nomes(ficha["evolui_de"]) or "não evolui de nenhum Pokémon"),
        "Evolui para: " + (nomes(ficha["evolui_para"]) or "não evolui para nenhum Pokémon"),
        "Golpes aprendidos por nível (Pokémon Red/Blue): " + ", ".join(
            f"{golpe['nome']} (nível {golpe['nivel']})"
            for golpe in ficha["golpes"]
        ),
        f"Descrição da Pokédex (em inglês): {ficha['descricao']}",
    ]

    return "\n".join(linhas)
