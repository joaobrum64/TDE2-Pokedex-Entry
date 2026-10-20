# Tabela de efetividade de tipos da Primeira Geração (Red, Green, Blue e
# Yellow), usada no contexto do LLM para que vantagens e fraquezas venham
# de dados, e não da memória do modelo (nos testes, todos os modelos
# erraram "Fogo contra Água" sem esta tabela).
#
# Particularidades da Gen 1 em relação às gerações seguintes: Fantasma
# não afeta Psíquico (imunidade causada por um erro do jogo original),
# Inseto e Venenoso são super efetivos um contra o outro, e Gelo causa
# dano normal em Fogo. Os tipos Fada, Aço e Sombrio não existem.
#
# EFETIVIDADE[atacante][defensor] = multiplicador; o que não aparece é 1.
EFETIVIDADE = {
    "normal": {"rock": 0.5, "ghost": 0},
    "fire": {
        "fire": 0.5, "water": 0.5, "grass": 2, "ice": 2,
        "bug": 2, "rock": 0.5, "dragon": 0.5,
    },
    "water": {
        "fire": 2, "water": 0.5, "grass": 0.5, "ground": 2,
        "rock": 2, "dragon": 0.5,
    },
    "electric": {
        "water": 2, "electric": 0.5, "grass": 0.5, "ground": 0,
        "flying": 2, "dragon": 0.5,
    },
    "grass": {
        "fire": 0.5, "water": 2, "grass": 0.5, "poison": 0.5,
        "ground": 2, "flying": 0.5, "bug": 0.5, "rock": 2, "dragon": 0.5,
    },
    "ice": {
        "water": 0.5, "grass": 2, "ice": 0.5, "ground": 2,
        "flying": 2, "dragon": 2,
    },
    "fighting": {
        "normal": 2, "ice": 2, "poison": 0.5, "flying": 0.5,
        "psychic": 0.5, "bug": 0.5, "rock": 2, "ghost": 0,
    },
    "poison": {
        "grass": 2, "poison": 0.5, "ground": 0.5, "bug": 2,
        "rock": 0.5, "ghost": 0.5,
    },
    "ground": {
        "fire": 2, "electric": 2, "grass": 0.5, "poison": 2,
        "flying": 0, "bug": 0.5, "rock": 2,
    },
    "flying": {
        "electric": 0.5, "grass": 2, "fighting": 2, "bug": 2,
        "rock": 0.5,
    },
    "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5},
    "bug": {
        "fire": 0.5, "grass": 2, "fighting": 0.5, "poison": 2,
        "flying": 0.5, "psychic": 2, "ghost": 0.5,
    },
    "rock": {
        "fire": 2, "ice": 2, "fighting": 0.5, "ground": 0.5,
        "flying": 2, "bug": 2,
    },
    "ghost": {"normal": 0, "psychic": 0, "ghost": 2},
    "dragon": {"dragon": 2},
}

TIPOS = list(EFETIVIDADE)


def multiplicador_defensivo(tipo_atacante, tipos_defensor):
    # Dano recebido por um Pokémon de um ou dois tipos.
    multiplicador = 1

    for tipo in tipos_defensor:
        multiplicador *= EFETIVIDADE[tipo_atacante].get(tipo, 1)

    return multiplicador


def resumo_defensivo(tipos_defensor):
    # {"fraquezas": {tipo: 2 ou 4}, "resistencias": {tipo: 0.5 ou 0.25},
    #  "imunidades": [tipo]}
    fraquezas = {}
    resistencias = {}
    imunidades = []

    for atacante in TIPOS:
        valor = multiplicador_defensivo(atacante, tipos_defensor)

        if valor == 0:
            imunidades.append(atacante)
        elif valor > 1:
            fraquezas[atacante] = valor
        elif valor < 1:
            resistencias[atacante] = valor

    return {
        "fraquezas": fraquezas,
        "resistencias": resistencias,
        "imunidades": imunidades,
    }


def super_efetivo_contra(tipo_atacante):
    # Tipos que recebem dano dobrado de golpes deste tipo.
    return [
        defensor
        for defensor, valor in EFETIVIDADE[tipo_atacante].items()
        if valor > 1
    ]
