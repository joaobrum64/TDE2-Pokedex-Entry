import sys
import time
from pathlib import Path

from backend import llm
from backend.pokedex import Pokedex


# Roda um conjunto fixo de perguntas no LLM, com o mesmo prompt e o
# mesmo contexto da API, e salva as respostas para comparar modelos ou
# mudanças no prompt.
#
# Uso (com o Ollama aberto):
#   python -m backend.testar_llm                      # modelo de llm.py
#   python -m backend.testar_llm gemma3:4b qwen2.5:7b # outros modelos
#   python -m backend.testar_llm --cpu llama3.2:3b    # força CPU
#   python -m backend.testar_llm --cpu --rapido       # só 3 perguntas
#
# --cpu ignora a GPU, para medir o tempo como numa máquina sem placa de
# vídeo. As respostas vão para saidas/teste_llm_<modelo>.md.

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_DIR / "saidas"

# (Pokémon detectados na imagem, perguntas em sequência na mesma
# conversa, o que se espera da resposta)
CASOS = [
    ([25], ["O que você sabe sobre ele?"],
     "Estilo de Pokédex; 0,4 m, 6,0 kg, Elétrico, evolui para Raichu."),
    ([25], ["Quais golpes ele aprende e em que nível?"],
     "Os 7 golpes de Red/Blue com níveis certos; pode citar que no Yellow é diferente."),
    ([25], ["Qual a altura do Blastoise?"],
     "1,6 m, comparando com os 0,4 m do Pikachu."),
    ([133], ["Para quem ele evolui?"],
     "Vaporeon, Jolteon e Flareon."),
    ([150], ["Qual o Ataque Especial dele?"],
     "Explica o stat Especial único da Gen 1: 154."),
    ([25], ["Em que rota eu encontro ele?"],
     "Não está nas fichas: diz que não consta ou responde em 'Registro externo' (Floresta de Viridian)."),
    ([1, 4, 7], ["Qual dos três é o mais rápido?"],
     "Charmander (Velocidade 65) > Bulbasaur (45) > Squirtle (43)."),
    ([6], ["Ele é bom contra um Blastoise?"],
     "Fogo é fraco contra Água, em 'Registro externo'."),
    ([25], ["Fale sobre o Lucario."],
     "Não consta nos registros de Kanto (outra geração)."),
    ([25], ["Qual a capital da França?"],
     "Recusa educada, sem sair do personagem."),
    ([9], ["Quanto ele pesa?", "E comparado ao Pikachu?"],
     "85,5 kg; depois compara com os 6,0 kg do Pikachu (usa o histórico)."),
]


def rodar_modelo(pokedex, modelo, forcar_cpu, rapido=False):
    llm.MODELO_OLLAMA = modelo

    opcoes = {"num_gpu": 0} if forcar_cpu else None

    linhas = [
        f"# Teste do LLM: {modelo}{' (CPU)' if forcar_cpu else ''}",
        "",
    ]

    tempos = []

    for numeros, perguntas, esperado in (CASOS[:3] if rapido else CASOS):
        historico = []

        nomes = ", ".join(pokedex.nome_por_numero[n] for n in numeros)

        linhas.append(f"## Detectados: {nomes}")
        linhas.append(f"_Esperado: {esperado}_")
        linhas.append("")

        for pergunta in perguntas:
            perguntas_recentes = [
                m["texto"] for m in historico if m["autor"] == "usuario"
            ][-2:]

            contexto = pokedex.montar_contexto_conversa(
                numeros,
                [pergunta, *reversed(perguntas_recentes)],
            )

            inicio = time.time()

            try:
                resposta = llm.responder(
                    pergunta,
                    contexto,
                    historico,
                    opcoes,
                    pokedex.lembrete_para_pergunta(numeros, pergunta),
                )
            except llm.LLMIndisponivel as erro:
                resposta = f"[ERRO] {erro}"

            tempo = time.time() - inicio
            tempos.append(tempo)

            print(f"[{modelo}] {nomes} | {pergunta} ({tempo:.1f}s)")

            linhas.append(f"**Pergunta:** {pergunta} ({tempo:.1f} s)")
            linhas.append("")
            linhas.append(resposta)
            linhas.append("")

            historico += [
                {"autor": "usuario", "texto": pergunta},
                {"autor": "ia", "texto": resposta},
            ]

    media = sum(tempos) / len(tempos)

    linhas.insert(2, f"Tempo médio por resposta: {media:.1f} s")
    linhas.insert(3, "")

    OUTPUT_PATH.mkdir(exist_ok=True)

    caminho = OUTPUT_PATH / f"teste_llm_{modelo.replace(':', '_')}.md"
    caminho.write_text("\n".join(linhas), encoding="utf-8")

    print(f"Tempo médio: {media:.1f} s | respostas em {caminho}")


def main():
    argumentos = sys.argv[1:]

    forcar_cpu = "--cpu" in argumentos

    # --rapido roda só os 3 primeiros casos (útil para medir tempo na CPU).
    rapido = "--rapido" in argumentos

    modelos = [
        a for a in argumentos if a not in ("--cpu", "--rapido")
    ] or [llm.MODELO_OLLAMA]

    pokedex = Pokedex()

    for modelo in modelos:
        rodar_modelo(pokedex, modelo, forcar_cpu, rapido)


if __name__ == "__main__":
    main()
