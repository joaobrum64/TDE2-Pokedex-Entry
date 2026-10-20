import threading
import uuid
from collections import OrderedDict


# Conversas guardadas só na memória: somem quando o backend para e
# nada é gravado em disco. Acima deste número, as mais antigas são
# descartadas para a memória não crescer sem limite.
MAXIMO_CONVERSAS = 500


class Conversas:
    # Uma conversa por imagem enviada (ou por busca manual), com os
    # Pokémon detectados e o histórico de perguntas e respostas.

    def __init__(self):
        self._conversas = OrderedDict()
        self._trava = threading.Lock()

    def criar(self, numeros):
        conversa_id = uuid.uuid4().hex

        with self._trava:
            self._conversas[conversa_id] = {
                "numeros": list(numeros),
                "mensagens": [],
            }

            while len(self._conversas) > MAXIMO_CONVERSAS:
                self._conversas.popitem(last=False)

        return conversa_id

    def obter(self, conversa_id):
        # Cópia, para quem lê não ver o histórico mudar no meio.
        with self._trava:
            conversa = self._conversas.get(conversa_id)

            if conversa is None:
                return None

            return {
                "numeros": list(conversa["numeros"]),
                "mensagens": list(conversa["mensagens"]),
            }

    def registrar(self, conversa_id, pergunta, resposta):
        # Só perguntas respondidas entram no histórico.
        with self._trava:
            conversa = self._conversas.get(conversa_id)

            if conversa is None:
                return

            conversa["mensagens"].append(
                {"autor": "usuario", "texto": pergunta}
            )
            conversa["mensagens"].append(
                {"autor": "ia", "texto": resposta}
            )
