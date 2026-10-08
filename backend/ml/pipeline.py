from pathlib import Path
import torch
from PIL import Image
from torchvision.models.detection import (
fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import (
FastRCNNPredictor,
)
from torchvision.transforms import functional as F
from backend.ml.modelo import ModeloPokemon


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DETECTOR_PATH = (
ROOT_DIR
/ "modelos"
/ "pokemon_object_detector.pth"
)
SCORE_MINIMO_DETECTOR = 0.80
CONFIANCA_MINIMA_CLASSIFICADOR = 25.0
CONFIANCA_MINIMA_FALLBACK = 60.0
IOA_THRESHOLD = 0.80
IOA_MESMA_ESPECIE = 0.30


class PipelinePokemon:

    def __init__(
        self,
        caminho_classificador=None,
        caminho_detector=DETECTOR_PATH,
    ):
        self.dispositivo = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.detector = self.carregar_detector(caminho_detector)
        if caminho_classificador is None:
            self.classificador = ModeloPokemon()
        else:
            self.classificador = ModeloPokemon(caminho_classificador)

    def carregar_detector(self, caminho):
        if not caminho.exists():
            raise FileNotFoundError(
                f"Detector nao encontrado: {caminho}"
            )

        checkpoint = torch.load(
            caminho,
            map_location=self.dispositivo,
            weights_only=False,
        )

        modelo = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
        )

        quantidade_entradas = (
            modelo
            .roi_heads
            .box_predictor
            .cls_score
            .in_features
        )

        modelo.roi_heads.box_predictor = (
            FastRCNNPredictor(
                quantidade_entradas,
                2,
            )
        )

        modelo.load_state_dict(
            checkpoint["model_state_dict"]
        )

        modelo = modelo.to(
            self.dispositivo
        )

        modelo.eval()

        return modelo

    def calcular_area(self, box):
        x1, y1, x2, y2 = box

        largura = max(
            0,
            x2 - x1,
        )

        altura = max(
            0,
            y2 - y1,
        )

        return largura * altura

    def calcular_intersecao(
        self,
        box_a,
        box_b,
    ):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        x1 = max(
            ax1,
            bx1,
        )

        y1 = max(
            ay1,
            by1,
        )

        x2 = min(
            ax2,
            bx2,
        )

        y2 = min(
            ay2,
            by2,
        )

        largura = max(
            0,
            x2 - x1,
        )

        altura = max(
            0,
            y2 - y1,
        )

        return largura * altura

    def calcular_ioa(
        self,
        box_menor,
        box_maior,
    ):
        area_menor = self.calcular_area(
            box_menor
        )

        if area_menor <= 0:
            return 0.0

        intersecao = self.calcular_intersecao(
            box_menor,
            box_maior,
        )

        return (
            intersecao
            / area_menor
        )

    def remover_boxes_redundantes(
        self,
        deteccoes,
    ):
        deteccoes_ordenadas = sorted(
            deteccoes,
            key=lambda item: item["score"],
            reverse=True,
        )

        resultado = []

        for deteccao_atual in deteccoes_ordenadas:
            box_atual = deteccao_atual[
                "box"
            ]

            area_atual = self.calcular_area(
                box_atual
            )

            redundante = False

            for deteccao_mantida in resultado:
                box_mantida = deteccao_mantida[
                    "box"
                ]

                area_mantida = self.calcular_area(
                    box_mantida
                )

                if area_atual <= area_mantida:
                    box_menor = box_atual
                    box_maior = box_mantida
                else:
                    box_menor = box_mantida
                    box_maior = box_atual

                ioa = self.calcular_ioa(
                    box_menor,
                    box_maior,
                )

                if ioa >= IOA_THRESHOLD:
                    redundante = True
                    break

            if not redundante:
                resultado.append(
                    deteccao_atual
                )

        return resultado

    def detectar_regioes(
        self,
        imagem,
        score_minimo=None,
        remover_redundantes=True,
    ):
        if score_minimo is None:
            score_minimo = SCORE_MINIMO_DETECTOR

        imagem_tensor = F.to_tensor(
            imagem
        )

        imagem_tensor = imagem_tensor.to(
            self.dispositivo
        )

        with torch.no_grad():
            resultado = self.detector(
                [imagem_tensor]
            )[0]

        boxes = (
            resultado["boxes"]
            .detach()
            .cpu()
        )

        scores = (
            resultado["scores"]
            .detach()
            .cpu()
        )

        labels = (
            resultado["labels"]
            .detach()
            .cpu()
        )

        deteccoes = []

        for box, score, label in zip(
            boxes,
            scores,
            labels,
        ):
            if label.item() != 1:
                continue

            score_valor = score.item()

            if (
                score_valor
                < score_minimo
            ):
                continue

            x1 = int(
                box[0].item()
            )

            y1 = int(
                box[1].item()
            )

            x2 = int(
                box[2].item()
            )

            y2 = int(
                box[3].item()
            )

            deteccoes.append(
                {
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    "score": (
                        score_valor * 100
                    ),
                }
            )

        if not remover_redundantes:
            return deteccoes

        return self.remover_boxes_redundantes(
            deteccoes
        )

    def classificar_regioes(
        self,
        imagem,
        deteccoes,
    ):
        resultados = []

        for deteccao in deteccoes:
            box = deteccao["box"]

            recorte = imagem.crop(
                box
            )

            previsoes = (
                self.classificador.prever(
                    recorte
                )
            )

            melhor = previsoes[0]

            resultados.append(
                {
                    "box": box,
                    "confianca_detector": (
                        deteccao["score"]
                    ),
                    "pokemon": (
                        melhor["pokemon"]
                    ),
                    "confianca": (
                        melhor["confianca"]
                    ),
                    "top_5": previsoes,
                    "fallback": False,
                }
            )

        return resultados

    def filtrar_resultados(
        self,
        resultados,
    ):
        resultados_validos = []

        for resultado in resultados:
            if (
                resultado["confianca"]
                >= CONFIANCA_MINIMA_CLASSIFICADOR
            ):
                resultados_validos.append(
                    resultado
                )

        return resultados_validos

    def tentar_fallback(
        self,
        imagem,
    ):
        previsoes = self.classificador.prever(
            imagem
        )

        melhor_previsao = previsoes[0]

        confianca = melhor_previsao[
            "confianca"
        ]

        if (
            confianca
            < CONFIANCA_MINIMA_FALLBACK
        ):
            return None

        largura, altura = imagem.size

        resultado = {
            "box": (
                0,
                0,
                largura,
                altura,
            ),
            "confianca_detector": None,
            "pokemon": melhor_previsao[
                "pokemon"
            ],
            "confianca": confianca,
            "top_5": previsoes,
            "fallback": True,
        }

        return resultado

    def unir_mesma_especie(
        self,
        resultados,
    ):
        mantidos = []

        for resultado in sorted(
            resultados,
            key=lambda item: item["confianca"],
            reverse=True,
        ):
            repetido = False

            for mantido in mantidos:
                if mantido["pokemon"] != resultado["pokemon"]:
                    continue

                if self.calcular_area(resultado["box"]) <= self.calcular_area(
                    mantido["box"]
                ):
                    box_menor = resultado["box"]
                    box_maior = mantido["box"]
                else:
                    box_menor = mantido["box"]
                    box_maior = resultado["box"]

                if (
                    self.calcular_ioa(box_menor, box_maior)
                    >= IOA_MESMA_ESPECIE
                ):
                    repetido = True
                    break

            if not repetido:
                mantidos.append(resultado)

        return mantidos

    def processar(
        self,
        imagem,
    ):
        if not isinstance(
            imagem,
            Image.Image,
        ):
            raise TypeError(
                "A imagem deve ser uma imagem PIL."
            )

        imagem = imagem.convert(
            "RGB"
        )

        deteccoes = self.detectar_regioes(
            imagem
        )

        resultados = self.classificar_regioes(
            imagem,
            deteccoes,
        )

        resultados_validos = self.unir_mesma_especie(
            self.filtrar_resultados(
                resultados
            )
        )

        if len(resultados_validos) == 0:
            fallback = self.tentar_fallback(
                imagem
            )

            if fallback is not None:
                resultados_validos.append(
                    fallback
                )

        return resultados_validos

