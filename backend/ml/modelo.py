from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODELO_PATH = (
ROOT_DIR
/ "modelos"
/ "pokemon_resnet18_v2.pth"
)

TRANSFORM_IMAGEM = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225],
),
])

class ModeloPokemon:

    def __init__(self):
        self.dispositivo = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.modelo = None
        self.classes = None

        self.carregar_modelo()

    def carregar_modelo(self):
        if not MODELO_PATH.exists():
            raise FileNotFoundError(
                f"Modelo nao encontrado em: {MODELO_PATH}"
            )

        checkpoint = torch.load(
            MODELO_PATH,
            map_location=self.dispositivo,
            weights_only=False,
        )

        self.classes = checkpoint["classes"]

        quantidade_classes = checkpoint["num_classes"]

        modelo = resnet18(weights=None)

        quantidade_entradas = modelo.fc.in_features

        modelo.fc = nn.Linear(
            quantidade_entradas,
            quantidade_classes,
        )

        modelo.load_state_dict(
            checkpoint["model_state_dict"]
        )

        modelo = modelo.to(
            self.dispositivo
        )

        modelo.eval()

        self.modelo = modelo

    def preparar_imagem(self, imagem):
        if not isinstance(imagem, Image.Image):
            raise TypeError(
                "A imagem deve ser uma imagem PIL."
            )

        imagem = imagem.convert("RGB")

        imagem = TRANSFORM_IMAGEM(
            imagem
        )

        imagem = imagem.unsqueeze(0)

        imagem = imagem.to(
            self.dispositivo
        )

        return imagem

    def prever(self, imagem):
        imagem_processada = self.preparar_imagem(
            imagem
        )

        with torch.no_grad():
            saidas = self.modelo(
                imagem_processada
            )

            probabilidades = torch.softmax(
                saidas,
                dim=1,
            )

        probabilidades = probabilidades[0]

        quantidade_resultados = min(
            5,
            len(self.classes),
        )

        top_probabilidades, top_indices = torch.topk(
            probabilidades,
            k=quantidade_resultados,
        )

        resultados = []

        for probabilidade, indice in zip(
            top_probabilidades,
            top_indices,
        ):
            nome_pokemon = self.classes[
                indice.item()
            ]

            confianca = (
                probabilidade.item()
                * 100
            )

            resultados.append(
                {
                    "pokemon": nome_pokemon,
                    "confianca": round(
                        confianca,
                        2,
                    ),
                }
            )

        return resultados
