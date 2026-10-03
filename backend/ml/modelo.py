from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import (
ConvNeXt_Tiny_Weights,
EfficientNet_B0_Weights,
ResNet18_Weights,
convnext_tiny,
efficientnet_b0,
resnet18,
)


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELO_PATH = (
ROOT_DIR
/ "modelos"
/ "pokemon_resnet18_v3.pth"
)
TRANSFORM_IMAGEM = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225],
),
])

def criar_arquitetura(nome, quantidade_classes, pre_treinado=False):
 
    if nome == "resnet18":
        modelo = resnet18(
            weights=ResNet18_Weights.DEFAULT if pre_treinado else None
        )
        modelo.fc = nn.Linear(modelo.fc.in_features, quantidade_classes)

    elif nome == "efficientnet_b0":
        modelo = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT if pre_treinado else None
        )
        modelo.classifier[1] = nn.Linear(
            modelo.classifier[1].in_features,
            quantidade_classes,
        )

    elif nome == "convnext_tiny":
        modelo = convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.DEFAULT if pre_treinado else None
        )
        modelo.classifier[2] = nn.Linear(
            modelo.classifier[2].in_features,
            quantidade_classes,
        )

    else:
        raise ValueError(f"Arquitetura desconhecida: {nome}")

    return modelo


class ModeloPokemon:

    def __init__(self, caminho=MODELO_PATH):
        self.dispositivo = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.modelo = None
        self.classes = None

        self.carregar_modelo(caminho)

    def carregar_modelo(self, caminho):
        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo nao encontrado em: {caminho}"
            )

        checkpoint = torch.load(
            caminho,
            map_location=self.dispositivo,
            weights_only=False,
        )

        self.classes = checkpoint["classes"]

        quantidade_classes = checkpoint["num_classes"]

        modelo = criar_arquitetura(
            checkpoint["architecture"],
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
