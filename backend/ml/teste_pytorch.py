import torch


def testar_pytorch():
    print("=" * 60)
    print("TESTE DO PYTORCH")
    print("=" * 60)

    print()
    print(f"Versão do PyTorch: {torch.__version__}")

    cuda_disponivel = torch.cuda.is_available()

    print()
    print(f"CUDA disponível: {cuda_disponivel}")

    if not cuda_disponivel:
        print()
        print("A GPU NVIDIA não está disponível para o PyTorch.")
        print("O treinamento ainda poderia usar CPU, mas não vamos seguir")
        print("antes de verificar a configuração da GPU.")
        return

    print()
    print(f"Versão CUDA do PyTorch: {torch.version.cuda}")

    quantidade_gpus = torch.cuda.device_count()

    print(f"Quantidade de GPUs detectadas: {quantidade_gpus}")

    print()
    print("GPUs:")

    for indice in range(quantidade_gpus):
        nome_gpu = torch.cuda.get_device_name(indice)

        print(f"{indice}: {nome_gpu}")

    dispositivo = torch.device("cuda")

    print()
    print(f"Dispositivo selecionado: {dispositivo}")

    tensor_cpu = torch.tensor(
        [1.0, 2.0, 3.0]
    )

    tensor_gpu = tensor_cpu.to(dispositivo)

    print()
    print(f"Tensor original: {tensor_cpu}")
    print(f"Dispositivo original: {tensor_cpu.device}")

    print()
    print(f"Tensor na GPU: {tensor_gpu}")
    print(f"Dispositivo do tensor: {tensor_gpu.device}")

    resultado = tensor_gpu * 2

    print()
    print(f"Resultado calculado na GPU: {resultado}")

    print()
    print("=" * 60)
    print("PYTORCH + GPU FUNCIONANDO")
    print("=" * 60)


if __name__ == "__main__":
    testar_pytorch()