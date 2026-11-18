# Projeto SIN393 Introdução à Visão Computacional

Esse repositório é referente ao projeto da disciplina SIN393 Introdução à Visão Computacional, ministrada pelo professor João Fernando Mari.

Como foi pedido na espeficicação do projeto, o repositorio contém o código utilizado para a realização do trabalho.

## Descrição do Projeto

### Dataset

O trabalho foi realizado sobre o dataset Chest X-Ray Images (Pneumonia), disponivel no kaggle em: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia.

O Dataset contém 1666 imagens de classe Normal, e 4266 pneumonia (viral ou bacterial), totalizando 5932 imagens.
As imagens são de tamanhos váriaveis, por isso é realizado o resizing no código.

O objetivo do trabalho é realizar a comparação de duas Redes Neurais Convolucionais na classificação de imagens de raio-x pulmonar para a classificação em duas classes: normal ou pneumonia.

### Modelos Escolhidos

As redes escolhidas foram a ResNet50 e a MobileNetV3-Large, ambas redes pre-treinadas no imagenet. 

Essa escolha foi feita com um intuito de comparar uma rede mais robusta, como a ResNet50 que possui cerca de 24M parametros, com uma rede mais leve, a MobileNetV3-Large que possui cerca de 6M parametros.

### Setup

Todos os testes foram feitos no Google Colab, utilizando a gpu NVIDIA Tesla T4.

Os modelos pre-treinados foram tirados do Torchvision.

Grande parte do código utilizado foi proveniente do código exemplo disponibilizado pelo professor João.

Por motivos de reprodutibilidade e comparação mais justa, todos o testes foram feitos com uma semente fixa, e parâmetros iguais a menos que seja especificado o contrario. 






