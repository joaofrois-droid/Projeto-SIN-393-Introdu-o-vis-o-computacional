# Detecção de pneumonia em radiografias de tórax com Deep Learning: ResNet50 × MobileNetV3

Projeto final da disciplina **SIN 393 – Introdução à Visão Computacional (2026)**
Universidade Federal de Viçosa – Campus Rio Paranaíba · Prof. João Fernando Mari

**Integrantes:** `João Vitor de Freitas Frois - 9881` · `Cainá de Almeida Lopes - 9892 `

📄 **Artigo (formato IEEE):** [`Relatorio.pdf`](Relatorio.pdf)

🎥 **Vídeo de apresentação:** [`LINK DO YOUTUBE`](https://www.youtube.com/)

---

## Sumário

- [Descrição do projeto](#descrição-do-projeto)
- [Dataset](#dataset)
- [Metodologia](#metodologia)
- [Resultados](#resultados)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como executar](#como-executar)
- [Arquivos gerados](#arquivos-gerados)
- [Limitações](#limitações)
- [Referências](#referências)

---

## Descrição do projeto

O projeto classifica radiografias de tórax pediátricas em duas classes, **normal** e **pneumonia**, usando redes neurais convolucionais pré-treinadas no ImageNet e ajustadas (*fine-tuning*) com PyTorch.

O trabalho investiga duas questões experimentais:

1. **Comparação entre arquiteturas (questão principal):** uma rede grande, a **ResNet50** (23,5 M parâmetros), contra uma rede leve, feita para dispositivos móveis, a **MobileNetV3-Large** (4,2 M parâmetros). A comparação avalia o desempenho de classificação e o custo computacional: tamanho do modelo, tempo de treino e tempo de inferência em GPU e CPU.
2. **Efeito do desbalanceamento das classes (questão secundária):** o conjunto de treino tem cerca de 3 imagens de pneumonia para cada imagem normal. Os dois modelos são treinados com a função de perda *Cross-Entropy* **com** e **sem** pesos por classe, e os resultados são comparados.

## Dataset

**Chest X-Ray Images (Pneumonia)**, disponível no Kaggle:
<https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>

O dataset foi apresentado na primeira etapa da disciplina e aprovado pelo professor para a realização do trabalho.

São 5.856 radiografias de tórax (incidência anteroposterior) de crianças de 1 a 5 anos, publicadas por Kermany et al. (2018). A classe **pneumonia** reúne casos bacterianos e virais, que neste trabalho são tratados como uma única classe. As imagens têm tamanhos variados, por isso todas são redimensionadas antes de entrar na rede.

| Pasta original | Normal | Pneumonia | Total | Uso neste projeto |
|---|---:|---:|---:|---|
| `train` | 1.341 | 3.875 | 5.216 | dividida em treino e validação **por paciente** |
| `val` | 8 | 8 | 16 | não utilizada (pequena demais) |
| `test` | 234 | 390 | 624 | teste, usado **apenas na avaliação final** |

### Divisão treino/validação por paciente

Muitos pacientes têm mais de uma radiografia: 1.005 pacientes com pneumonia têm várias imagens, e um deles tem 30. Se as imagens de um mesmo paciente caírem no treino e na validação, o modelo "reconhece" o paciente e a validação fica otimista (**vazamento de dados**).

Para evitar isso, o identificador do paciente é extraído do nome do arquivo:

- `person123_bacteria_456.jpeg` → `person123`
- `IM-0115-0001.jpeg` → `IM-0115`
- `NORMAL2-IM-0383-0001.jpeg` → `NORMAL2-IM-0383`

Os **pacientes** são então divididos com `train_test_split` (scikit-learn), estratificado por classe, com semente fixa (`SEED = 42`) e proporção 80/20. Todas as imagens de cada paciente vão para o mesmo conjunto.

| Conjunto | Pacientes | Imagens | Normal | Pneumonia |
|---|---:|---:|---:|---:|
| Treino | 2.276 | 4.209 | 1.077 (25,6%) | 3.132 (74,4%) |
| Validação | 570 | 1.007 | 264 (26,2%) | 743 (73,8%) |
| Teste | — | 624 | 234 (37,5%) | 390 (62,5%) |

Com a divisão por paciente, na ResNet50 com pesos, o F1 macro de validação (0,90) ficou praticamente igual ao do teste (0,896). Nas outras três configurações, a validação (F1 macro ≈ 0,94) ainda ficou acima do teste, o que se deve à diferença de distribuição entre os conjuntos `train` e `test` originais (veja [Limitações](#limitações)).

## Metodologia

### Ambiente e base do código

- Todos os experimentos foram executados no **Google Colab**, com GPU **NVIDIA Tesla T4**.
- Os modelos pré-treinados foram obtidos do **torchvision**.
- Grande parte do código (laço de treino, avaliação e relatórios) partiu do **código de exemplo disponibilizado pelo Prof. João Fernando Mari** na disciplina, com adaptações para este problema. As principais adaptações foram:
  - divisão treino/validação por paciente;
  - pesos por classe;
  - seleção do melhor modelo pela validação;
  - curva ROC;
  - análise qualitativa;
  - medição do custo computacional.
- Para garantir reprodutibilidade e uma comparação justa, **todos os experimentos usam a mesma semente fixa e os mesmos hiperparâmetros**, salvo indicação em contrário. As duas arquiteturas são treinadas e avaliadas exatamente nas mesmas condições.

### Modelos

Os dois modelos vêm do `torchvision`, com pesos pré-treinados no ImageNet (`IMAGENET1K_V2`). A última camada é substituída por uma camada linear com 2 saídas, e **todos os pesos são ajustados** (*fine-tuning* completo).

| Modelo | Parâmetros | Tamanho salvo |
|---|---:|---:|
| ResNet50 | 23,5 M | 90,0 MB |
| MobileNetV3-Large | 4,2 M | 16,2 MB |

### Pré-processamento e *data augmentation*

Todas as imagens são redimensionadas para 224×224 e normalizadas com a média e o desvio padrão do ImageNet. Durante o **treino**, são aplicadas as transformações abaixo:

| Transformação | Parâmetros | Justificativa |
|---|---|---|
| `RandomAffine` | rotação ±7°, translação 5%, escala 0,9–1,1 | Simula variações de posicionamento do paciente e de enquadramento. Os valores são pequenos para preservar a anatomia. |
| `ColorJitter` | brilho ±20%, contraste ±20% | Simula diferenças de exposição e de equipamento entre radiografias. |
| `RandomErasing` | p = 0,25, área 2–8% | Torna o modelo mais robusto a oclusões e a marcações sobre a imagem (letras "R", eletrodos, textos). |

**Não** foi usado espelhamento horizontal, porque o tórax não é simétrico (o coração fica à esquerda) e as radiografias têm marcações de lateralidade. Validação e teste recebem apenas o redimensionamento e a normalização.

### Treinamento

| Hiperparâmetro | Valor |
|---|---|
| Otimizador | SGD, *momentum* 0,9 |
| Taxa de aprendizado inicial | 5 × 10⁻⁴ |
| *Scheduler* | `StepLR`: a taxa de aprendizado é multiplicada por 0,1 a cada 10 épocas |
| Épocas | 25 |
| Tamanho do lote | 128 |
| Função de perda | *Cross-Entropy*, com ou sem pesos por classe |
| Pesos por classe | `n_total / (n_classes × n_classe)`: normal = 1,954; pneumonia = 0,672 |
| Semente | 42 (Python, NumPy, PyTorch e DataLoaders) |

**Seleção do modelo:** a cada época, são guardados os pesos da época com **maior F1 macro na validação** (em caso de empate, a de menor perda). Esses pesos são usados no teste. O conjunto de teste não participa de nenhuma decisão de treinamento.

### Avaliação

- Matriz de confusão.
- Acurácia, precisão, *recall* e F1-score por classe, com agregação **macro** (média simples entre as classes, que dá o mesmo peso às duas apesar do desbalanceamento). O relatório do scikit-learn também mostra a média *weighted*.
- Curva ROC e AUC, com pneumonia como classe positiva.
- Análise qualitativa: exemplos de acertos e dos erros mais confiantes (falsos positivos e falsos negativos).
- Custo computacional: número de parâmetros, tamanho do arquivo, tempo médio por época e tempo de inferência por imagem em GPU e CPU.

## Resultados

Os resultados abaixo são do **conjunto de teste**. Os experimentos rodaram no Google Colab com GPU NVIDIA T4.

### Desempenho

| Modelo | Pesos por classe | Acurácia | F1 macro | AUC | Recall normal | Recall pneumonia | FP | FN | Melhor época |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | com | 0,902 | 0,896 | **0,955** | **0,863** | 0,926 | 32 | 29 | 16 |
| ResNet50 | sem | **0,909** | **0,899** | 0,952 | 0,803 | 0,972 | 46 | **11** | 23 |
| MobileNetV3 | com | 0,833 | 0,803 | 0,950 | 0,590 | **0,980** | 96 | 8 | 11 |
| MobileNetV3 | sem | 0,849 | 0,827 | 0,954 | 0,654 | 0,967 | 81 | 13 | 23 |

FP = normal classificado como pneumonia · FN = pneumonia classificada como normal · Melhor época contada a partir de 1 (o notebook imprime as épocas a partir de 0)

### Custo computacional

| Modelo | Parâmetros | Tamanho | Tempo por época (treino + validação) | Inferência GPU | Inferência CPU |
|---|---:|---:|---:|---:|---:|
| ResNet50 | 23,5 M | 90,0 MB | 104–110 s | 6,4–7,0 ms/img | 144–147 ms/img |
| MobileNetV3-Large | 4,2 M | 16,2 MB | 89–97 s | 6,3–9,1 ms/img | 20–28 ms/img |

Os intervalos mostram os valores medidos nas duas execuções (com e sem pesos).

A MobileNetV3 tem **5,6× menos parâmetros**, um arquivo **5,5× menor** e é **5 a 7× mais rápida na CPU**. Na GPU, com uma imagem por vez, os tempos das duas são parecidos.

### Principais conclusões

1. **ResNet50 × MobileNetV3:** nas mesmas condições de avaliação, a ResNet50 foi melhor nas duas configurações (F1 macro de cerca de 0,90 contra 0,80–0,83). A MobileNetV3 errou principalmente imagens normais. Ainda assim, a **AUC das duas é praticamente igual (0,950–0,955)**: a MobileNetV3 ordena as imagens tão bem quanto a ResNet50, mas suas probabilidades ficaram deslocadas para "pneumonia" no conjunto de teste. Na validação, ela teve F1 macro de 0,94, então o problema aparece na mudança de distribuição entre os conjuntos de treino e de teste do dataset.
2. **Desbalanceamento:** os pesos por classe **não melhoraram** o desempenho de forma consistente: o F1 macro e a AUC variaram pouco na ResNet50 e pioraram na MobileNetV3. Eles mudaram o **tipo de erro**: na ResNet50, reduziram os falsos positivos (de 46 para 32), mas aumentaram os falsos negativos (de 11 para 29). Em triagem médica, deixar passar uma pneumonia costuma ser mais grave que um alarme falso. Por esse critério, a **ResNet50 sem pesos** teve o melhor equilíbrio: recall de pneumonia de 0,972 com o maior F1 macro (0,899). A MobileNetV3 com pesos deixou passar ainda menos pneumonias (8), mas classificou 96 radiografias normais como pneumonia.
3. **Análise qualitativa:** na ResNet50, os falsos positivos se concentram nas imagens normais com prefixo `NORMAL2-IM-*`: ela erra 17–24% delas, contra 6–10% das imagens `IM-*`. Isso sugere diferenças de aquisição entre os lotes de imagens, ou rótulos questionáveis. Outros erros envolvem imagens com pouco contraste, mal posicionadas ou com marcações sobre o tórax.

## Estrutura do repositório

```
.
├── README.md
├── Relatorio.pdf              # artigo no formato IEEE
├── requirements.txt
├── TrabalhoVisaoTrue.ipynb    # execução COM pesos por classe (USE_CLASS_WEIGHTS = True), com saídas
└── TrabalhoVisaoFalse.ipynb   # execução SEM pesos por classe (USE_CLASS_WEIGHTS = False), com saídas
```

Os dois notebooks têm **o mesmo código**. A única diferença é o valor de `USE_CLASS_WEIGHTS` na célula de configuração do experimento. Cada notebook treina e avalia a ResNet50 e a MobileNetV3 em sequência.

O dataset **não** faz parte do repositório: ele deve ser baixado do Kaggle (veja abaixo).

## Como executar

### Requisitos

- Python 3.10 ou superior
- GPU com CUDA (recomendado). Na CPU, o treino é muito lento.
- Bibliotecas listadas em [`requirements.txt`](requirements.txt): PyTorch, torchvision, scikit-learn, pandas, NumPy, Matplotlib e Pillow.

Tempo aproximado de uma execução completa (dois modelos, 25 épocas cada) numa GPU T4: **cerca de 85 minutos**.

### Obter o dataset

1. Acesse a [página do dataset no Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) e clique em **Download**. É preciso ter uma conta no Kaggle (gratuita).
2. O arquivo baixado se chama `archive.zip`. Descompacte-o: o resultado ocupa cerca de 2,4 GB e tem esta estrutura:
   ```
   chest_xray/
   ├── train/          ← usada (NORMAL/ e PNEUMONIA/)
   ├── test/           ← usada (NORMAL/ e PNEUMONIA/)
   ├── val/            ← ignorada pelo notebook
   ├── chest_xray/     ← cópia repetida do dataset (pode ser apagada)
   └── __MACOSX/       ← metadados do macOS (pode ser apagada)
   ```
   O zip traz o dataset **duplicado**. Basta a pasta `chest_xray/` de cima, com `train/` e `test/`. Apagar a cópia interna e a `__MACOSX/` economiza cerca de 1,2 GB. A pasta `val/` pode ficar: o notebook não a utiliza, porque cria a sua própria validação a partir de `train/`.

### Opção 1 – Google Colab (recomendada)

1. Na raiz do seu Google Drive (**Meu Drive**), crie uma pasta chamada `DATASET` e envie para ela a pasta `chest_xray/`:
   ```
   Meu Drive/DATASET/chest_xray/
   ├── train/
   └── test/
   ```
   O envio tem milhares de arquivos e pode levar bastante tempo. No código, a raiz aparece como `My Drive`, mas é a mesma pasta "Meu Drive" da interface em português.
2. Abra `TrabalhoVisaoTrue.ipynb` ou `TrabalhoVisaoFalse.ipynb` no Colab.
3. Selecione um ambiente com GPU: **Ambiente de execução → Alterar o tipo de ambiente de execução → GPU T4**.
4. Execute todas as células (**Ambiente de execução → Executar tudo**) e autorize o acesso ao Google Drive quando for solicitado.
   - O notebook copia o dataset do Drive para o disco do Colab, o que torna a leitura das imagens bem mais rápida.
   - Os resultados são salvos em `My Drive/exp-18-CE-pesos/` ou `My Drive/exp-18-CE-sem-pesos/`.

### Opção 2 – Execução local

```bash
git clone https://github.com/joaofrois-droid/Projeto-SIN-393-Introdu-o-vis-o-computacional.git
cd Projeto-SIN-393-Introdu-o-vis-o-computacional
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Para usar GPU NVIDIA, instale o PyTorch com suporte a CUDA seguindo as instruções de <https://pytorch.org/get-started/locally/>.

Depois, mova a pasta `chest_xray/` (veja [Obter o dataset](#obter-o-dataset)) para dentro da pasta do repositório, **ao lado dos notebooks**:

```
.
├── TrabalhoVisaoTrue.ipynb
├── TrabalhoVisaoFalse.ipynb
└── chest_xray/
    ├── train/
    └── test/
```

Por fim, abra o notebook e execute todas as células:

```bash
jupyter notebook TrabalhoVisaoTrue.ipynb
```

Na execução local, os resultados são salvos em `./resultados/exp-18-CE-pesos/` ou `./resultados/exp-18-CE-sem-pesos/`.

### Reproduzir a comparação com e sem pesos

Execute `TrabalhoVisaoTrue.ipynb` e `TrabalhoVisaoFalse.ipynb`. Também é possível usar um único notebook e trocar o valor da variável abaixo entre as execuções:

```python
USE_CLASS_WEIGHTS = True   # com pesos por classe
USE_CLASS_WEIGHTS = False  # sem pesos por classe
```

## Arquivos gerados

Cada execução salva na pasta do experimento (`exp-18-...`) os arquivos abaixo, para cada modelo (`ResNet` / `ResNet50` e `MobileNetV3`):

| Arquivo | Conteúdo |
|---|---|
| `model_<modelo>.pth` | pesos (`state_dict`) da melhor época |
| `loss_history_<modelo>.pdf`, `acc_history_<modelo>.pdf`, `lr_history_<modelo>.pdf` | curvas de perda, acurácia/F1 macro de validação e taxa de aprendizado |
| `confusion_matrix_<modelo>_test.png` | matriz de confusão no teste |
| `roc_<modelo>_test.png` | curva ROC no teste |
| `classification_report_<modelo>_test.txt` | matriz de confusão e relatório de classificação |
| `class_report_detailed_<modelo>_test.csv` | predição e probabilidades de cada imagem de teste |
| `qualitative_<modelo>_test.png` | exemplos de acertos e erros |
| `errors_<modelo>_test.csv` | lista de todos os erros, ordenada por confiança |
| `comparison_models.csv` | tabela comparativa entre os dois modelos |
| `roc_comparison_test.png` | curvas ROC dos dois modelos no mesmo gráfico |

Para carregar um modelo salvo:

```python
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('model_ResNet.pth', map_location='cpu'))
model.eval()
```

## Limitações

- **Uma única execução (semente) por configuração.** Algumas diferenças entre configurações são pequenas e podem ser explicadas pela variação natural do treinamento. O ideal seria repetir cada experimento com várias sementes e reportar média ± desvio padrão.
- **Mudança de distribuição entre treino e teste.** Os conjuntos `train` e `test` originais do dataset têm características diferentes, o que explica parte da queda de desempenho do teste em relação à validação.

## Referências

- KERMANY, D. S. et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. **Cell**, v. 172, n. 5, p. 1122–1131, 2018. DOI: [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010).
- HE, K. et al. Deep Residual Learning for Image Recognition. In: **IEEE Conference on Computer Vision and Pattern Recognition (CVPR)**, 2016, p. 770–778.
- HOWARD, A. et al. Searching for MobileNetV3. In: **IEEE/CVF International Conference on Computer Vision (ICCV)**, 2019, p. 1314–1324.
- MOONEY, P. **Chest X-Ray Images (Pneumonia)**. Kaggle, 2018. Disponível em: <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>.
- PyTorch / torchvision – Models and pre-trained weights: <https://docs.pytorch.org/vision/stable/models.html>.
