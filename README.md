# Telco Customer Churn Prediction

Projeto de engenharia de Machine Learning end-to-end: do dado bruto ao modelo rastreado com MLflow, com pipeline reprodutível e boas práticas de engenharia de software.

**Tech Challenge — PosTech | Autores:** Patrick Covre · Beatriz Ferrante · Cecilia Rocha

---

## O que o projeto faz

Uma operadora de telecomunicações precisa identificar clientes com risco de cancelamento (churn). Este projeto treina e compara modelos de ML — incluindo uma rede neural MLP com PyTorch — e registra todos os experimentos no MLflow para rastreabilidade.

---

## Setup

### Pré-requisitos

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation) instalado

### Instalação

```bash
# Clone o repositório
git clone <url-do-repo>
cd tech-challenge-step-1

# Instala todas as dependências via Poetry
poetry install
```

### Rodar os notebooks

Abra o VSCode, selecione o kernel do Poetry (`tech-challenge-...py3.14`) e execute as células em ordem:

1. `src/notebooks/eda.ipynb` — análise exploratória
2. `src/notebooks/etapa1_baselines.ipynb` — baselines no MLflow
3. `src/notebooks/etapa2_experimentos.ipynb` — MLP PyTorch + comparação

### Visualizar experimentos no MLflow

Após rodar os notebooks, abra o PowerShell **em qualquer diretório** e rode:

```powershell
# Substitua <usuario> pelo seu nome de usuário do Windows
& $PY314 -m mlflow ui --backend-store-uri "sqlite:///C:/Users/<usuario>/Tech-challenge-step-1/Tech-challenge-step-1/mlruns.db" --port 5000
```

> **Importante:** use o caminho absoluto com barras `/` — o caminho relativo `sqlite:///mlruns.db` só funciona se o terminal estiver na raiz do projeto.

Acesse `http://localhost:5000`.

---

## Estrutura do projeto

```
tech-challenge/
│
├── src/
│   ├── config.py               ← Configurações centrais
│   ├── pipeline.py             ← Orquestrador do pipeline completo
│   │
│   ├── data/
│   │   ├── churn.csv           ← Dataset (Telco Customer Churn — IBM)
│   │   └── loader.py           ← Carregamento e pré-processamento
│   │
│   ├── models/
│   │   ├── mlp.py              ← Arquitetura da rede neural (PyTorch)
│   │   ├── mlp_trainer.py      ← Loop de treino com early stopping
│   │   ├── registry.py         ← Catálogo de modelos disponíveis
│   │   └── trainer.py          ← Treino/avaliação de modelos sklearn
│   │
│   ├── service/
│   │   └── mlflow_service.py   ← Registro padronizado no MLflow
│   │
│   ├── notebooks/
│   │   ├── eda.ipynb                   ← Etapa 1: análise exploratória
│   │   ├── etapa1_baselines.ipynb      ← Etapa 1: baselines registrados
│   │   └── etapa2_experimentos.ipynb  ← Etapa 2: MLP + comparação de modelos
│   │
│   ├── docs/                   ← Gráficos gerados pelos notebooks
│   └── tests/                  ← Testes automatizados (pytest)
│
├── pyproject.toml              ← Dependências, linting, pytest
├── Makefile                    ← Atalhos de linha de comando
└── mlruns.db                   ← Banco de dados do MLflow (gerado ao rodar)
```

---

## Guia dos arquivos — o que cada um faz e por que existe

### `src/config.py` — Configurações centrais

**O que faz:** Define três dataclasses de configuração:
- `DataConfig` — caminho do dataset, nome da coluna alvo, tamanho do split
- `MLPConfig` — arquitetura da rede neural, hiperparâmetros de treino
- `MLflowConfig` — URI do banco e nome do experimento
- `setup_logging()` — configura logging estruturado (sem `print()`)

**Por que existe:** Centralizar todas as configurações em um único lugar. Quando você quiser trocar o dataset ou ajustar hiperparâmetros, mexe só aqui — sem precisar caçar valores espalhados pelo código.

```python
# Exemplo: mudar o experimento sem tocar em outros arquivos
mlflow_cfg = MLflowConfig(experiment_name="meu-experimento-v2")
```

---

### `src/data/loader.py` — Carregamento e pré-processamento

**O que faz:** A classe `ChurnDataLoader`:
1. Lê qualquer CSV e faz limpeza específica do dataset Telco (converte `TotalCharges` de string para float, remove nulos)
2. Codifica a coluna alvo (Yes/No → 1/0)
3. Divide em treino/teste com estratificação (preserva a proporção de churn)
4. Aplica `StandardScaler` nas features numéricas e `OneHotEncoder` nas categóricas
5. Retorna os arrays prontos para sklearn e PyTorch

**Por que existe:** Sem esse módulo, cada notebook repetiria o mesmo código de limpeza. Se o dataset mudar ou você descobrir um bug no pré-processamento, corrige em um lugar só e todos os notebooks se beneficiam.

```python
loader = ChurnDataLoader(path="src/data/churn.csv", target="Churn")
df = loader.load()
X_train, X_test, y_train, y_test, preprocessor = loader.get_splits(df)
```

---

### `src/models/mlp.py` — Arquitetura da rede neural

**O que faz:** Define a classe `ChurnMLP` (herda de `nn.Module`). A arquitetura é configurável via listas:
- `hidden_sizes` — número de neurônios por camada oculta
- `dropout_rates` — taxa de dropout por camada (0.0 = sem BatchNorm nem Dropout)

Padrão: `[128, 64, 32]` com dropouts `[0.3, 0.2, 0.0]`.

**Por que existe:** Separar a definição da arquitetura do loop de treino é uma boa prática em PyTorch. Você pode instanciar o modelo em qualquer lugar, salvar seus pesos e trocar a arquitetura sem reescrever o treino.

```python
# Testar uma arquitetura maior sem mudar o trainer
model = ChurnMLP(input_dim=46, hidden_sizes=[256, 128, 64], dropout_rates=[0.4, 0.3, 0.0])
```

---

### `src/models/registry.py` — Catálogo de modelos

**O que faz:** O dicionário `MODEL_REGISTRY` mapeia um nome para um modelo instanciado + seus hiperparâmetros. Inclui: DummyClassifier, Logistic Regression, Ridge, Decision Tree, Random Forest, MLP sklearn, CatBoost e XGBoost (quando instalados).

**Por que existe:** Para adicionar um novo modelo ao experimento, você só registra aqui — o pipeline e os notebooks pegam automaticamente. Sem esse padrão, você precisaria editar múltiplos arquivos e corria risco de esquecer de logar os parâmetros corretos no MLflow.

```python
# Adicionar um novo modelo: só isso, em um lugar só
MODEL_REGISTRY["meu_modelo"] = {
    "model": MeuClassificador(param=valor),
    "params": {"param": valor},
}
```

---

### `src/models/trainer.py` — Treino e avaliação sklearn

**O que faz:** Dois componentes:
- `METRICS_REGISTRY` — dicionário de métricas disponíveis (accuracy, f1_macro, precision, recall, roc_auc)
- `SklearnTrainer.fit_evaluate()` — treina qualquer modelo sklearn e retorna as métricas calculadas

**Por que existe:** Padroniza o cálculo de métricas. Sem isso, cada modelo calcularia métricas de forma diferente e você poderia usar `average="binary"` em um lugar e `average="macro"` em outro — tornando a comparação entre modelos inválida.

```python
# Adicionar uma nova métrica para todos os modelos de uma vez
from sklearn.metrics import average_precision_score
METRICS_REGISTRY["pr_auc"] = lambda yt, _yp, prob: average_precision_score(yt, prob)
```

---

### `src/models/mlp_trainer.py` — Treino PyTorch com early stopping

**O que faz:** A classe `PyTorchMLPTrainer`:
1. Separa 10% do treino para validação interna (early stopping)
2. Executa o loop de treino com mini-batches
3. Aplica `StepLR` para decaimento da taxa de aprendizado
4. Para automaticamente quando a loss de validação para de melhorar (`EarlyStopping`)
5. Avalia no conjunto de teste e retorna métricas + lista de losses por época

**Por que existe:** O loop de treino PyTorch é verboso. Isolado aqui, o notebook fica limpo — só chama `fit_evaluate()` e recebe os resultados. O early stopping é fundamental para evitar overfitting sem precisar definir o número exato de épocas manualmente.

---

### `src/service/mlflow_service.py` — Registro no MLflow

**O que faz:** A classe `MLflowService` padroniza como os experimentos são logados:
- `log_sklearn_run()` — loga modelo sklearn, métricas, parâmetros e info do dataset
- `log_pytorch_run()` — idem para PyTorch, mais a curva de loss por época

**Por que existe:** Sem esse serviço, cada notebook chamaria `mlflow.log_metric()`, `mlflow.log_params()` etc. de forma diferente. Experimentos logados de forma inconsistente são difíceis de comparar na UI do MLflow. Centralizando aqui, todos os runs seguem o mesmo padrão.

---

### `src/pipeline.py` — Orquestrador

**O que faz:** A classe `ChurnPipeline` conecta todos os módulos em sequência:
`ChurnDataLoader` → `SklearnTrainer` (para cada modelo do registry) → `PyTorchMLPTrainer` → `MLflowService`

**Por que existe:** Permite rodar o experimento completo com uma linha de Python, sem abrir nenhum notebook. Também serve de base para automação futura (CI/CD, agendamento de retreino).

```bash
make run
# equivalente a: python -c "from src.pipeline import ChurnPipeline; ChurnPipeline().run()"
```

---

### `src/notebooks/` — Notebooks por etapa

Os notebooks são a **interface narrativa** do projeto. Eles não contêm lógica de negócio — importam dos módulos `src/` e apresentam a análise de forma legível.

| Notebook | Etapa | O que entrega |
|---|---|---|
| `eda.ipynb` | 1 | Análise exploratória + justificativa das métricas escolhidas |
| `etapa1_baselines.ipynb` | 1 | DummyClassifier e Logistic Regression registrados no MLflow |
| `etapa2_experimentos.ipynb` | 2 | MLP PyTorch + comparação de 6+ modelos + análise de custo FP/FN |

**Por que separar notebooks de módulos Python?**

Notebooks são ótimos para exploração e apresentação, mas péssimos para reuso e teste. Código em `.py` pode ser importado, versionado, testado com pytest e executado em CI. A combinação dos dois é o padrão da indústria.

---

### `pyproject.toml` — Fonte única de verdade

**O que faz:** Define tudo sobre o projeto em um arquivo só:
- Dependências de produção (pandas, torch, mlflow, sklearn...)
- Dependências de desenvolvimento (ruff, pytest)
- Configuração do linter (ruff) e do pytest

**Por que existe:** Sem isso, cada desenvolvedor instalaria versões diferentes das bibliotecas e os resultados seriam irreproduíveis. O `poetry.lock` garante que todos usem exatamente as mesmas versões.

---

### `Makefile` — Atalhos de linha de comando

**O que faz:** Agrupa os comandos mais usados:

```bash
make install    # instala dependências
make lint       # verifica estilo com ruff
make lint-fix   # corrige automaticamente
make test       # roda pytest
make run        # executa o pipeline completo
make mlflow     # sobe a UI do MLflow na porta 5000
```

**Por que existe:** Padroniza como qualquer pessoa da equipe (ou um CI) executa as tarefas, sem precisar decorar comandos longos ou consultar documentação.

---

## Como tudo se conecta

```
eda.ipynb
    └── ChurnDataLoader  ←── src/data/loader.py

etapa1_baselines.ipynb
    ├── ChurnDataLoader
    ├── SklearnTrainer   ←── src/models/trainer.py
    └── MLflowService    ←── src/service/mlflow_service.py

etapa2_experimentos.ipynb
    ├── ChurnDataLoader
    ├── MODEL_REGISTRY   ←── src/models/registry.py
    ├── SklearnTrainer
    ├── PyTorchMLPTrainer ←── src/models/mlp_trainer.py
    │       └── ChurnMLP ←── src/models/mlp.py
    └── MLflowService

pipeline.py  (roda tudo acima via linha de comando)
```

---

## Como adicionar um novo modelo ou dataset

### Caso 1 — Novo modelo sklearn (ex: SVM, LightGBM)

Edite **apenas** [`src/models/registry.py`](src/models/registry.py):

```python
# 1. Importe o modelo no topo do arquivo
from sklearn.svm import SVC

# 2. Adicione a entrada no dicionário MODEL_REGISTRY
MODEL_REGISTRY["svm"] = {
    "model": SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
    "params": {"C": 1.0, "kernel": "rbf"},
}
```

Pronto. Na próxima execução dos notebooks o SVM aparece automaticamente na tabela comparativa e é registrado no MLflow.

---

### Caso 2 — Nova arquitetura MLP PyTorch

Se quiser mudar camadas/dropout sem criar novo arquivo, edite [`src/config.py`](src/config.py):

```python
mlp_cfg = MLPConfig(
    hidden_sizes=[256, 128, 64, 32],   # arquitetura mais profunda
    dropout_rates=[0.4, 0.3, 0.2, 0.0],
    learning_rate=5e-4,
    epochs=150,
    early_stopping_patience=15,
)
```

Se quiser uma arquitetura completamente diferente (ex: com skip connections), crie um novo arquivo `src/models/minha_rede.py` herdando de `nn.Module`:

```python
# src/models/minha_rede.py
import torch.nn as nn

class MinhaRede(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # defina as camadas aqui
        ...

    def forward(self, x):
        ...
        return x.squeeze(1)  # retorna um logit por amostra
```

Depois instancie essa rede dentro de `PyTorchMLPTrainer` (ou crie um trainer dedicado seguindo o mesmo padrão de `mlp_trainer.py`).

---

### Caso 3 — Novo dataset

**Passo 1 — Coloque o CSV em `src/data/`**

```
src/data/
├── churn.csv          ← dataset original
└── novo_dataset.csv   ← novo arquivo
```

**Passo 2 — Atualize `DataConfig` em [`src/config.py`](src/config.py)**

```python
data_cfg = DataConfig(
    path=Path("src/data/novo_dataset.csv"),
    target="nome_da_coluna_alvo",   # coluna binária (0/1 ou Yes/No)
    test_size=0.2,
)
```

**Passo 3 — Verifique se o pré-processamento se aplica**

Abra [`src/data/loader.py`](src/data/loader.py) e revise o método `_clean()`:

```python
def _clean(self, df):
    # Esta linha é específica do dataset Telco (TotalCharges vem como string)
    # Se o seu dataset não tiver esse problema, pode remover ou adaptar
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Esta linha funciona para qualquer dataset com target Yes/No
    # Se o seu alvo já for 0/1, o (== "Yes") retornará False para 0 e True para 1
    # Nesse caso substitua por: df[self.target] = df[self.target].astype(int)
    if df[self.target].dtype == object:
        df[self.target] = (df[self.target] == "Yes").astype(int)
```

**Passo 4 — Atualize o `MLflowConfig` para separar os experimentos**

```python
mlflow_cfg = MLflowConfig(experiment_name="novo-dataset-experimento-v1")
```

**Passo 5 — Rode os notebooks normalmente**

O `ChurnDataLoader` detecta automaticamente quais colunas são numéricas e quais são categóricas e aplica o pré-processamento correto. Nenhuma outra mudança é necessária.

---

### Caso 4 — Novo modelo + novo dataset ao mesmo tempo

Combine os passos acima. A ordem recomendada é:

```
1. Adiciona o CSV em src/data/
2. Atualiza DataConfig (path + target)
3. Verifica _clean() em loader.py
4. Adiciona o modelo em registry.py
5. Atualiza experiment_name no MLflowConfig
6. Roda os notebooks
7. Compara no MLflow UI (http://localhost:5000)
```

---

## Boas práticas aplicadas

| Prática | Onde |
|---|---|
| Seeds fixados para reprodutibilidade | `config.py` (`RANDOM_STATE = 42`) |
| Logging estruturado (sem `print()`) | `config.py` + `setup_logging()` |
| Validação cruzada estratificada | `etapa2_experimentos.ipynb` seção 10 |
| Versão do dataset registrada (SHA-256) | `loader.py` + `mlflow_service.py` |
| Early stopping | `mlp_trainer.py` |
| Linting com ruff | `pyproject.toml` + `Makefile` |
| Testes automatizados | `src/tests/` |
