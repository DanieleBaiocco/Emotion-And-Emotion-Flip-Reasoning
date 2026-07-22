# MELD Emotion & Emotion-Flip Reasoning

Progetto da riga di comando per il **riconoscimento delle emozioni** e il **rilevamento dei trigger di cambiamento emotivo (Emotion Flip Reasoning)** nelle conversazioni multi-speaker.

Dato un dialogo composto da una sequenza di frasi e dai relativi speaker, il modello predice per ogni turno:

- l'**emozione** associata alla frase;
- la probabilità che la frase sia un **trigger di cambiamento emotivo**;
- una decisione binaria `trigger` (`0` oppure `1`) ottenuta usando la soglia selezionata durante la validazione.

Il progetto è pensato per essere utilizzato interamente da terminale: lo stesso package Python viene usato sia per il training sia per l'inference.

---

## Struttura del progetto

```text
meld-efr-cli/
├── artifacts/                 # Modello allenato e file necessari all'inference
│   ├── best_model.pt
│   ├── tokenizer/
│   ├── encoder_config/
│   └── ...
├── data/                      # Dataset train / validation / test
├── examples/                  # Dialoghi di esempio
├── src/
│   └── meld_efr/
│       ├── cli.py
│       ├── data.py
│       ├── inference.py
│       ├── model.py
│       ├── training.py
│       └── utils.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Requisiti

- Python **3.10+**
- PyTorch
- Transformers
- scikit-learn
- NumPy

Per il training è consigliata una GPU NVIDIA compatibile con CUDA. L'inference può essere eseguita anche su CPU.

---

# Installazione

Clonare il repository:

```bash
git clone <REPOSITORY_URL>
cd Emotion-And-Emotion-Flip-Reasoning
```

Prendi il modello trainato da GitHub:
```
git lfs install
git lfs pull
```

Opzionalmente creare un ambiente virtuale:

```bash
python -m venv .venv
```

Attivarlo su Linux/macOS:

```bash
source .venv/bin/activate
```

Su Windows:

```powershell
.venv\Scripts\activate
```

Installare il progetto:

```bash
pip install -e .
```

L'opzione `-e` installa il progetto in **editable mode**, quindi le modifiche effettuate nei file sotto `src/` vengono utilizzate immediatamente senza dover reinstallare il package.

Dopo l'installazione:

```bash
meld-efr --help
```

oppure:

```bash
python -m meld_efr --help
```

---

# Inference / Prediction

Per eseguire l'inference è necessario un checkpoint allenato, normalmente:

```text
artifacts/best_model.pt
```

Comando:

```bash
meld-efr predict \
  --checkpoint artifacts/best_model.pt \
  --input examples/dialogue1.json \
  --output artifacts/dialogue1_predictions.json \
  --device auto
```

Equivalente:

```bash
python -m meld_efr predict \
  --checkpoint artifacts/best_model.pt \
  --input examples/dialogue1.json \
  --output artifacts/dialogue1_predictions.json \
  --device auto
```

## Opzioni di `predict`

| Opzione | Obbligatoria | Descrizione |
|---|---:|---|
| `--checkpoint PATH` | Sì | Percorso del checkpoint `.pt` allenato. Normalmente `artifacts/best_model.pt`. |
| `--input PATH` | Sì | File JSON contenente un dialogo oppure una lista di dialoghi. |
| `--output PATH` | No | File JSON in cui salvare le predizioni. Se omesso, le predizioni vengono stampate a terminale. |
| `--device DEVICE` | No | Device per l'inference: `auto`, `cpu`, `cuda`, `cuda:0`, ecc. |

Con:

```bash
--device auto
```

il programma usa automaticamente CUDA se è disponibile una GPU compatibile; altrimenti usa la CPU.

---

# Formato dell'input

Ogni dialogo deve contenere:

- `utterances`: lista ordinata dei turni della conversazione;
- `speakers`: speaker associato a ciascuna utterance.

Le due liste devono avere la stessa lunghezza.

Per l'inference **non è necessario fornire emozioni o trigger reali**.

---

# Esempio completo con `dialogue1.json`

File:

```text
examples/dialogue1.json
```

Contenuto:

```json
{
  "utterances": [
    "I finally got the promotion!",
    "That is amazing, congratulations!",
    "Thanks. I was sure they would choose someone else.",
    "You earned it. We should celebrate tonight."
  ],
  "speakers": [
    "A",
    "B",
    "A",
    "B"
  ]
}
```

Esecuzione:

```bash
meld-efr predict \
  --checkpoint artifacts/best_model.pt \
  --input examples/dialogue1.json \
  --output artifacts/dialogue1_predictions.json \
  --device auto
```

Output di esempio:

```json
[
  {
    "trigger_threshold": 0.7500000000000002,
    "predictions": [
      {
        "index": 0,
        "speaker": "A",
        "utterance": "I finally got the promotion!",
        "emotion": "joy",
        "trigger_probability": 0.083826,
        "trigger": 0
      },
      {
        "index": 1,
        "speaker": "B",
        "utterance": "That is amazing, congratulations!",
        "emotion": "joy",
        "trigger_probability": 0.155064,
        "trigger": 0
      },
      {
        "index": 2,
        "speaker": "A",
        "utterance": "Thanks. I was sure they would choose someone else.",
        "emotion": "joy",
        "trigger_probability": 0.881737,
        "trigger": 1
      },
      {
        "index": 3,
        "speaker": "B",
        "utterance": "You earned it. We should celebrate tonight.",
        "emotion": "joy",
        "trigger_probability": 0.900187,
        "trigger": 1
      }
    ]
  }
]
```

## Come leggere l'output

Per esempio:

```json
{
  "emotion": "joy",
  "trigger_probability": 0.881737,
  "trigger": 1
}
```

significa che:

1. il modello ha classificato l'utterance come `joy`;
2. ha stimato una probabilità di circa `0.882` che sia un trigger di cambiamento emotivo;
3. ha assegnato `trigger = 1` perché tale probabilità supera la soglia del modello.

Il campo:

```json
"trigger_threshold": 0.7500000000000002
```

rappresenta la soglia usata per convertire `trigger_probability` nella decisione binaria `trigger`.

La soglia viene selezionata sui dati di validation durante il training e salvata insieme al modello.

---

# Training

La struttura prevista del dataset è:

```text
data/
├── MELD_train_efr.json
├── MELD_val_efr.json
└── MELD_test_efr.json
```

Esempio:

```bash
meld-efr train \
  --train-path data/MELD_train_efr.json \
  --val-path data/MELD_val_efr.json \
  --test-path data/MELD_test_efr.json \
  --output-dir artifacts \
  --epochs 30 \
  --batch-size 4 \
  --grad-accum-steps 2 \
  --encoder-lr 8e-6 \
  --head-lr 7e-5 \
  --dropout 0.5 \
  --weight-decay 0.05 \
  --patience 5 \
  --device auto
```

Durante il training il modello viene valutato sul validation set e il checkpoint migliore viene salvato nella cartella di output.

Le principali metriche sono:

- `emotion_macro_f1`: F1 macro-mediato sulle classi di emozione;
- `trigger_f1+`: F1-score della classe positiva dei trigger.

---

## Principali opzioni di `train`

| Opzione | Descrizione |
|---|---|
| `--train-path` | Percorso del dataset di training. |
| `--val-path` | Percorso del dataset di validation. |
| `--test-path` | Percorso del dataset di test. |
| `--output-dir` | Cartella in cui salvare modello, metriche e configurazioni. |
| `--epochs` | Numero massimo di epoche. |
| `--patience` | Numero di epoche senza miglioramento prima dell'early stopping. |
| `--batch-size` | Numero di dialoghi per batch. |
| `--grad-accum-steps` | Step di accumulo dei gradienti. Utile quando la memoria GPU è limitata. |
| `--encoder-lr` | Learning rate dell'encoder Transformer pre-addestrato. |
| `--head-lr` | Learning rate dei layer contestuali e delle head del task. |
| `--dropout` | Dropout usato per la regolarizzazione. |
| `--weight-decay` | Weight decay usato per la regolarizzazione. |
| `--device` | `auto`, `cpu`, `cuda`, `cuda:0`, ecc. |

Per vedere tutte le opzioni:

```bash
meld-efr train --help
```

---

# File prodotti dal training

La cartella `artifacts/` può contenere:

```text
artifacts/
├── best_model.pt
├── tokenizer/
├── encoder_config/
├── history.json
├── metrics.json
├── training_config.json
└── label_mapping.json
```

In particolare:

- `best_model.pt`: pesi del miglior modello;
- `tokenizer/`: tokenizer necessario all'inference;
- `encoder_config/`: configurazione dell'encoder;
- `history.json`: andamento del training;
- `metrics.json`: metriche finali;
- `training_config.json`: configurazione usata per il training;
- `label_mapping.json`: mapping tra label numeriche ed emozioni.

---

# Uso dopo `git clone`

Una volta pubblicato il repository con `best_model.pt` e gli altri file necessari:

```bash
git clone <REPOSITORY_URL>
cd meld-efr-cli
pip install -e .
```

Poi è possibile eseguire subito:

```bash
meld-efr predict \
  --checkpoint artifacts/best_model.pt \
  --input examples/dialogue1.json \
  --device auto
```

Non serve aprire Colab o altri notebook per fare inference.

> I file di modello di grandi dimensioni, come `.pt`, `.bin` e `.safetensors`, dovrebbero normalmente essere gestiti con **Git LFS** quando il repository viene pubblicato su GitHub.

---

# Quick start

```bash
# 1. Clona il repository
git clone <REPOSITORY_URL>
cd meld-efr-cli

# 2. Installa
pip install -e .

# 3. Esegui una prediction
meld-efr predict \
  --checkpoint artifacts/best_model.pt \
  --input examples/dialogue1.json \
  --output predictions.json \
  --device auto
```

Il workflow finale è quindi:

```text
git clone
    ↓
pip install -e .
    ↓
meld-efr predict ...
    ↓
predizione delle emozioni e dei trigger emotivi
```
