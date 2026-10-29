# Flux2 10ref + Wan 2.2 A (quantizzato)

Crea immagini catturando la posa di un modello 3D in **Daz 3D** e applicandola a qualunque personaggio con **Flux 2** e la LoRA [Flux.2-Klein-9B-MatchingPose](https://huggingface.co/nhathoangfoto/Flux.2-Klein-9B-MatchingPose).
Poi animale con **Wan 2.2 A quantizzato**, ottimizzato per girare con soli **16 GB di RAM**.

## Software necessario

| Software | Link |
|---|---|
| Anaconda | https://www.anaconda.com/download |
| Cursor (editor, opzionale) | https://cursor.com/ |
| Python 3.10 | https://www.python.org/downloads/release/python-3100/ |
| Daz 3D | https://www.daz3d.com/ |
| Git | https://git-scm.com/downloads |

## Installazione

### 1. Clona il repository

```bat
git clone https://github.com/asprho-arkimete/Flux2_10ref_wan22A_Quantition.git
cd Flux2_10ref_wan22A_Quantition
```

### 2. Crea e attiva l'ambiente virtuale

```bat
py -3.10 -m venv vrealpose
vrealpose\Scripts\activate
```

### 3. Installa le librerie

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -r requisiti.txt
```

> Se la tua GPU o i tuoi driver non supportano CUDA 13.2, scegli l'indice PyTorch adatto (es. `cu128`) su https://pytorch.org/get-started/locally/

### 4. Crea le cartelle necessarie

```bat
mkdir lora
mkdir "lora wan"
mkdir "render pose"
mkdir clips
```

### 5. Scarica le LoRA

- **Cattura della posa (obbligatoria):** scarica [Flux.2-Klein-9B-MatchingPose](https://huggingface.co/nhathoangfoto/Flux.2-Klein-9B-MatchingPose) e mettila nella cartella `lora`.
- **Altre LoRA per Flux 2 (opzionale):** [Mega LoRA Flux2](https://huggingface.co/Asprho/megalora/tree/main) nella cartella `lora`.
- **LoRA per Wan 2.2 A:** scaricale da [Civitai](https://civitai.com/) o [Civitai Red](https://civitai.red/) filtrando per modello **Wan 2.2 14B A**, e mettile tutte nella cartella `lora wan`.

## Avvio

```bat
python realpose.py
```

## Struttura delle cartelle

| Cartella | Contenuto |
|---|---|
| `lora` | LoRA per Flux 2 |
| `lora wan` | LoRA per Wan 2.2 A |
| `render pose` | Immagini generate dalla posa catturata |
| `clips` | Video animati generati |
