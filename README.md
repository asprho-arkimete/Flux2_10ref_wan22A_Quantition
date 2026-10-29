Crea immagini catturando la posa del modello 3d in daz3d;
sul qualunque personaggio con Flux 2 e lora: Flux.2-Klein-9B-MatchingPose;
Animala con Wan 22A Quantitizzato con low ram 16GB
Software Istalling:
Anaconda:https://cursor.com/
cursor:https://cursor.com/
python 3.10:https://www.python.org/downloads/release/python-3100/
Daz 3d: https://www.daz3d.com/
download lora cattura: https://huggingface.co/nhathoangfoto/Flux.2-Klein-9B-MatchingPose
mettilo nella cartella 'lora'
clona repo: git clone: https://github.com/asprho-arkimete/Flux2_10ref_wan22A_Quantition.git

crea area virtuale: python 3.10 -m venv vrealpose
cd vrealpose\script
activate cd.. cd..

download altri lora:(mega lora flux2): https://huggingface.co/Asprho/megalora/tree/main

per wan22A scarica i lora che preferisci da civitai e civitai_red
https://civitai.com/
https://civitai.red/
con filtro modello Wan2.2 14B A e metti tutti i lora nella cartella lora wan

*Librerie*
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu134
pip install -r requisiti.txt

cartelle da creare necessarie: 
mkdir:[.\lora; .\lora wan .\render pose .\clips]

run python realpose.py

