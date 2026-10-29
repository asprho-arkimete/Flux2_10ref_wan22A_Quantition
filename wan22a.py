import os
import re
import random
import torch
import numpy as np
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, BitsAndBytesConfig
from diffusers.utils import export_to_video, load_image
from optimum.quanto import freeze, qfloat8, quantize
import tkinter as tk
from tkinter import HORIZONTAL, VERTICAL, ttk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
from moviepy import VideoFileClip, concatenate_videoclips
from translate import Translator

dir_lora = './lora wan'
path_referimento = None

translator = Translator(to_lang="en", from_lang="it")


def traduci(testo, max_len=300):
    frasi = re.split(r'(?<=[.!?])\s+', testo)
    blocchi, corrente = [], ''
    for f in frasi:
        if len(corrente) + len(f) + 1 <= max_len:
            corrente = (corrente + ' ' + f).strip()
        else:
            if corrente:
                blocchi.append(corrente)
            corrente = f
    if corrente:
        blocchi.append(corrente)
    return ' '.join(translator.translate(b) for b in blocchi)


def estrai_secondi(widget, default=1):
    """Legge un numero da un widget con testo tipo '5 secondi'."""
    m = re.search(r'\d+', str(widget.get()))
    return int(m.group()) if m else default

def estrai_numero(widget, default=16):
    """Legge un numero (anche decimale) da un widget, es. '16' o '16 fps'."""
    m = re.search(r'\d+(?:[.,]\d+)?', str(widget.get()))
    return float(m.group().replace(',', '.')) if m else default


pipe = None
old_lora = 'nolora'
old_riferimento = ''
image = None
old_prompt = ''
height = None
width = None
traduzione = ''

from tqdm import tqdm

def wan22a():
    global lora, prompt, path_referimento, steps, translator, FPS
    global pipe, old_lora, image, old_prompt, old_riferimento, height, width, traduzione
    global clips_totale, clip

    if path_referimento is None:
        print("Trascina prima un'immagine nel canvas")
        return

    model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    dtype = torch.bfloat16

    # ---------- Caricamento pipeline (una sola volta) ----------
    if pipe is None:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", quantization_config=bnb,
            dtype=torch.bfloat16,
        )
        transformer_2 = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer_2", quantization_config=bnb,
            dtype=torch.bfloat16,
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id, transformer=transformer, transformer_2=transformer_2, dtype=dtype
        )
        quantize(pipe.text_encoder, weights=qfloat8)
        freeze(pipe.text_encoder)
        print("transformer, text_encoder quantizzati")

        pipe.enable_model_cpu_offload()

        def libera_gpu(module, args, kwargs):
            for nome in ("transformer_2", "vae"):
                comp = getattr(pipe, nome)
                if next(comp.parameters()).device.type == "cuda":
                    comp.to("cpu")
            torch.cuda.empty_cache()

        pipe.transformer.register_forward_pre_hook(libera_gpu, with_kwargs=True)
        pipe.vae.enable_tiling()

    # ---------- LoRA ----------
    nome_lora = lora.get()
    if old_lora != nome_lora:
        if old_lora != 'nolora':
            pipe.unload_lora_weights()
        path_lora = os.path.join(dir_lora, nome_lora)
        if not path_lora.endswith(".safetensors"):
            path_lora += ".safetensors"
        if os.path.exists(path_lora):
            try:
                nome = os.path.basename(path_lora).split('.')[0]
                pipe.load_lora_weights(path_lora, adapter_name=nome)
                pipe.set_adapters(nome, adapter_weights=[1.0])
                print(f"lora: {nome} caricato")
            except Exception as error:
                print(f"Errore: {error}")
        old_lora = nome_lora

    # ---------- Immagine di riferimento ----------
    if old_riferimento != path_referimento:
        img = Image.open(path_referimento).convert("RGB")
        max_area = 256 * 360
        aspect_ratio = img.height / img.width
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = img.resize((width, height), Image.BICUBIC)
        old_riferimento = path_referimento

    # ---------- Prompt ----------
    prompt_ita = prompt.get('1.0', tk.END).strip()
    if prompt_ita == '':
        prompt_ita = "Anima questa immagine facendo muovere i soggetti nel contesto rappresentato"

    if old_prompt != prompt_ita:
        print(f"Prompt ITA: {prompt_ita}")
        traduzione = traduci(prompt_ita)
        print(f"Prompt ENG: {traduzione}")
        old_prompt = prompt_ita

    negative_prompt = (
        "bad anatomy, deformed body, distorted limbs, extra legs, extra arms, missing limbs, "
        "extra fingers, missing fingers, fused fingers, malformed hands, malformed feet, extra toes, "
        "deformed tongue, deformed mouth, deformed face, asymmetrical face, distorted eyes, "
        "disfigured, mutated, unnatural proportions, long neck"
    )
    generator = torch.Generator(device='cpu').manual_seed(random.randint(1, 2**31 - 1))

    # ---------- Durate e numero di clip ----------
    fps_val = int(round(estrai_numero(FPS, default=16)))   # FPS come numero
    durata_clip = max(1, estrai_secondi(clip))
    durata_totale = max(durata_clip, estrai_secondi(clips_totale))
    numero_clips = max(1, durata_totale // durata_clip)

    # Wan richiede num_frames = 4k + 1
    frames = int(fps_val * durata_clip)
    frames = (frames // 4) * 4 + 1

    # ---------- Generazione ----------
    clips_files = []
    current_image = image
    pipe.set_progress_bar_config(leave=False)   # la barra degli step sparisce a fine clip

    barra = tqdm(range(1, numero_clips + 1), desc='Clips totali')
    for k in barra:
        barra.set_postfix_str(f"clip {k}/{numero_clips}")
        output = pipe(
            image=current_image,
            prompt=traduzione,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=frames,
            guidance_scale=3.5,
            num_inference_steps=int(float(steps.get())),
            generator=generator,
        ).frames[0]

        # dalla seconda clip in poi scarta il primo frame (è l'ultimo della clip precedente)
        da_esportare = output if k == 1 else output[1:]

        nome_file = f"clips_1{k}.mp4"
        export_to_video(da_esportare, nome_file, fps=fps_val)
        clips_files.append(nome_file)

        # L'ultimo frame diventa l'immagine di partenza della clip successiva
        ultimo = output[-1]
        if isinstance(ultimo, np.ndarray):
            if ultimo.dtype != np.uint8:
                ultimo = (ultimo * 255).clip(0, 255).astype(np.uint8)
            ultimo = Image.fromarray(ultimo)
        current_image = ultimo

    # ---------- Unione ----------
    clips_open = [VideoFileClip(c) for c in clips_files]
    video = concatenate_videoclips(clips_open)
    cartella_out = "clips"
    os.makedirs(cartella_out, exist_ok=True)

    out_path = os.path.join(cartella_out, f"clips completa {nome_lora}.mp4")
    j = 2
    while os.path.exists(out_path):
        out_path = os.path.join(cartella_out, f"clips completa {nome_lora}_{j}.mp4")
        j += 1

    video.write_videofile(out_path, fps=fps_val)
    print(f"Video salvato in: {out_path}")
    video.close()
    for c in clips_open:
        c.close()
    for c in clips_files:
        try:
            os.remove(c)
        except OSError as e:
            print(f"Impossibile eliminare {c}: {e}")






window = TkinterDnD.Tk()
window.geometry("1024x800")
window.title("WAN 2.2A Image To Video")
window.config(background='gray')
window.resizable(False,False)

# --- Anteprima first frame ---
frame_canvas = tk.Frame(window, bg='gray')
frame_canvas.grid(row=0, column=0, sticky='nw', padx=10, pady=(10, 5))

def f_dragedrop(event):
    global path_referimento
    path = window.tk.splitlist(event.data)[0]
    if not path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
        return
    path_referimento = path
    print("Caricamento image", path_referimento)
    img = Image.open(path_referimento).convert('RGB')
    w,h=img.size
    rw,rh=300,300
    
    if w>=h:
        rh=(rw*h)//w
    else:
        rw=(rh*w)//h
    img= img.resize((rw,rh),Image.BICUBIC)
    
    foto = ImageTk.PhotoImage(img)
    frame.delete('all')
    frame.create_image(150, 150, image=foto)
    frame.image = foto

frame = tk.Canvas(frame_canvas, width=300, height=300, bg='red', highlightthickness=0)
frame.create_text(150, 150, text='first frame\nVideo', font=("Arial", 15), fill='white', justify='center')
frame.grid(row=0, column=0, sticky='nw')
frame.drop_target_register(DND_FILES)
frame.dnd_bind('<<Drop>>', f_dragedrop)

# --- Strumenti ---
frame_strumenti = tk.Frame(window, bg='gray')
frame_strumenti.grid(row=1, column=0, sticky='nw', padx=10)

# Steps + Genera sulla stessa riga
frame_steps = tk.Frame(frame_strumenti, bg='gray')
frame_steps.grid(row=0, column=0, sticky='nw', pady=5)

lab_stp = tk.Label(frame_steps, text='Steps: 8', bg='gray')
lab_stp.grid(row=0, column=0, columnspan=2, sticky='nw', pady=(0, 3))

def f_step(val=None):
    global steps, lab_stp
    lab_stp.config(text=f'Steps: {int(float(steps.get()))}')

steps = ttk.Scale(frame_steps, from_=1, to=50, orient=HORIZONTAL, length=150, command=f_step)
steps.grid(row=1, column=0, sticky='nw', padx=(0, 10))
steps.set(8)

import threading as T
def run_wan():
    genera.config(state='disabled')

    def lavoro():
        try:
            wan22a()
        finally:
            window.after(0, lambda: (genera.config(state='normal'), load_clips()))

    T.Thread(target=lavoro, daemon=True).start()

genera = tk.Button(frame_steps, text='Genera', bg='light green',command=run_wan)
genera.grid(row=1, column=1, sticky='nw')

# Prompt
lab_prompt = tk.Label(frame_strumenti, text='Prompt', bg='gray')
lab_prompt.grid(row=1, column=0, sticky='nw', pady=(10, 0))

prompt = tk.Text(frame_strumenti, width=60, height=6, wrap='word')
prompt.grid(row=2, column=0, sticky='nw', pady=5)

frame_strumenti2=tk.Frame(window,bg='gray')
frame_strumenti2.grid(row=2,column=0,sticky='nw')

# LoRA
lab_lora = tk.Label(frame_strumenti2, text='LoRA', bg='gray')
lab_lora.grid(row=3, column=0, sticky='nw', pady=5,padx=5)

def load_lora(event=None):
    global lora
    models = ['nolora']
    if os.path.isdir(dir_lora):
        models += [os.path.splitext(l)[0] for l in os.listdir(dir_lora) if l.lower().endswith('.safetensors')]
    lora['values'] = models
    lora.update_idletasks()

lora = ttk.Combobox(frame_strumenti2, values=[], state='readonly', width=10)
lora.grid(row=4, column=0, sticky='nw', pady=5,padx=5)
lora.bind('<Button-1>', load_lora)
lora.set('nolora')
load_lora()

lab_clip = tk.Label(frame_strumenti2, text='Durata Clip', bg='gray')
lab_clip.grid(row=3, column=1, sticky='nw', pady=(10, 0))
clip = ttk.Combobox(frame_strumenti2, values=['1 secondo','2 secondi','3 secondi','4 secondi','5 secondi'], state='readonly', width=10)
clip.grid(row=4, column=1, sticky='nw', pady=5,padx=5)
clip.set('2 secondi')

def f_second(event=None):
    global clips_totale
    clips_totals=[]
    for s in range(1,20):
        if s==1:
            clips_totals.append(f'{s} secondo')
        else:
            clips_totals.append(f'{s} secondi')
    clips_totale['values']=clips_totals
    clips_totale.update_idletasks()

lab_clip_totale = tk.Label(frame_strumenti2, text='Durata Totale', bg='gray')
lab_clip_totale.grid(row=3, column=2, sticky='nw', pady=5,padx=5)
clips_totale = ttk.Combobox(frame_strumenti2, values=[], state='readonly', width=10)
clips_totale.grid(row=4, column=2,sticky='nw', pady=5,padx=5)
clips_totale.bind('<Button-1>',f_second)
clips_totale.set('5 secondi')

f_second()

lab_FPS = tk.Label(frame_strumenti2, text='Durata Totale', bg='gray')
lab_FPS.grid(row=3, column=3, sticky='nw', pady=5,padx=5)
FPS = ttk.Combobox(frame_strumenti2, values=['8','16','24'], state='readonly', width=10)
FPS.grid(row=4, column=3,sticky='nw', pady=5,padx=5)
FPS.set(16)

CARTELLA_CLIPS = "clips"
ESTENSIONI_VIDEO = ('.mp4', '.mkv', '.mov')


def load_clips():
    listaclips.delete(0, tk.END)
    if not os.path.isdir(CARTELLA_CLIPS):
        return
    for nome_file in sorted(os.listdir(CARTELLA_CLIPS)):
        if nome_file.lower().endswith(ESTENSIONI_VIDEO):
            listaclips.insert(tk.END, nome_file)


frame_list = tk.Frame(window, bg='gray')
frame_list.grid(row=0, column=1, sticky='nw')

aggiorna = tk.Button(frame_list, text='Aggiorna', command=load_clips)
aggiorna.grid(row=0, column=0, sticky='nw', padx=5, pady=5)

# Contenitore per listbox + scrollbar
frame_box = tk.Frame(frame_list, bg='gray')
frame_box.grid(row=1, column=0, sticky='nw', padx=5, pady=5)

listaclips = tk.Listbox(frame_box, width=30, height=25)
scroll_y = ttk.Scrollbar(frame_box, orient=VERTICAL, command=listaclips.yview)
scroll_x = ttk.Scrollbar(frame_box, orient=HORIZONTAL, command=listaclips.xview)
listaclips.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

listaclips.grid(row=0, column=0, sticky='nsew')
scroll_y.grid(row=0, column=1, sticky='ns')
scroll_x.grid(row=1, column=0, sticky='ew')

load_clips()

def _indice_selezionato():
    sel = listaclips.curselection()
    return sel[0] if sel else None


def _seleziona(indice):
    listaclips.selection_clear(0, tk.END)
    listaclips.selection_set(indice)
    listaclips.activate(indice)
    listaclips.see(indice)


def elimina_clip():
    i = _indice_selezionato()
    if i is None:
        return
    listaclips.delete(i)
    if listaclips.size() > 0:
        _seleziona(min(i, listaclips.size() - 1))   # seleziona l'elemento successivo


def sposta_su():
    i = _indice_selezionato()
    if i is None or i == 0:
        return
    testo = listaclips.get(i)
    listaclips.delete(i)
    listaclips.insert(i - 1, testo)
    _seleziona(i - 1)


def sposta_giu():
    i = _indice_selezionato()
    if i is None or i >= listaclips.size() - 1:
        return
    testo = listaclips.get(i)
    listaclips.delete(i)
    listaclips.insert(i + 1, testo)
    _seleziona(i + 1)


import threading
from PIL import Image, ImageTk

video_corrente = None   # clip aperta in anteprima
nome_corrente = None
foto_canvas = None      # riferimento all'immagine, evita che sparisca dal canvas


def f_render():
    nomi = listaclips.get(0, tk.END)
    if not nomi:
        print("La lista è vuota")
        return

    video_clips = []
    video_completo = None
    try:
        for nome in nomi:
            video_clips.append(VideoFileClip(os.path.join(CARTELLA_CLIPS, nome)))
        video_completo = concatenate_videoclips(video_clips, method="compose")

        path_out = "./video_completo.mp4"
        j = 2
        while os.path.exists(path_out):
            path_out = f"./video_completo_{j}.mp4"
            j += 1

        try:
            # codifica con la GPU NVIDIA (serve ffmpeg con supporto NVENC)
            video_completo.write_videofile(path_out, codec='h264_nvenc', audio=False)
        except Exception as e:
            print(f"NVENC non disponibile ({e}), uso la CPU")
            video_completo.write_videofile(path_out, codec='libx264', audio=False)
        print(f"Video salvato in: {path_out}")
    finally:
        if video_completo is not None:
            video_completo.close()
        for c in video_clips:
            c.close()


def avvia_render():
    threading.Thread(target=f_render, daemon=True).start()


def carica_clip_selezionata(event=None):
    """Apre la clip selezionata e adatta la scala al suo numero di frame."""
    global video_corrente, nome_corrente
    sel = listaclips.curselection()
    if not sel:
        return
    nome = listaclips.get(sel[0])
    if nome == nome_corrente:
        return
    if video_corrente is not None:
        video_corrente.close()
    video_corrente = VideoFileClip(os.path.join(CARTELLA_CLIPS, nome))
    nome_corrente = nome
    n_frames = int(video_corrente.duration * video_corrente.fps)
    scorriframes.configure(from_=0, to=max(1, n_frames - 1))
    scorriframes.set(0)
    f_scorrif()


def frame_corrente():
    """Restituisce (immagine PIL, indice frame) scelto con la scala."""
    if video_corrente is None:
        return None, None
    idx = int(round(float(scorriframes.get())))
    t = idx / video_corrente.fps
    t = max(0, min(t, video_corrente.duration - 1 / video_corrente.fps))
    return Image.fromarray(video_corrente.get_frame(t)), idx


def estrai_frame():
    img, idx = frame_corrente()
    if img is None:
        print("Seleziona prima una clip nella lista")
        return
    out = os.path.join(CARTELLA_CLIPS, f"{os.path.splitext(nome_corrente)[0]}_f{idx}.png")
    img.save(out)
    print(f"Fotogramma salvato in: {out}")


def f_scorrif(valore=None):
    """Mostra nel canvas il fotogramma scelto con la scala."""
    global foto_canvas,LAB_scorriframes,estrai_fotogramma
    img, idx = frame_corrente()
    if img is None:
        return
    w, h = img.size
    scala = min(300 / w, 300 / h)
    img = img.resize((max(1, int(w * scala)), max(1, int(h * scala))), Image.BICUBIC)
    foto_canvas = ImageTk.PhotoImage(img)
    frame.delete('all')
    frame.create_image(150, 150, image=foto_canvas)
    LAB_scorriframes.config(text=f'frame: {int(scorriframes.get())}')
    estrai_fotogramma.config(text=f'Estrai fotogramma: {int(scorriframes.get())}')
    LAB_scorriframes.update_idletasks()
    estrai_fotogramma.update_idletasks()


frame_button_list = tk.Frame(frame_list, bg='gray')
frame_button_list.grid(row=1, column=1, sticky='nw', padx=5, pady=5)

elimina = tk.Button(frame_button_list, text='Elimina', font=("Arial", 10), bg='red', fg='white',
                    command=elimina_clip)
elimina.grid(row=0, column=0, sticky='nw', padx=5, pady=10)

Su = tk.Button(frame_button_list, text='Su', font=("Arial", 10), bg='gray', fg='black',
               command=sposta_su)
Su.grid(row=1, column=0, sticky='nw', padx=5, pady=10)

Giu = tk.Button(frame_button_list, text='Giu', font=("Arial", 10), bg='dark gray', fg='white',
                command=sposta_giu)
Giu.grid(row=2, column=0, sticky='nw', padx=5, pady=10)

Rendering = tk.Button(frame_button_list, text='Rendering', font=("Arial", 10), bg='light green', fg='black',
                      command=avvia_render)
Rendering.grid(row=3, column=0, sticky='nw', padx=5, pady=10)

estrai_fotogramma = tk.Button(frame_button_list, text='Estrai fotogramma', font=("Arial", 10),
                              bg='dark gray', fg='white', command=estrai_frame)
estrai_fotogramma.grid(row=4, column=0, sticky='nw', padx=5, pady=10)

scorriframes = ttk.Scale(frame_button_list, from_=0, to=1, command=f_scorrif)
scorriframes.grid(row=5, column=0, sticky='nw', padx=5, pady=10)

LAB_scorriframes = ttk.Label(frame_button_list,text=f'frame: 1')
LAB_scorriframes.grid(row=5, column=1, sticky='nw', padx=5, pady=10)



# quando selezioni una clip nella lista, si apre l'anteprima
listaclips.bind('<<ListboxSelect>>', carica_clip_selezionata)


window.mainloop()