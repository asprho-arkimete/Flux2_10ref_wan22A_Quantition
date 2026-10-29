
import asyncio
from logging import exception
import math
import sys
import tkinter as tk
from tkinter import HORIZONTAL, Label, ttk
from tkinterdnd2 import TkinterDnD, DND_FILES

window = TkinterDnD.Tk()
window.title("Real pose")
window.geometry("1300x700")
window.resizable(False, False)

frame_canvas = tk.Frame(window)
frame_canvas.grid(row=0, column=0, padx=10, pady=10, sticky='nw')

paths_referenze = [None] * 10   # indice 0 = posa, 1..9 = riferimenti piccoli

from tkinter import filedialog
from PIL import Image, ImageTk

def drag_drop(event, canvas, index_canvas):
    global paths_referenze
    path = window.tk.splitlist(event.data)[0]   # gestisce path con spazi
    paths_referenze[index_canvas] = path
    img = Image.open(path).convert('RGB')
    w, h = img.size
    rw = rh = big_canvas_size
    if index_canvas > 0:
        rw = rh = small_canvas_size

    if w > h:
        rh = (rw * h) // w
    else:
        rw = (rh * w) // h
    img = img.resize((rw, rh), Image.BICUBIC)
    photo = ImageTk.PhotoImage(img)
    canvas.delete('all')
    canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2, image=photo)
    canvas.image = photo   # evita che l'immagine venga eliminata dal garbage collector
    canvas.update()

    print(f"paths riferimenti: {paths_referenze}")


def f_delete_riferimento(canvas, indice, testo, font):
    global paths_referenze
    paths_referenze[indice]=None
    canvas.delete('all')
    w=int(canvas['width'])
    h=int(canvas['height'])
    canvas.create_text((w//2, h//2), text=testo, font=font, fill='white', justify='center')

# Canvas grande per riferimento posa
big_canvas_size = 400
ref1 = tk.Canvas(frame_canvas, width=big_canvas_size, height=big_canvas_size, bg='red')
ref1.create_text((big_canvas_size // 2, big_canvas_size // 2), text='Riferimento \nPersonaggio', font=("Arial", 24), fill='white')
ref1.grid(row=0, column=0, rowspan=2, sticky='nw', padx=5, pady=5)
ref1.drop_target_register(DND_FILES)
ref1.dnd_bind('<<Drop>>', lambda e: drag_drop(e, ref1, 0))
ref1.bind('<Double-Button-1>', lambda e: f_delete_riferimento(ref1, 0, 'Riferimento \nPersonaggio', ("Arial", 24)))

# Frame per le 9 canvas piccole
frame_canvas2 = tk.Frame(frame_canvas)
frame_canvas2.grid(row=0, column=1, padx=10, pady=10, sticky='nw')

small_canvas_size = 128
small_canvases = []
background_colors = ['light blue', 'pink', 'green', 'orange', 'black', 'grey', 'light green', 'light salmon', 'sky blue']
for i in range(9):
    riga = i // 3
    colonna = i % 3
    testo = 'riferimento 2\nposa / altro' if i == 0 else f'riferimento {i+2}'
    ref = tk.Canvas(frame_canvas2, width=small_canvas_size, height=small_canvas_size, bg=background_colors[i % len(background_colors)])
    ref.create_text((small_canvas_size // 2, small_canvas_size // 2),
                   text=testo, font=("Arial", 12), fill='white', justify='center')
    ref.grid(row=riga, column=colonna, sticky='nw', padx=4, pady=4)
    ref.drop_target_register(DND_FILES)
    ref.dnd_bind('<<Drop>>', lambda e, c=ref, n=i+1: drag_drop(e, c, n))
    ref.bind('<Double-Button-1>', lambda e, c=ref, n=i+1, t=testo: f_delete_riferimento(c, n, t, ("Arial", 12)))
    small_canvases.append(ref)

# Frame per output
frame_out = tk.Frame(frame_canvas)
frame_out.grid(row=0, column=2, padx=10, pady=10, sticky='nw')
out = tk.Canvas(frame_out, width=400, height=400, bg='dark green')
out.create_text(
    (200, 200),
    text="VISUAL OUTPUT",
    font=("Arial", 10),
    fill='white'
)
out.grid(row=0, column=0)


frame_button = tk.Frame(window)
frame_button.grid(row=2, column=0, sticky='nw', padx=5)

import os
import webbrowser

def f_daz():
        path_daz = os.path.abspath("./renderposeDaz.duf")
        try:
            os.startfile(path_daz)
            print("avvio file daz 3d")
        except exception as error:
            print(f"daz3d non trovato: {error} scarica daz 3d da: https://www.daz3d.com/get_studio")
            webbrowser.open("https://www.daz3d.com/get_studio")
        finally:
            pass
   
daz3d = tk.Button(frame_button, text='Pose DAZ 3D', bg='light green', command=f_daz)
daz3d.grid(row=0, column=0, sticky='nw', padx=5)

lab_step=tk.Label(frame_button,text='STEPS')
lab_step.grid(row=0,column=1,padx=5,sticky='nw')

steps=tk.Scale(frame_button,from_=1, to=50,orient=HORIZONTAL)
steps.set(8)
steps.grid(row=1,column=1,padx=5,sticky='nw') 

import os
import gc
from googletrans import Translator
def traduci(testo_ita):
    translator = Translator()
    risultato = translator.translate(testo_ita, src='it', dest='en')
    print(risultato.text)
    return risultato.text


from diffusers.utils import logging as dlog
dlog.set_verbosity_error()
import torch
from diffusers import Flux2KleinPipeline
from optimum.quanto import freeze, qfloat8, quantize
import random

pipe=None
old_lora1=None
old_lora2=None
old_paths_referenze = None
old_images=[]

def flux2():
    global pipe,old_lora1,old_lora2,old_paths_referenze,old_images,out
    def f_resize(i,rw,rh):
        img=Image.open(paths_referenze[i]).convert("RGB")
        w,h=img.size
        if w>=h:
            rh=(rw*h)//w
        else:
            rw=(rh*w)//h
        img=img.resize((rw,rh),Image.BICUBIC)
        return img

    print("genera image")
    global paths_referenze,text,steps,lora1,lora2,scalelora1,scalelora2
    os.makedirs("./clips",exist_ok=True)
    dtype = torch.bfloat16
    #gestisci le images di riferimento
    if old_paths_referenze!=paths_referenze:
        images=[]
        for i in range(0,len(paths_referenze)):
            print(f"referimento {i}: {paths_referenze[i]}")
            if paths_referenze[i] is not None and os.path.exists(paths_referenze[i]): 
                images.append(f_resize(i,512,512))
        if len(images)>4:
            #crea collage
            cw=2000//4
            righe = math.ceil(len(images)/4)
            collage=Image.new('RGB',(2000,righe*cw),'white')
            n=0
            for i in range(0,len(paths_referenze)):
                if paths_referenze[i] is None or not os.path.exists(paths_referenze[i]):
                    continue
                img=Image.open(paths_referenze[i]).convert("RGB")
                w,h=img.size
                rw,rh=cw,cw
                if w>=h:
                    rh=(rw*h)//w
                else:
                    rw=(rh*w)//h
                img=img.resize((rw,rh),Image.BICUBIC)
                posx=(n%4)*cw
                posy=(n//4)*cw
                collage.paste(img,(posx,posy))
                n=n+1
            images=[collage]
        old_paths_referenze=list(paths_referenze)
        old_images=images
    images=old_images

    #ricrea la pipeline solo se è la prima volta o se cambia una lora
    if pipe is None or old_lora1!=lora1.get() or old_lora2!=lora2.get():
        pipe=None
        gc.collect()
        torch.cuda.empty_cache()
        pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-9B", dtype=dtype)
        adapters=[]
        pesi=[]
        #lora 1
        nome_lora1=lora1.get()
        if nome_lora1 in azioni and azioni.index(nome_lora1)<len(paths_loras1):
            nome_lora1=paths_loras1[azioni.index(nome_lora1)]
        path_lora1= f"./lora//{nome_lora1}"
        if not ".safetensors" in path_lora1:
            path_lora1=path_lora1+".safetensors"
        if os.path.exists(path_lora1):
            print(f"Lora1: {path_lora1}")
            nome_ad1=os.path.basename(path_lora1).split('.')[0]
            pipe.load_lora_weights(path_lora1,adapter_name=nome_ad1)
            adapters.append(nome_ad1)
            pesi.append(float(scalelora1.get()))

        #lora 2
        path_lora2= f"./lora//{lora2.get()}"
        if not ".safetensors" in path_lora2:
            path_lora2=path_lora2+".safetensors"
        if os.path.exists(path_lora2):
            print(f"Lora2: {path_lora2}")
            nome_ad2=os.path.basename(path_lora2).split('.')[0]
            pipe.load_lora_weights(path_lora2,adapter_name=nome_ad2)
            adapters.append(nome_ad2)
            pesi.append(float(scalelora2.get()))

        if adapters:
            pipe.set_adapters(adapters,adapter_weights=pesi)

        quantize(pipe.transformer, weights=qfloat8)
        freeze(pipe.transformer)
        quantize(pipe.text_encoder, weights=qfloat8)
        freeze(pipe.text_encoder)

        pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
        old_lora1=lora1.get()
        old_lora2=lora2.get()

    prompt =traduci(text.get('1.0',tk.END))
    image_riferimento=None
    
    if len(images)==0:
        image_riferimento=None
    elif len(images)==1:
        image_riferimento= images
    else:
        image_riferimento= list(images) 
    print(f"Numero imagini di riferimento: {len(images)}")

    seme=random.randint(0,2**32-1)
    print(f"seed: {seme}")
    image = pipe(
        prompt=prompt,
        image=image_riferimento,
        height=1024,
        width=1024,
        guidance_scale=1.0,
        num_inference_steps=int(steps.get()),
        generator=torch.Generator('cpu').manual_seed(seme)
    ).images[0]

    # Resize image for display
    image_resized = image.resize((400, 400))
    # Convert PIL Image to ImageTk.PhotoImage
    photo = ImageTk.PhotoImage(image_resized)
    # Clear the canvas before displaying new image
    out.delete('all')
    out.create_image((200,200), image=photo)
    # Keep reference to avoid garbage collection
    out.image = photo
    out.update_idletasks()

    # Ensure output directory exists
    output_dir = "./clips"
    os.makedirs(output_dir,exist_ok=True)
    path_out = os.path.join(output_dir, "flux-klein.png")
    k = 1
    while os.path.exists(path_out):
        path_out = os.path.join(output_dir, f"flux-klein_{k}.png")
        k += 1
    image.save(path_out)
    
import threading as T
def run_flux2():
    T.Thread(target=flux2,daemon=True).start()

genera = tk.Button(frame_button, text='Genera Image', bg='green', command=run_flux2)
genera.grid(row=0, column=2, sticky='nw', padx=5)

import json
import os

prompts=[]
paths_loras1=[]
azioni=[]

def loadlora(event=None):
    global lora1,azioni,paths_loras1,prompts,lora2
    azioni.clear()
    prompts.clear()
    paths_loras1.clear()
    with open("presets_prompts.json", encoding='utf-8') as f:
        file=json.load(f)
    altri_loras2=[]
    altri_loras2.append('nolora')
    for dizionario in file:
        azioni.append(dizionario['azione'])
        prompts.append(dizionario['prompt'])
        paths_loras1.append(dizionario['nome_lora1'])
    
    for altri_lora in os.listdir('lora'):
        if altri_lora.lower().replace('.safetensors','') in paths_loras1:
            pass
        else:
            azioni.append(altri_lora)
        altri_loras2.append(altri_lora)
    lora1['values']=azioni
    lora2['values']=altri_loras2


lab_lora1=tk.Label(frame_button,text='Lora 1')
lab_lora1.grid(row=0,column=3,sticky='nw', padx=5)
lora1=ttk.Combobox(frame_button,values=[])
lora1.grid(row=0,column=4,sticky='nw', padx=5)
lora1.set('nolora')
lora1.bind('<Button-1>',loadlora)

scalelora1=tk.Scale(frame_button,from_=0.10,to=2.00,orient=HORIZONTAL,resolution=0.10)
scalelora1.set(1.0)
scalelora1.grid(row=1,column=4)


def f_selezione(event=None):
    global lora1, prompts, text
    if lora1.current() >= 0 and lora1.current() < len(prompts):
        text.delete('1.0', tk.END)
        text.insert('1.0', prompts[lora1.current()])
    else:
        text.delete('1.0', tk.END)
        


lora1.bind('<<ComboboxSelected>>', f_selezione)


lab_lora2=tk.Label(frame_button,text='Lora 2')
lab_lora2.grid(row=0,column=5,sticky='nw', padx=5)
lora2=ttk.Combobox(frame_button,values=[])
lora2.grid(row=0,column=6,sticky='nw', padx=5)
lora2.set('nolora')
lora2.bind('<Button-1>',loadlora)

scalelora2=tk.Scale(frame_button,from_=0.10,to=2.00,orient=HORIZONTAL,resolution=0.10)
scalelora2.set(0.8)
scalelora2.grid(row=1,column=5)

loadlora()


def f_wan_run():
    btn_wan22a.config(state='disabled')

    def run():
        try:
            os.system(f'"{sys.executable}" wan22a.py')
        finally:
            window.after(0, lambda: btn_wan22a.config(state='normal'))

    T.Thread(target=run, daemon=True).start()
    
btn_wan22a = tk.Button(frame_button, text="Wan 2.2 A", bg='light blue', command=f_wan_run)
btn_wan22a.grid(row=0, column=7, sticky='nw', padx=5)

frame_text=tk.Frame(window,bg='light gray')
frame_text.grid(row=3, column=0, sticky='nw')
text = tk.Text(frame_text, width=80,height=8)
text.grid(row=1, column=0, sticky='nw', padx=5,pady=5)

window.mainloop()