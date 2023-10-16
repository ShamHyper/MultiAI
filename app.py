import shutil
import os
from PIL import Image
import numpy as np
import gradio as gr
from rembg import remove
from local_eng import *

current_directory = os.path.dirname(os.path.abspath(__file__))

def rem_bg_def(inputs): 
    pics = 0
    try:
        pics += 1
        print(f"[{loc_log}|{loc_pic} #{pics}]{loc_st_del}")

        outputs = remove(inputs)

        print(f"[{loc_log}|{loc_pic} #{pics}]{loc_del}")

    except PermissionError:
        pics -= 1
        print(f"[{loc_err}|{loc_pic} #{pics}]{loc_perm_err}")
        pass
    except FileNotFoundError:
        pics -= 1
        print(f"[{loc_err}|{loc_pic} #{pics}]{loc_nofile_err}")
        pass
    if pics == 0: 
        print(f"[{loc_err}]{loc_need_pic}")
    if pics > 0:
        print(f"{loc_count_del}{pics}.")
    if pics < 0:
        print(f"[{loc_err}]{loc_contact}")
    return outputs

def rem_bg_def_batch(inputs, outputs): 
    temp_dir = inputs
    pics = 0
    for filename in os.listdir(inputs):
        outputs = "rembg_outputs"
        inputs = os.path.abspath(temp_dir)
        try:
            pics += 1
            print(f"[{loc_log}|{loc_pic} #{pics}]{loc_st_del}")
            print(f"Removing from: {inputs}")

            inputs = os.path.join(inputs, filename)
            outputs = os.path.join(outputs, f"{filename[:-4]}_output.png")

            input_image = Image.open(inputs)
            output_image = remove(input_image)
            output_image.save(outputs)

            print(f"[{loc_log}|{loc_pic} #{pics}]{loc_del}")

        except PermissionError:
            pics -= 1
            print(f"[{loc_err}|{loc_pic} #{pics}]{loc_perm_err}")
            pass
        except FileNotFoundError:
            pics -= 1
            print(f"[{loc_err}|{loc_pic} #{pics}]{loc_nofile_err}")
            pass
    outputs = current_directory + r"\rembg_outputs"
    return outputs


def flip_text(x):
    return x[::-1]


def flip_image(x):
    return np.fliplr(x)


with gr.Blocks(theme=gr.themes.Soft(primary_hue="red", secondary_hue="orange")) as app:
    gr.Markdown("Removing background from image.")
    with gr.Tab("BgRemoverLite"):
        with gr.Row():
            gr.Label("Remove background from single image", scale=0.5)
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image(width=200, height=240)
        with gr.Row():
            rembg_button = gr.Button("Remove one background")
        with gr.Row():
            gr.Label("Remove background from all images in dir")    
        with gr.Row():
            image_input_dir = gr.Textbox(label="Enter dir", placeholder="Enter dir like this: D:\Python\MultiAI")
            image_output_dir = gr.Textbox(label="Output")
        with gr.Row():
            rembg_batch_button = gr.Button("Remove some backgrounds", scale=0.5)

    rembg_button.click(rem_bg_def, inputs=image_input, outputs=image_output)
    rembg_batch_button.click(rem_bg_def_batch, inputs=image_input_dir, outputs=image_output_dir)

app.launch()

pycache_directory = os.path.join(current_directory, '__pycache__')
shutil.rmtree(pycache_directory)
