import shutil
import os
import gradio as gr
import numpy

from PIL import Image
from rembg import remove
from local_eng import *

current_directory = os.path.dirname(os.path.abspath(__file__))

def clear_cache():
    outputs = current_directory + r"\__pycache__"
    pycache_directory = os.path.join(current_directory, '__pycache__')
    shutil.rmtree(pycache_directory)
    return outputs

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

with gr.Blocks(theme=gr.themes.Soft(primary_hue="red", secondary_hue="orange")) as app:
    gr.Markdown("MultiAI v0.3.1")
    with gr.Tab("BgRemoverLite"):
        with gr.Row():
            gr.Label("Remove background from single image")
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image(width=200, height=240)
        with gr.Row():
            rembg_button = gr.Button("Remove one background")
        with gr.Row():
            gr.Label("Remove background from all images in dir")    
        with gr.Row():
            image_input_dir = gr.Textbox(label="Enter dir", placeholder="Enter dir like this: D:\Python\MultiAI")
            image_output_dir = gr.Textbox(label="Output", placeholder="BrRemoverLite outputs will be here")
        with gr.Row():
            rembg_batch_button = gr.Button("Remove some backgrounds")
    with gr.Tab("Clear cache"):
        with gr.Row():
            cache_dir = gr.Textbox(label="Cache Dir", placeholder="Cache dir will be here")
        with gr.Row():
            clear_cache_button = gr.Button("Click here to clear Python cache")

    rembg_button.click(rem_bg_def, inputs=image_input, outputs=image_output)
    rembg_batch_button.click(rem_bg_def_batch, inputs=image_input_dir, outputs=image_output_dir)
    
    clear_cache_button.click(clear_cache, outputs=cache_dir)

app.launch()
