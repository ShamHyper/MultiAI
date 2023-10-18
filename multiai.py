import shutil
import os
import gradio as gr
import numpy

from PIL import Image, UnidentifiedImageError
from rembg import remove

from nsfw_detector import predict

from tqdm import tqdm

import urllib.request

ver = "MultiAI v0.6.1"

current_directory = os.path.dirname(os.path.abspath(__file__))

modelname = "nsfw_mobilenet2.224x224.h5" 
url = "https://s3.amazonaws.com/ir_public/nsfwjscdn/nsfw_mobilenet2.224x224.h5"

def check_file(filename):
    files_in_directory = os.listdir(current_directory)

    if filename in files_in_directory:
        print("NSFW Model detected")
    else:
        print("NSFW Model undected. Downloading...")
        urllib.request.urlretrieve(url, modelname)

check_file(modelname)

def clear_cache():
    outputs = current_directory + r"\__pycache__"
    pycache_directory = os.path.join(current_directory, '__pycache__')
    shutil.rmtree(pycache_directory)
    return outputs

def rem_bg_def(inputs): 
    try:
        outputs = remove(inputs)
    except PermissionError:
        pass
    except FileNotFoundError:
        pass
    except UnidentifiedImageError:
        pass
    return outputs

def rem_bg_def_batch(inputs, outputs): 
    temp_dir = inputs
    for filename in tqdm(os.listdir(inputs)):
        outputs = "rembg_outputs"
        inputs = os.path.abspath(temp_dir)
        try:
            inputs = os.path.join(inputs, filename)
            outputs = os.path.join(outputs, f"{filename[:-4]}_output.png")

            input_image = Image.open(inputs)
            output_image = remove(input_image)
            output_image.save(outputs)
        except PermissionError:
            pass
        except FileNotFoundError:
            pass
        except UnidentifiedImageError:
            pass
    outputs = current_directory + r"\rembg_outputs"
    return outputs

def detector(detector_input, detector_slider, outputs):
    model = predict.load_model('nsfw_mobilenet2.224x224.h5')
    FOLDER_NAME = str(detector_input)
    THRESHOLD = detector_slider
    
    dirarr = [f'{FOLDER_NAME}/{f}' for f in os.listdir(FOLDER_NAME)]
    
    for file in tqdm(dirarr):
        try:
            result = predict.classify(model, file)
            keys_list = list(result.keys())
            x = keys_list[0]
            
            value_nsfw_1 = result[x]['porn']
            value_nsfw_2 = result[x]['hentai']
            value_nsfw_3 = result[x]['sexy']
            value_sfw = result[x]['neutral']
            
            if value_nsfw_1 > THRESHOLD or value_nsfw_2 > THRESHOLD or value_nsfw_3 > THRESHOLD*1.5:
                if value_sfw < THRESHOLD:
                    shutil.copyfile(file, f'./detector_outputs_nsfw/{file.split("/")[-1]}')
            else:
                shutil.copyfile(file, f'./detector_outputs_plain/{file.split("/")[-1]}')
        except (PermissionError, FileNotFoundError, UnidentifiedImageError):
            pass
    
    outputs = "NSFW: " + os.path.abspath('./detector_outputs_nsfw') + "\nPlain: " + os.path.abspath('./detector_outputs_plain')
    return outputs


def detector_clear(outputs):
    outputs_dir1 = os.path.join(current_directory, 'detector_outputs_nsfw')
    shutil.rmtree(outputs_dir1)
    outputs_dir2 = os.path.join(current_directory, 'detector_outputs_plain')
    shutil.rmtree(outputs_dir2)
    folder_path1 = 'detector_outputs_nsfw'
    os.makedirs(folder_path1)
    file = open(f"{folder_path1}/outputs will be here.txt", "w")
    file.close()
    folder_path2 = 'detector_outputs_plain'
    os.makedirs(folder_path2)
    file = open(f"{folder_path2}/outputs will be here.txt", "w")
    file.close()
    outputs = "Done!"
    return(outputs)
            
def clearp_bgr_def(outputs):
    outputs_dir = os.path.join(current_directory, 'rembg_outputs')
    shutil.rmtree(outputs_dir)
    folder_path = 'rembg_outputs'
    os.makedirs(folder_path)
    file = open(f"{folder_path}/outputs will be here.txt", "w")
    file.close()
    outputs = "Done!"
    return(outputs)

with gr.Blocks(title=ver,theme=gr.themes.Soft(primary_hue="red", secondary_hue="orange")) as multiai:
    gr.Markdown(ver)
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
        with gr.Row():
            gr.Label("Clear outputs")
        with gr.Row():
            clearp_bgr_button = gr.Button("Clear outputs")
            clearp_bgr = gr.Textbox(label="Clearing progress")
    with gr.Tab("NSFW Detector"):
        with gr.Row():
            gr.Label("Detect NSFW images from dir")
        with gr.Row():
            detector_input = gr.Textbox(label="Enter dir", placeholder="Enter dir like this: D:\Python\MultiAI")
            detector_slider = gr.Slider(value=0.36, 
                                        label="Threshold (larger number = simpler detection | smaller number = stricter one)", 
                                        minimum=0.000001, 
                                        maximum=0.999999)
        with gr.Row():
            detector_button = gr.Button("Click here to start")
            detector_output = gr.Textbox(label="Output", placeholder="NSFW Detector outputs will be here")
        with gr.Row():
            gr.Label("Clear outputs")
        with gr.Row():
            detector_clear_button = gr.Button("Clear outputs")
            clearp = gr.Textbox(label="Clearing progress")
    with gr.Tab("Clear cache"):
        with gr.Row():
            cache_dir = gr.Textbox(label="Cache Dir", placeholder="Cache dir will be here")
        with gr.Row():
            clear_cache_button = gr.Button("Click here to clear Python cache")

    rembg_button.click(rem_bg_def, inputs=image_input, outputs=image_output)
    rembg_batch_button.click(rem_bg_def_batch, inputs=image_input_dir, outputs=image_output_dir)
    clearp_bgr_button.click(clearp_bgr_def, outputs=clearp_bgr)
    
    detector_button.click(detector, inputs=[detector_input, detector_slider], outputs=detector_output)
    detector_clear_button.click(detector_clear, outputs=clearp)
    
    clear_cache_button.click(clear_cache)


multiai.queue()
multiai.launch(inbrowser=True)
