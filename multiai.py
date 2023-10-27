import time
start_time = time.time()
import shutil as sh
import os
from tqdm import tqdm
from icecream import ic
import json
import random

from PIL import Image, UnidentifiedImageError
from rembg import remove

from nsfw_detector import predict
import urllib.request

from upscalers import upscale

from clip_interrogator import Config, Interrogator
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

class init:
    ver = "[Beta]MultiAI v1.4.6"
    print(f"Initializing {ver} launch...")

    with open("config.json") as json_file:
        data = json.load(json_file)

    debug = data.get("debug_mode")
    if debug == "False":
        debug = False
    elif debug == "True":
        debug = True
    else:
        print("Something wrong in config.json. Check them out!")
        
    if debug is False:
        ic.disable()
    elif debug is True:
        ic.enable()
        
    inbrowser = data.get("start_in_browser")
    if inbrowser == "False":
        inbrowser = False
    elif inbrowser == "True":
        inbrowser = True
    else:
        print("Something wrong in config.json. Check them out!")

    share_gradio = data.get("share_gradio")
    if share_gradio == "False":
        share_gradio = False
    elif share_gradio == "True":
        share_gradio = True
    else:
        print("Something wrong in config.json. Check them out!")
  
    ic()
    ic(f"Start in browser: {inbrowser}")
    ic(f"Debug mode: {debug}")

    current_directory = os.path.dirname(os.path.abspath(__file__))
    ic()
    ic(current_directory)

    modelname = "nsfw_mobilenet2.224x224.h5"
    url = "https://s3.amazonaws.com/ir_public/nsfwjscdn/nsfw_mobilenet2.224x224.h5"
    
    def check_file(filename):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        files_in_directory = os.listdir(current_directory)

        if filename in files_in_directory:
            ic()
            ic("NSFW Model 1 detected")
        else:
            ic()
            ic("NSFW Model undected. Downloading...")
            urllib.request.urlretrieve(init.url, init.modelname)
    
    check_file(modelname)

end_time = time.time()
total_time = round(end_time - start_time)
ic(f"Executing init time: {total_time}s")
    
class multi:
    def rem_bg_def(inputs):
        try:
            outputs = remove(inputs)
            ic()
            ic("Removing bg...")
        except (PermissionError, FileNotFoundError, UnidentifiedImageError) as e:
            ic()
            ic(f"Error: {e}")
            pass
        return outputs

    def rem_bg_def_batch(inputs):
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
            except (PermissionError, FileNotFoundError, UnidentifiedImageError) as e:
                ic()
                ic(f"Error: {e}")
                pass
        outputs = init.current_directory + r"\rembg_outputs"
        return outputs

    def detector(detector_input, detector_slider):
        ic()
        init.check_file(init.modelname)
        ic("Loading NSFW model...")
        model = predict.load_model("nsfw_mobilenet2.224x224.h5")
        ic("Model nsfw_mobilenet2.224x224.h5 loaded!")
        FOLDER_NAME = str(detector_input)
        THRESHOLD = detector_slider
        nsfw = 0
        plain = 0

        dirarr = [f"{FOLDER_NAME}/{f}" for f in os.listdir(FOLDER_NAME)]

        for file in tqdm(dirarr):
            try:
                ic()
                result = predict.classify(model, file)
                ic(result)
                keys_list = list(result.keys())
                ic(keys_list)
                x = keys_list[0]
                ic(x)

                value_nsfw_1 = result[x]["porn"]
                value_nsfw_2 = result[x]["hentai"]
                value_nsfw_3 = result[x]["sexy"]
                value_sfw = result[x]["neutral"]

                if (value_nsfw_1 > THRESHOLD or value_nsfw_2 > THRESHOLD or value_nsfw_3 > THRESHOLD * 1.5) and value_sfw < THRESHOLD:
                    sh.copyfile(file, f'./detector_outputs_nsfw/{file.split("/")[-1]}')
                    nsfw += 1
                else:
                    sh.copyfile(file, f'./detector_outputs_plain/{file.split("/")[-1]}')
                    plain += 1
                ic()
                ic(result)

            except (PermissionError, FileNotFoundError, UnidentifiedImageError) as e:
                ic()
                ic(f"Error: {e}")
                pass

        outputs = (
            "["
            + str(nsfw)
            + "]"
            + "NSFW: "
            + os.path.abspath("./detector_outputs_nsfw")
            + "\n["
            + str(plain)
            + "]"
            + "Plain: "
            + os.path.abspath("./detector_outputs_plain")
        )
        return outputs

    def detector_clear():
        ic()
        ic("Removing dirs...")
        outputs_dir1 = os.path.join(init.current_directory, "detector_outputs_nsfw")
        sh.rmtree(outputs_dir1)
        outputs_dir2 = os.path.join(init.current_directory, "detector_outputs_plain")
        sh.rmtree(outputs_dir2)
        folder_path1 = "detector_outputs_nsfw"
        os.makedirs(folder_path1)
        file = open(f"{folder_path1}/outputs will be here.txt", "w")
        file.close()
        folder_path2 = "detector_outputs_plain"
        os.makedirs(folder_path2)
        file = open(f"{folder_path2}/outputs will be here.txt", "w")
        file.close()
        outputs = "Done!"
        return outputs

    def clearp_bgr_def():
        ic()
        ic("Removing dirs...")
        outputs_dir = os.path.join(init.current_directory, "rembg_outputs")
        sh.rmtree(outputs_dir)
        folder_path = "rembg_outputs"
        os.makedirs(folder_path)
        file = open(f"{folder_path}/outputs will be here.txt", "w")
        file.close()
        outputs = "Done!"
        return outputs
    
    def uspc(upsc_image_input, scale_factor, model_ups):
        ic()
        ic("Start upscaling...")
        ic("Model:" + model_ups)
        ic("Scale factor:" + str(scale_factor))
        tmp_img_ndr = Image.fromarray(upsc_image_input)
        upsc_image_output = upscale(model_ups, tmp_img_ndr, scale_factor)
        return upsc_image_output
    
    def spc(file_spc, clip_checked):
        ic()
        init.check_file(init.modelname)
        ic("Loading NSFW model...")
        model_nsfw = predict.load_model("nsfw_mobilenet2.224x224.h5")
        ic("Model nsfw_mobilenet2.224x224.h5 loaded!")
        img = Image.fromarray(file_spc, 'RGB')
        img.save('tmp.png')
        dir_img_fromarray = os.path.join(os.getcwd(), "tmp.png")

        result = predict.classify(model_nsfw, dir_img_fromarray)
        ic(result.keys())
        x = next(iter(result.keys()))
        ic(x)

        values = result[x]
        total_sum = sum(values.values())
        percentages = {k: round((v / total_sum) * 100, 1) for k, v in values.items()}

        spc_output = ""
        if clip_checked is True:
            ic("Loading clip model and cfgs...")
            ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k"))
            ic("Clip model loaded!")
            clip = Image.open(dir_img_fromarray).convert('RGB')
            spc_output += f"Prompt:\n{ci.interrogate(clip)}\n\n"

        spc_output += f"Drawings: {percentages['drawings']}%\n"
        spc_output += f"Porn: {percentages['porn']}%\n"
        spc_output += f"Hentai: {percentages['hentai']}%\n"
        spc_output += f"Sexy: {percentages['sexy']}%\n"
        spc_output += f"Neutral: {percentages['neutral']}%"

        tmp_file = "tmp.png"
        try:
            os.remove(tmp_file)
        except FileNotFoundError as e:
            ic()
            ic(f"Error: {e}")
            pass

        return spc_output
    
    def prompt_generator(prompt_input, pg_prompts, pg_max_length, randomize_temp):
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model_tokinezer = GPT2LMHeadModel.from_pretrained('FredZhang7/anime-anything-promptgen-v2')

        prompt = prompt_input
        if randomize_temp is True:
            tempreture_pg = (random.randint(4, 9)/10)
            ic(tempreture_pg)
        elif randomize_temp is False:
            tempreture_pg = 0.7
            ic("Temperature default")
            ic(tempreture_pg)

        nlp = pipeline('text-generation', model=model_tokinezer, tokenizer=tokenizer)
        outs = nlp(prompt, 
                   max_length=pg_max_length, 
                   num_return_sequences=pg_prompts, 
                   do_sample=True, 
                   repetition_penalty=1.2, 
                   temperature=tempreture_pg, 
                   top_k=4, 
                   early_stopping=False)

        for i in tqdm(range(len(outs))):
            outs[i] = str(outs[i]['generated_text']).replace('  ', '').rstrip(',')
        promptgen_output = ('\n\n'.join(outs) + '\n')  
        return promptgen_output 