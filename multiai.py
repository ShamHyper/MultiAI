import shutil as sh
import os
from tqdm import tqdm
from icecream import ic
import json

from PIL import Image, UnidentifiedImageError
from rembg import remove

from nsfw_detector import predict
import urllib.request

from upscalers import upscale

from clip_interrogator import Config, Interrogator

class init:
    ver = "[Beta]MultiAI v1.1.0"
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
        
    clear_need = data.get("clear_need")
    if clear_need == "False":
        clear_need = False
    elif clear_need == "True":
        clear_need = True
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
    ic()
    ic("Loading model...")
    model = predict.load_model("nsfw_mobilenet2.224x224.h5")
    ic()
    ic("Model nsfw_mobilenet2.224x224.h5 loaded!")
    
    ic()
    ic("Loading clip model and cfgs...")
    ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k"))
    
    def clear_cache():
        ic("Clearing cache...")
        try:
            cache1 = os.path.join(init.current_directory, ".ruff_cache")
            sh.rmtree(cache1)
            cache1 = os.path.join(init.current_directory, "__pycache__")
            sh.rmtree(cache1)
        except PermissionError:
            pass
        except FileNotFoundError:
            pass
        return("Done")
    
class multi:
    def rem_bg_def(inputs):
        try:
            outputs = remove(inputs)
            ic()
            ic("Removing bg...")
        except PermissionError:
            ic()
            ic("PermissionError")
            pass
        except FileNotFoundError:
            ic()
            ic("FileNotFoundError")
            pass
        except UnidentifiedImageError:
            ic()
            ic("UnidentifiedImageError")
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
            except PermissionError:
                ic()
                ic("PermissionError")
                pass
            except FileNotFoundError:
                ic()
                ic("FileNotFoundError")
                pass
            except UnidentifiedImageError:
                ic()
                ic("UnidentifiedImageError")
                pass
        outputs = init.current_directory + r"\rembg_outputs"
        return outputs

    def detector(detector_input, detector_slider):
        multi.check_file(init.modelname)
        ic()
        ic("Loading model...")
        model = predict.load_model("nsfw_mobilenet2.224x224.h5")
        ic()
        ic("Model nsfw_mobilenet2.224x224.h5 loaded!")
        FOLDER_NAME = str(detector_input)
        THRESHOLD = detector_slider
        nsfw = 0
        plain = 0

        dirarr = [f"{FOLDER_NAME}/{f}" for f in os.listdir(FOLDER_NAME)]

        for file in tqdm(dirarr):
            try:
                result = predict.classify(model, file)
                ic()
                keys_list = list(result.keys())
                ic()
                x = keys_list[0]
                ic()

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

            except (PermissionError, FileNotFoundError, UnidentifiedImageError, ValueError):
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
        img = Image.fromarray(file_spc, 'RGB')
        img.save('tmp.png')
        dir_img_fromarray = init.current_directory+r"\tmp.png"
        
        result = predict.classify(init.model, dir_img_fromarray)
        ic()
        keys_list = list(result.keys())
        ic()
        x = keys_list[0]
        ic()

        value_drawings = result[x]["drawings"]
        value_porn = result[x]["porn"]
        value_hentai = result[x]["hentai"]
        value_sexy = result[x]["sexy"]
        value_neutral = result[x]["neutral"]
        
        total_sum = value_drawings + value_porn + value_hentai + value_sexy + value_neutral
        value_drawings_precent = (value_drawings / total_sum) * 100
        value_porn_precent = (value_porn / total_sum) * 100
        value_hentai_precent = (value_hentai / total_sum) * 100
        value_sexy_precent = (value_sexy / total_sum) * 100
        value_neutral_precent = (value_neutral / total_sum) * 100
        
        clip = Image.open(dir_img_fromarray).convert('RGB')
        
        if clip_checked is True:
            spc_output = str(f"Prompt: {init.ci.interrogate(clip)}\n\nDrawings: {round(value_drawings_precent, 1)}%\nPorn: {round(value_porn_precent, 1)}%\nHentai: {round(value_hentai_precent, 1)}%\nSexy: {round(value_sexy_precent, 1)}%\nNeutral: {round(value_neutral_precent, 1)}%")
        elif clip_checked is False:
            spc_output = str(f"Drawings: {round(value_drawings_precent, 1)}%\nPorn: {round(value_porn_precent, 1)}%\nHentai: {round(value_hentai_precent, 1)}%\nSexy: {round(value_sexy_precent, 1)}%\nNeutral: {round(value_neutral_precent, 1)}%")

        tmp_file = "tmp.png"
        try:
            os.remove(tmp_file)
        except FileNotFoundError:
            ic()
            ic("FileNotFoundError")
            pass

        return(spc_output)
        
    if init.clear_need is True:
        if init.debug is True:
            ic(init.clear_cache())
        elif init.debug is False:
            init.clear_cache()   