import shutil
import os

from PIL import Image, UnidentifiedImageError
from rembg import remove

from nsfw_detector import predict

from tqdm import tqdm

import urllib.request

import json

from icecream import ic


class init:
    ver = "[Beta]MultiAI v0.8.0"
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
        
    start_in_browser = data.get("start_in_browser")
    if start_in_browser == "False":
        start_in_browser = False
    elif start_in_browser == "True":
        start_in_browser = True
    else:
        print("Something wrong in config.json. Check them out!")
    
    ic()
    ic(f"Start in browser: {start_in_browser}")
    ic(f"Debug mode: {debug}")

    current_directory = os.path.dirname(os.path.abspath(__file__))
    ic()
    ic(current_directory)

    modelname = "nsfw_mobilenet2.224x224.h5"
    url = "https://s3.amazonaws.com/ir_public/nsfwjscdn/nsfw_mobilenet2.224x224.h5"
    
    def clear_cache():
        ic()
        ic("Clearing cache...")
        try:
            cache1 = os.path.join(init.current_directory, ".ruff_cache")
            shutil.rmtree(cache1)
            cache1 = os.path.join(init.current_directory, "__pycache__")
            shutil.rmtree(cache1)
        except PermissionError:
            ic()
            ic("PermissionError")
            pass
        except FileNotFoundError:
            ic()
            ic("FileNotFoundError")
            pass    
    
class multi:
    def check_file(filename):
        files_in_directory = os.listdir(init.current_directory)

        if filename in files_in_directory:
            ic()
            ic("NSFW Model 1 detected")
        else:
            ic()
            ic("NSFW Model undected. Downloading...")
            urllib.request.urlretrieve(init.url, init.modelname)

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
                    shutil.copyfile(file, f'./detector_outputs_nsfw/{file.split("/")[-1]}')
                    nsfw += 1
                else:
                    shutil.copyfile(file, f'./detector_outputs_plain/{file.split("/")[-1]}')
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
        shutil.rmtree(outputs_dir1)
        outputs_dir2 = os.path.join(init.current_directory, "detector_outputs_plain")
        shutil.rmtree(outputs_dir2)
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
        shutil.rmtree(outputs_dir)
        folder_path = "rembg_outputs"
        os.makedirs(folder_path)
        file = open(f"{folder_path}/outputs will be here.txt", "w")
        file.close()
        outputs = "Done!"
        return outputs
    
    init.clear_cache()
    
    