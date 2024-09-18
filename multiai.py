import shutil as sh
import os
from tqdm import tqdm
import json
import random

from PIL import Image, UnidentifiedImageError
from rembg import remove

from nsfw_detector import predict
import urllib.request

from upscalers import upscale

from clip_interrogator import Config, Interrogator
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

import cv2
from numba import cuda

from keras.models import load_model
import numpy as np

version = "[BETA] MultiAI v1.13.0-b2"

##################################################################################################################################

class init:
    ver = version
    
    print(f"Initializing {ver} launch...")
    
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    modelname = "nsfw_mobilenet2.224x224.h5"
    url = "https://vdmstudios.ru/server_archive/nsfw_mobilenet2.224x224.h5"
    
    modelname_h5 = "model.h5"
    url_h5 = "https://vdmstudios.ru/server_archive/model.h5"
    
    def check_file(filename):
        files_in_directory = os.listdir(init.current_directory)

        if filename in files_in_directory:
            if config.debug: 
                print("NSFW Model detected")
        else:
            if config.debug: 
                print("NSFW Model undected. Downloading...")
            urllib.request.urlretrieve(init.url, init.modelname)
    
    def checkfile_h5(filename):
        files_in_directory = os.listdir(init.current_directory)
        
        if filename in files_in_directory:
            if config.debug:
                print("H5 Model detected")
        else:
            if config.debug: 
                print("H5 Model undected. Downloading...")
            urllib.request.urlretrieve(init.url_h5, init.modelname_h5)
    
    def delete_tmp_pngs():
        output_dir = "tmp"
        try:
            rm_tmp = os.path.join(init.current_directory, output_dir)
            sh.rmtree(rm_tmp)
        except (PermissionError, FileNotFoundError, FileExistsError, Exception):
            pass
        
        tmp_file = "tmp.png"
        try:
            os.remove(tmp_file)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            pass
    
    def preloader():
        if config.debug: 
            print("Preloading models...")
        models.ci_load()
        models.nsfw_load()
        models.tokenizer_load()  
        models.h5_load()
        preloaded_tb = "Done!"
        return preloaded_tb
    
    def clear_all():
        multi.BgRemoverLite_Clear()
        multi.NSFWDetector_Clear()
        multi.VideoAnalyzerBatch_Clear()
        multi.AID_Clear()
        if config.debug: 
            print("All outputs cleared!")
        clear_all_tb = "All outputs deleted!"
        return clear_all_tb
        
##################################################################################################################################

class config:
    def save_config_gr(settings_debug_mode, settings_start_in_browser, settings_share_gradio, settings_preload_models, settings_clear_on_start):
        settings = {
            "debug_mode": str(settings_debug_mode),
            "start_in_browser": str(settings_start_in_browser),
            "share_gradio": str(settings_share_gradio),
            "preload_models": str(settings_preload_models),
            "clear_on_start": str(settings_clear_on_start)
        }
        
        if "dev_config.json" in os.listdir("settings"):
            json_file = "dev_config.json"
        elif "config.json" in os.listdir("settings"):
            json_file = "config.json"
            
        json_file = "settings/" + json_file
    
        with open(json_file, "w") as file:
            json.dump(settings, file, indent=4)
            
        settings_save_progress = f"Settings saved to [{json_file}]. Restart MultiAI!"
        
        return settings_save_progress
    
    if "dev_config.json" in os.listdir("settings"):
        with open("settings/dev_config.json") as json_file:
            data = json.load(json_file)
            print("dev_config.json loaded")
    elif "config.json" in os.listdir("settings"):
        with open("settings/config.json") as json_file:
            data = json.load(json_file)
            print("config.json loaded")

    debug = data.get("debug_mode", "False").lower() == "true"
    inbrowser = data.get("start_in_browser", "False").lower() == "true"
    share_gradio = data.get("share_gradio", "False").lower() == "true"
    preload_models = data.get("preload_models", "False").lower() == "true"
    clear_on_start = data.get("clear_on_start", "False").lower() == "true"

    if not (debug or inbrowser or share_gradio or preload_models or clear_on_start):
        if debug: 
            print("Something wrong in config.json. Check them out!")

##################################################################################################################################

class models:   
    def nsfw_load():
        global model_nsfw, nsfw_status
        try:
            if nsfw_status is not True:
                init.check_file(init.modelname)
                model_nsfw = predict.load_model("nsfw_mobilenet2.224x224.h5")
                nsfw_status = True
            elif nsfw_status is True:
                if config.debug: 
                    print("NSFW model already loaded!")
        except NameError:
                init.check_file(init.modelname)
                model_nsfw = predict.load_model("nsfw_mobilenet2.224x224.h5")
                nsfw_status = True
        return model_nsfw, nsfw_status

    def tokenizer_load():
        global tokenizer, tokenizer_status, model_tokinezer
        try:
            if tokenizer_status is not True:
                tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model_tokinezer = GPT2LMHeadModel.from_pretrained("FredZhang7/anime-anything-promptgen-v2")
                tokenizer_status = True
            elif tokenizer_status is True:
                if config.debug: 
                    print("Tokinezer already loaded!")
        except NameError:
                tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model_tokinezer = GPT2LMHeadModel.from_pretrained("FredZhang7/anime-anything-promptgen-v2")
                tokenizer_status = True
        return tokenizer, tokenizer_status, model_tokinezer

    def ci_load():
        global ci, ci_status
        try:
            if ci_status is not True:
                ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k"))
                ci_status = True
            elif ci_status is True:
                if config.debug: 
                    print("CLIP already loaded!")
        except NameError:
                ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k"))
                ci_status = True
        return ci, ci_status
    
    def h5_load():
        global model_h5, h5_status
        try:
            if h5_status is not True:
                init.checkfile_h5(init.modelname_h5)
                model_h5 = load_model("model.h5")
                h5_status = True
            elif h5_status is True:
                if config.debug: 
                    print("H5 model already loaded!")
        except NameError:
                init.checkfile_h5(init.modelname_h5)
                model_h5 = load_model("model.h5")
                h5_status = True
        return model_h5, h5_status
    
##################################################################################################################################
                  
class multi:
    def BgRemoverLite(inputs):
        try:
            outputs = remove(inputs)
        except (PermissionError, FileNotFoundError, UnidentifiedImageError) as e:
            if config.debug: 
                print(f"Error: {e}")
            pass
        return outputs

    def BgRemoverLiteBatch(inputs):
        temp_dir = inputs
        for filename in tqdm(os.listdir(inputs)):
            outputs = "outputs/rembg_outputs"
            inputs = os.path.abspath(temp_dir)
            try:
                inputs = os.path.join(inputs, filename)
                outputs = os.path.join(outputs, f"{filename[:-4]}_output.png")

                input_image = Image.open(inputs)
                output_image = remove(input_image)
                output_image.save(outputs)
            except (PermissionError, FileNotFoundError, UnidentifiedImageError) as e:
                if config.debug: 
                    print(f"Error: {e}")
                pass
        outputs = init.current_directory + r"\outputs" + r"\rembg_outputs"
        return outputs
    
    def BgRemoverLite_Clear():
        outputs_dir = os.path.join(init.current_directory, "outputs/rembg_outputs")
        sh.rmtree(outputs_dir)
        folder_path = "outputs/rembg_outputs"
        os.makedirs(folder_path)
        file = open(f"{folder_path}/outputs will be here.txt", "w")
        file.close()
        outputs = "Done!"
        return outputs

##################################################################################################################################

    def NSFW_Detector(detector_input, detector_slider, detector_skeep_dr, drawings_threshold):
        if detector_skeep_dr is True:
            if config.debug: 
                print("I will skip drawings!")
        models.nsfw_load()
        init.check_file(init.modelname)
        FOLDER_NAME = str(detector_input)
        THRESHOLD = detector_slider
        DRAW_THRESHOLD = drawings_threshold
        nsfw = 0
        plain = 0

        dirarr = [f"{FOLDER_NAME}/{f}" for f in os.listdir(FOLDER_NAME)]

        for file in tqdm(dirarr):
            try:
                result = predict.classify(model_nsfw, file)
                keys_list = list(result.keys())
                x = keys_list[0]

                value_nsfw_1 = result[x]["porn"]
                value_nsfw_2 = result[x]["hentai"]
                value_nsfw_3 = result[x]["sexy"]
                value_draw = result[x]["drawings"]

                if detector_skeep_dr is False:
                    if value_nsfw_1 > THRESHOLD or value_nsfw_2 > THRESHOLD or value_nsfw_3 > THRESHOLD * 1.3:
                        sh.copyfile(file, f"./outputs/detector_outputs_nsfw/{file.split("/")[-1]}")
                        nsfw += 1
                    else:
                        sh.copyfile(file, f"./outputs/detector_outputs_plain/{file.split("/")[-1]}")
                        plain += 1
                        
                elif detector_skeep_dr is True:
                    if value_draw > DRAW_THRESHOLD or value_nsfw_2 > THRESHOLD * 1.5:
                        if config.debug: 
                            print(f"I skipped this pic, because value_draw[{value_draw}] > DRAW_THRESHOLD[{DRAW_THRESHOLD}]")
                        pass
                    elif value_nsfw_1 > THRESHOLD or value_nsfw_2 > THRESHOLD or value_nsfw_3 > THRESHOLD * 1.3:
                        sh.copyfile(file, f"./outputs/detector_outputs_nsfw/{file.split("/")[-1]}")
                        nsfw += 1
                    else:
                        sh.copyfile(file, f"./outputs/detector_outputs_plain/{file.split("/")[-1]}")
                        plain += 1
                        
            except (PermissionError, FileNotFoundError, UnidentifiedImageError) as e:
                if config.debug: 
                    print(f"Error: {e}")
                pass

        outputs = (
            f"[{str(nsfw)}] NSFW: {os.path.abspath("./outputs/detector_outputs_nsfw")}\n"
            f"[{str(plain)}] Plain: {os.path.abspath("./outputs/detector_outputs_plain")}"
        )
        return outputs

    def NSFWDetector_Clear():
        outputs_dir1 = os.path.join(init.current_directory, "outputs/detector_outputs_nsfw")
        sh.rmtree(outputs_dir1)
        outputs_dir2 = os.path.join(init.current_directory, "outputs/detector_outputs_plain")
        sh.rmtree(outputs_dir2)
        folder_path1 = "outputs/detector_outputs_nsfw"
        os.makedirs(folder_path1)
        file = open(f"{folder_path1}/outputs will be here.txt", "w")
        file.close()
        folder_path2 = "outputs/detector_outputs_plain"
        os.makedirs(folder_path2)
        file = open(f"{folder_path2}/outputs will be here.txt", "w")
        file.close()
        outputs = "Done!"
        return outputs

##################################################################################################################################

    def Upscaler(upsc_image_input, scale_factor, model_ups):
        tmp_img_ndr = Image.fromarray(upsc_image_input)
        upsc_image_output = upscale(model_ups, tmp_img_ndr, scale_factor)
        return upsc_image_output
    
##################################################################################################################################
    
    def ImageAnalyzer(file_spc, clip_checked):
        img = Image.fromarray(file_spc, "RGB")
        img.save("tmp.png")
        dir_img_fromarray = os.path.join(os.getcwd(), "tmp.png")

        models.nsfw_load()
        result = predict.classify(model_nsfw, dir_img_fromarray)
        x = next(iter(result.keys()))

        values = result[x]
        total_sum = sum(values.values())
        percentages = {k: round((v / total_sum) * 100, 1) for k, v in values.items()}

        spc_output = ""
        if clip_checked is True:
            models.ci_load()
            clip = Image.open(dir_img_fromarray).convert("RGB")
            spc_output += f"Prompt:\n{ci.interrogate(clip)}\n\n" 

        spc_output += f"Drawings: {percentages["drawings"]}%\n"
        spc_output += f"Porn: {percentages["porn"]}%\n"
        spc_output += f"Hentai: {percentages["hentai"]}%\n"
        spc_output += f"Sexy: {percentages["sexy"]}%\n"
        spc_output += f"Neutral: {percentages["neutral"]}%"

        tmp_file = "tmp.png"
        
        try:
            os.remove(tmp_file)
        except FileNotFoundError as e:
            if config.debug: 
                print(f"Error: {e}")
            pass

        return spc_output
    
##################################################################################################################################
    
    def VideoAnalyzer(file_Vspc):
        output_dir = "tmp"
        os.makedirs(output_dir, exist_ok=True)
        models.nsfw_load()
        cap = cv2.VideoCapture(file_Vspc)

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            output_file = os.path.join(output_dir, f"{frame_count + 1}.png")
            cv2.imwrite(output_file, frame)
            frame_count += 1
        
        dir_tmp = "tmp"
        total_sum = 0
        file_count = 0
        
        for file_name in tqdm(os.listdir(dir_tmp)):
            file_path = os.path.join(dir_tmp, file_name)
            
            result = predict.classify(model_nsfw, file_path)
            x = next(iter(result.keys()))
            values = result[x]
            file_sum = sum(values.values())
            total_sum += file_sum 
            file_count += 1  

        avg_sum = total_sum / file_count 
        percentages = {k: round((v / avg_sum ) * 100, 1) for k, v in values.items()}
        
        value1 = percentages["drawings"]
        value2 = percentages["hentai"]
        value3 = percentages["neutral"]
        value4 = percentages["porn"]
        value5 = percentages["sexy"]
        
        Vspc_output = f"Drawings: {value1}%\n"
        Vspc_output += f"Hentai: {value2}%\n"
        Vspc_output += f"Porn: {value4}%\n"
        Vspc_output += f"Sexy: {value5}%\n"
        Vspc_output += f"Neutral: {value3}%"
        
        rm_tmp = os.path.join(init.current_directory, dir_tmp)
        sh.rmtree(rm_tmp)
        cap.release()
        cv2.destroyAllWindows()
        
        return Vspc_output
    
    def process_frame(frame):
        result_frame = frame    
        return result_frame

    def VideoAnalyzerBatch(video_dir, vbth_slider, threshold_Vspc_slider):
        models.nsfw_load()
        _nsfw = 0
        _plain = 0
        out_cmd = str("")
        output_dir = "tmp"
        os.makedirs(output_dir, exist_ok=True)
        video_files = os.listdir(video_dir)
        
        for dir_Vspc in tqdm(video_files):
            _nsfw_factor = False
            _plain_factor = False
            cap = cv2.VideoCapture(os.path.join(video_dir, dir_Vspc))
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_gpu = cuda.to_device(frame)
                processed_frame_gpu = multi.process_frame(frame_gpu)

                processed_frame = processed_frame_gpu.copy_to_host()

                output_file = os.path.join(output_dir, f"{frame_count + 1}.png")
                cv2.imwrite(output_file, processed_frame)

                frame_count += 1

            cap.release()
            
            total_sum = 0
            file_count = 0
            
            for i, file_name in enumerate(os.listdir(output_dir)):
                if vbth_slider != 1:
                    if i % vbth_slider != 0:
                        continue
                elif vbth_slider == 1:
                    if config.debug is True:
                        print("Frame-Skip disabled!")
                    else:
                        pass
                    
                try:   
                    file_path = os.path.join(output_dir, file_name)
                    result = predict.classify(model_nsfw, file_path)
                    x = next(iter(result.keys()))
                    values = result[x]
                    file_sum = sum(values.values())
                    total_sum += file_sum 
                    file_count += 1
                except Exception:
                    pass
                
            try: 
                avg_sum = total_sum / file_count 
                percentages = {k: round((v / avg_sum ) * 100, 1) for k, v in values.items()}
                THRESHOLD = threshold_Vspc_slider
                
                value_nsfw_1 = percentages["porn"]
                value_nsfw_2 = percentages["hentai"]
                value_nsfw_3 = percentages["sexy"]
            except (ZeroDivisionError, Exception):
                pass
            
            try:
                if value_nsfw_1 > THRESHOLD or value_nsfw_2 > THRESHOLD or value_nsfw_3 > THRESHOLD:
                    video_path = os.path.join(video_dir, dir_Vspc)
                    sh.copy(video_path, "outputs/video_analyze_nsfw")
                    _nsfw += 1
                    _nsfw_factor = True
                else:
                    video_path = os.path.join(video_dir, dir_Vspc)
                    sh.copy(video_path, "outputs/video_analyze_plain")
                    _plain += 1
                    _plain_factor = True
            except (PermissionError, FileExistsError, Exception):
                pass
                
            cap.release()
            cv2.destroyAllWindows()
            
            rm_tmp = os.path.join(init.current_directory, output_dir)
            sh.rmtree(rm_tmp)
            os.makedirs(output_dir, exist_ok=True)
            
            if _nsfw_factor is True:  
                out_cmd = f"[+]NSFW: {_nsfw}"
                out_cmd += f"\nPlain: {_plain}"
                
            elif _plain_factor is True:
                out_cmd = f"NSFW: {_nsfw}"
                out_cmd += f"\n[+]Plain: {_plain}"
                
            if config.debug: 
                print("")
                print(out_cmd)
                print("")
            out_cmd = str("")
            avg_sum = 0
            percentages = 0
        bth_Vspc_output = "Ready!"
            
        return bth_Vspc_output

    def VideoAnalyzerBatch_Clear():
        output_dir = "tmp"
        try:
            outputs_dir1 = os.path.join(init.current_directory, "outputs/video_analyze_nsfw")
            sh.rmtree(outputs_dir1)
            outputs_dir2 = os.path.join(init.current_directory, "outputs/video_analyze_plain")
            sh.rmtree(outputs_dir2)
            outputs_dir3 = os.path.join(init.current_directory, "tmp")
            sh.rmtree(outputs_dir3)
            folder_path1 = "outputs/video_analyze_nsfw"
            os.makedirs(folder_path1)
            file = open(f"{folder_path1}/outputs will be here.txt", "w")
            file.close()
            folder_path2 = "outputs/video_analyze_plain"
            os.makedirs(folder_path2)
            file = open(f"{folder_path2}/outputs will be here.txt", "w")
            file.close()
            rm_tmp = os.path.join(init.current_directory, output_dir)
            sh.rmtree(rm_tmp)
        except (PermissionError, FileNotFoundError, FileExistsError, Exception):
            try:
                folder_path1 = "outputs/video_analyze_nsfw"
                os.makedirs(folder_path1)
                file = open(f"{folder_path1}/outputs will be here.txt", "w")
                file.close()
                folder_path2 = "outputs/video_analyze_plain"
                os.makedirs(folder_path2)
                file = open(f"{folder_path2}/outputs will be here.txt", "w")
                file.close()
            except (PermissionError, FileNotFoundError, FileExistsError, Exception):
                pass
            
        bth_Vspc_clear_output = "Done!"
        return bth_Vspc_clear_output 
    
##################################################################################################################################

    def PromptGenetator(prompt_input, pg_prompts, pg_max_length, randomize_temp):
        models.tokenizer_load()
        prompt = prompt_input
        if randomize_temp is True:
            tempreture_pg = (random.randint(4, 9)/10)
        elif randomize_temp is False:
            tempreture_pg = 0.7

        nlp = pipeline("text-generation", model=model_tokinezer, tokenizer=tokenizer)
        outs = nlp(prompt, 
                   max_length=pg_max_length, 
                   num_return_sequences=pg_prompts, 
                   do_sample=True, 
                   repetition_penalty=1.2, 
                   temperature=tempreture_pg, 
                   top_k=4, 
                   early_stopping=False)

        for i in tqdm(range(len(outs))):
            outs[i] = str(outs[i]["generated_text"]).replace("  ", "").rstrip(",")
        promptgen_output = ("\n\n".join(outs) + "\n")  
        return promptgen_output 
    
##################################################################################################################################

    def is_image_generated(test_image, threshold):
                test_image = test_image.resize((200, 200))  
                test_image = test_image.convert("L")  
                test_image = np.array(test_image) #/ 255.0
                test_image = np.expand_dims(test_image, axis=0)
                test_image = np.expand_dims(test_image, axis=-1)  
                result = model_h5.predict(test_image)
                predicted_max = float(np.max(result[0]))
                
                if config.debug: 
                    print("") 
                    print(f"Result array of detecting:{result}") 
                    print(f"MAX result of detecting:{predicted_max}") 
                    print(f"Threshold:{threshold}") 
                if predicted_max >= threshold:
                    isai = "This is an image created by AI"
                    return isai
                else:
                    isai = "This is an image created by HUMAN"
                    return isai
                

    def AiDetector_single(aid_input_single, threshold):
        models.h5_load()   
        img_h5 = Image.fromarray(aid_input_single)
        aid_output_single = multi.is_image_generated(img_h5, threshold)
        return aid_output_single
    

    def AiDetector_batch(aid_input_batch, threshold): 
        models.h5_load()   
        aid_ai_dir = os.path.join(init.current_directory, "outputs/aid_ai")
        aid_human_dir = os.path.join(init.current_directory, "outputs/aid_human")
        
        if not os.path.exists(aid_ai_dir):
            os.makedirs(aid_ai_dir)
        if not os.path.exists(aid_human_dir):
            os.makedirs(aid_human_dir)
        
        image_files = os.listdir(aid_input_batch)
        
        for image_file in image_files:
            try:
                img_path = os.path.join(aid_input_batch, image_file)
                
                if config.debug:
                    print(f"Processing image: {img_path}")
                
                img_h5 = Image.open(img_path)
                result = multi.is_image_generated(img_h5, threshold)
                
                if config.debug:
                    print(f"Result for {image_file}: {result}")
                
                if result == "This is an image created by AI":
                    sh.copyfile(img_path, os.path.join(aid_ai_dir, image_file))
                elif result == "This is an image created by HUMAN":
                    sh.copyfile(img_path, os.path.join(aid_human_dir, image_file))
            except (UnidentifiedImageError, PermissionError, FileNotFoundError, FileExistsError, Exception) as e:
                print(f"Error processing {image_file}: {e}")
        
        aid_output_batch = "Images sorted successfully!"
        return aid_output_batch
    
    
    def AID_Clear():
        outputs_dir1 = os.path.join(init.current_directory, "outputs/aid_ai")
        sh.rmtree(outputs_dir1)
        outputs_dir2 = os.path.join(init.current_directory, "outputs/aid_human")
        sh.rmtree(outputs_dir2)
        folder_path1 = "outputs/aid_ai"
        os.makedirs(folder_path1)
        file = open(f"{folder_path1}/outputs will be here.txt", "w")
        file.close()
        folder_path2 = "outputs/aid_human"
        os.makedirs(folder_path2)
        file = open(f"{folder_path2}/outputs will be here.txt", "w")
        file.close()
        outputs = "Done!"
        return outputs

##################################################################################################################################