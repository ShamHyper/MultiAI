import config
import models

import shutil as sh
import os
from tqdm import tqdm
import random
import gradio as gr
import gc

from PIL import Image, UnidentifiedImageError
from rembg import remove
from nsfw_detector import predict
from transformers import pipeline
import torch
import tensorflow as tf 
import cv2
import numpy as np

from upscalers import upscale
from upscalers import clear_on_device_caches
from datetime import datetime

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def BgRemoverLite(inputs):
    try:
        outputs = remove(inputs)
    except (PermissionError, FileNotFoundError, UnidentifiedImageError) as e: 
        gr.Error(f"Error: {e}")
        pass
    
    CODC_clear(silent=True)
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
            gr.Error(f"Error: {e}")
            pass
        
    outputs = config.current_directory + r"\outputs" + r"\rembg_outputs"
    
    CODC_clear(silent=True)
    return outputs

def BgRemoverLite_Clear():
    outputs_dir = os.path.join(config.current_directory, "outputs/rembg_outputs")
    sh.rmtree(outputs_dir)
    folder_path = "outputs/rembg_outputs"
    os.makedirs(folder_path)
    file = open(f"{folder_path}/outputs will be here.txt", "w")
    file.close()
    gr.Info("BgRemoverLite outputs cleared")

##################################################################################################################################

def NSFW_Detector(detector_input):         
    model, processor = models.nsfw_ng_load()
    
    nsfw = 0
    plain = 0
    
    FOLDER_NAME = str(detector_input)
    dirarr = [f"{FOLDER_NAME}/{f}" for f in os.listdir(FOLDER_NAME)]
    
    for file in tqdm(dirarr):
        try:
            with torch.no_grad():
                image = Image.open(file).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits

            predicted_label = logits.argmax(-1).item()
            predicted_class = model.config.id2label[predicted_label]

            if predicted_class == "normal":
                sh.copyfile(file, f'./outputs/detector_outputs_plain/{file.split("/")[-1]}')
                plain += 1
            elif predicted_class == "nsfw":
                sh.copyfile(file, f'./outputs/detector_outputs_nsfw/{file.split("/")[-1]}')
                nsfw += 1
        except Exception as e:
            gr.Error(f"Error: {e}")
            pass

    outputs = (
        f"[{str(nsfw)}] NSFW: {os.path.abspath('./outputs/detector_outputs_nsfw')}\n"
        f"[{str(plain)}] Plain: {os.path.abspath('./outputs/detector_outputs_plain')}"
    )
    CODC_clear(silent=True)
    return outputs

def NSFWDetector_Clear():
    outputs_dir1 = os.path.join(config.current_directory, "outputs/detector_outputs_nsfw")
    sh.rmtree(outputs_dir1)
    outputs_dir2 = os.path.join(config.current_directory, "outputs/detector_outputs_plain")
    sh.rmtree(outputs_dir2)
    folder_path1 = "outputs/detector_outputs_nsfw"
    os.makedirs(folder_path1)
    file = open(f"{folder_path1}/outputs will be here.txt", "w")
    file.close()
    folder_path2 = "outputs/detector_outputs_plain"
    os.makedirs(folder_path2)
    file = open(f"{folder_path2}/outputs will be here.txt", "w")
    file.close()
    gr.Info("Detector outputs cleared")

##################################################################################################################################

def nsfw_ng(file_nsfw_ng):
    model, processor = models.nsfw_ng_load()
    with torch.no_grad():
        inputs = processor(images=file_nsfw_ng, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        if config.debug:
            print(f"NSFW_NG logits: {logits}")

    predicted_label = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_label]
    
    return predicted_class

##################################################################################################################################

def Upscaler(upsc_image_input, scale_factor, model_ups):
    tmp_img_ndr = Image.fromarray(upsc_image_input)
    upsc_image_output = upscale(model_ups, tmp_img_ndr, scale_factor)
    
    CODC_clear(silent=True)
    return upsc_image_output

def Upscaler_2(upsc_image_input, scale_factor, model_ups):
    tmp_img_ndr = Image.fromarray(upsc_image_input)
    upsc_image_output = upscale(model_ups, tmp_img_ndr, scale_factor)
    
    return upsc_image_output

def Upscaler_batch(upsc_dir_input, scale_factor_batch, model_ups_batch):
    output_dir = "outputs/upscaler_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in tqdm(os.listdir(upsc_dir_input)):
        input_path = os.path.join(upsc_dir_input, filename)
        try:
            if os.path.isfile(input_path):
                img = Image.open(input_path)
                img_array = np.array(img)
                
                upsc_image_output = Upscaler_2(img_array, scale_factor_batch, model_ups_batch)
                
                output_path = os.path.join(output_dir, filename)
                upsc_image_output.save(output_path)
        except Exception as e:
            gr.Error(e)
            
    output_dir = f"Your outputs here: {output_dir}"
    CODC_clear(silent=True)
    return output_dir

def Upscaler_batch_Clear():
    outputs_dir = os.path.join(config.current_directory, "outputs/upscaler_outputs")
    sh.rmtree(outputs_dir)
    folder_path = "outputs/upscaler_outputs"
    os.makedirs(folder_path)
    file = open(f"{folder_path}/outputs will be here.txt", "w")
    file.close()
    gr.Info("Upscaler outputs cleared")

##################################################################################################################################

def ImageAnalyzer(file_spc, clip_checked, clip_chunk_size):
    img = Image.fromarray(file_spc, 'RGB')
    img.save('tmp.png')
    dir_img_fromarray = os.path.join(os.getcwd(), "tmp.png")
    
    spc_output = ""
    
    if clip_checked is True:
        ci = models.ci_load(clip_chunk_size)
        clip = Image.open(dir_img_fromarray).convert('RGB')

        if config.debug:
            gr.Info(f"Cache path: {ci.config.cache_path}")
            
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            spc_output += f"Prompt: {ci.interrogate(clip)}\n\n"
   
    model_nsfw = models.nsfw_load()
   
    result = predict.classify(model_nsfw, dir_img_fromarray)
    x = next(iter(result.keys()))

    values = result[x]
    total_sum = sum(values.values())
    percentages = {k: round((v / total_sum) * 100, 1) for k, v in values.items()}

    spc_output += f"Summary: {nsfw_ng(img)}\n\n"
    spc_output += f"Drawings: {percentages['drawings']}%\n"
    spc_output += f"Porn: {percentages['porn']}%\n"
    spc_output += f"Hentai: {percentages['hentai']}%\n"
    spc_output += f"Sexy: {percentages['sexy']}%\n"
    spc_output += f"Neutral: {percentages['neutral']}%"

    tmp_file = "tmp.png"
    
    try:
        os.remove(tmp_file)
    except FileNotFoundError as e:
        gr.Error(f"Error: {e}")
        pass
    
    del model_nsfw
    if clip_checked is True:
        del ci
    CODC_clear(silent=True)
    return spc_output

##################################################################################################################################

def VideoAnalyzer(file_Vspc):
    model, processor = models.nsfw_ng_load()
    
    cap = cv2.VideoCapture(file_Vspc)
    
    nsfw_count = 0
    normal_count = 0
    file_count = 0
    
    batch_images = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            batch_images.append(image)
            
            pbar.update(1)
            
            if len(batch_images) >= 16:
                nsfw_count, normal_count, file_count = process_batch(
                    model, processor, batch_images, nsfw_count, normal_count, file_count
                )
                batch_images = []
    
        if batch_images:
            nsfw_count, normal_count, file_count = process_batch(
                model, processor, batch_images, nsfw_count, normal_count, file_count
            )
    
    nsfw_percent = (nsfw_count / file_count) * 100 if file_count > 0 else 0
    
    Vspc_output = f"NSFW: {nsfw_count} frames\n"
    Vspc_output += f"Normal: {normal_count} frames\n\n"
    Vspc_output += f"NSFW percent: {nsfw_percent:.2f}%"
    
    cap.release()
    cv2.destroyAllWindows()
    CODC_clear(silent=True)
    
    return Vspc_output

def process_batch(model, processor, batch_images, nsfw_count, normal_count, file_count):
    with torch.no_grad():
        inputs = processor(images=batch_images, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        
        predicted_labels = logits.argmax(-1).tolist()
        for label in predicted_labels:
            predicted_class = model.config.id2label[label]
            if predicted_class == "nsfw":
                nsfw_count += 1
            else:
                normal_count += 1
            file_count += 1
    
    return nsfw_count, normal_count, file_count

def process_batch_2(model, processor, batch_images):
    nsfw_count = 0
    normal_count = 0
    
    with torch.no_grad():
        inputs = processor(images=batch_images, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_labels = logits.argmax(-1)
        
        for label in predicted_labels:
            predicted_class = model.config.id2label[label.item()]
            if predicted_class == "nsfw":
                nsfw_count += 1
            else:
                normal_count += 1

    return nsfw_count, normal_count

def process_frame(frame):
    result_frame = frame    
    return result_frame

def VideoAnalyzerBatch(video_dir, vbth_slider, threshold_Vspc_slider):  
    model, processor = models.nsfw_ng_load()
    _nsfw = 0
    _plain = 0
    output_dir = 'tmp'
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = os.listdir(video_dir)
    
    for dir_Vspc in tqdm(video_files, desc="Processing videos"):
        cap = cv2.VideoCapture(os.path.join(video_dir, dir_Vspc))
        frame_count = 0
        batch_images = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm(total=total_frames, desc=f"Processing frames in {dir_Vspc}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if vbth_slider != 1 and frame_count % vbth_slider != 0:
                    frame_count += 1
                    pbar.update(1)
                    continue

                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                batch_images.append(image)
                
                pbar.update(1)
                frame_count += 1

                if len(batch_images) >= 16:
                    nsfw_count, normal_count = process_batch_2(model, processor, batch_images)
                    batch_images = []

            if batch_images:
                nsfw_count, normal_count = process_batch_2(model, processor, batch_images)
        
        total_frames_classified = nsfw_count + normal_count
        nsfw_percent = (nsfw_count / total_frames_classified) * 100 if total_frames_classified > 0 else 0
        
        if nsfw_percent > threshold_Vspc_slider:
            video_path = os.path.join(video_dir, dir_Vspc)
            sh.copy(video_path, 'outputs/video_analyze_nsfw')
            _nsfw += 1
        else:
            video_path = os.path.join(video_dir, dir_Vspc)
            sh.copy(video_path, 'outputs/video_analyze_plain')
            _plain += 1
            
        cap.release()
        cv2.destroyAllWindows()
        
        rm_tmp = os.path.join(config.current_directory, output_dir)
        sh.rmtree(rm_tmp)
        os.makedirs(output_dir, exist_ok=True)

    del model
    CODC_clear(silent=True)
    return f"NSFW: {_nsfw} | Plain: {_plain}"

def VideoAnalyzerBatch_Clear():
    output_dir = 'tmp'
    try:
        outputs_dir1 = os.path.join(config.current_directory, "outputs/video_analyze_nsfw")
        sh.rmtree(outputs_dir1)
        outputs_dir2 = os.path.join(config.current_directory, "outputs/video_analyze_plain")
        sh.rmtree(outputs_dir2)
        outputs_dir3 = os.path.join(config.current_directory, "tmp")
        sh.rmtree(outputs_dir3)
        folder_path1 = "outputs/video_analyze_nsfw"
        os.makedirs(folder_path1)
        file = open(f"{folder_path1}/outputs will be here.txt", "w")
        file.close()
        folder_path2 = "outputs/video_analyze_plain"
        os.makedirs(folder_path2)
        file = open(f"{folder_path2}/outputs will be here.txt", "w")
        file.close()
        rm_tmp = os.path.join(config.current_directory, output_dir)
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
    
    gr.Info("Video Analyzer outputs cleared")    

##################################################################################################################################

def PromptGenetator(prompt_input, pg_prompts, pg_max_length, randomize_temp):
    tokenizer, model_tokinezer = models.tokenizer_load()
    prompt = prompt_input
    if randomize_temp is True:
        tempreture_pg = (random.randint(4, 9)/10)
    elif randomize_temp is False:
        tempreture_pg = 0.7

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
    
    del tokenizer, model_tokinezer
    CODC_clear(silent=True)
    return promptgen_output 

##################################################################################################################################

def is_image_generated(test_image, model_h5): 
    
    test_image = test_image.resize((512, 512))  
    test_image = test_image.convert('RGB')  
    test_image = np.array(test_image) / 255.0 

    test_image = np.expand_dims(test_image, axis=0)  

    result = model_h5.predict(test_image)
    
    predicted_prc_ai_raw = result[0][0] * 100 
    predicted_prc_human_raw = result[0][1] * 100 
    
    predicted_prc_ai = round(predicted_prc_ai_raw, 4)
    predicted_prc_human = round(predicted_prc_human_raw, 4)
 
    if predicted_prc_ai >= predicted_prc_human:
        predicted_prc = predicted_prc_ai
    elif predicted_prc_human >= predicted_prc_ai:
        predicted_prc = predicted_prc_human
            
    if predicted_prc == predicted_prc_ai:
        iig_text = f"This is an image created by AI\n\nAI: {predicted_prc_ai}%\nHUMAN: {predicted_prc_human}%"
        return iig_text
    elif predicted_prc == predicted_prc_human:
        iig_text = f"This is an image created by HUMAN\n\nAI: {predicted_prc_ai}%\nHUMAN: {predicted_prc_human}"
        return iig_text


def AiDetector_single(aid_input_single):  
    model_h5 = models.h5_load()
    img_h5 = Image.fromarray(aid_input_single)
    aid_output_single = is_image_generated(img_h5, model_h5)
    
    del model_h5
    CODC_clear(silent=True)
    return aid_output_single

def AiDetector_batch(aid_input_batch): 
    model_h5 = models.h5_load()
    if config.debug:
        print(f"Working in: {aid_input_batch}")

    aid_ai_dir = os.path.join(config.current_directory, "outputs/aid_ai")
    aid_human_dir = os.path.join(config.current_directory, "outputs/aid_human")
    
    if not os.path.exists(aid_ai_dir):
        os.makedirs(aid_ai_dir)
        if config.debug:
            print(f"Created AI directory: {aid_ai_dir}")
    if not os.path.exists(aid_human_dir):
        os.makedirs(aid_human_dir)
        if config.debug:
            print(f"Created HUMAN directory: {aid_human_dir}")
    
    image_files = os.listdir(aid_input_batch)
    
    for image_file in tqdm(image_files):
        try:
            img_path = os.path.join(aid_input_batch, image_file)
            
            img_h5 = Image.open(img_path)
            result = is_image_generated(img_h5, model_h5)
              
            if "This is an image created by AI" in result:
                dest_path = os.path.join(aid_ai_dir, image_file)
                sh.copyfile(img_path, dest_path)
            elif "This is an image created by HUMAN" in result:
                dest_path = os.path.join(aid_human_dir, image_file)
                sh.copyfile(img_path, dest_path)
        
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            pass
    
    del model_h5
    CODC_clear(silent=True)
    aid_output_batch = "Images sorted successfully!"
    return aid_output_batch

def AID_Clear():
    outputs_dir1 = os.path.join(config.current_directory, "outputs/aid_ai")
    sh.rmtree(outputs_dir1)
    outputs_dir2 = os.path.join(config.current_directory, "outputs/aid_human")
    sh.rmtree(outputs_dir2)
    folder_path1 = "outputs/aid_ai"
    os.makedirs(folder_path1)
    file = open(f"{folder_path1}/outputs will be here.txt", "w")
    file.close()
    folder_path2 = "outputs/aid_human"
    os.makedirs(folder_path2)
    file = open(f"{folder_path2}/outputs will be here.txt", "w")
    file.close()
    gr.Info("AI Detecting outputs cleared")

##################################################################################################################################

def silero_tts(tts_input, tts_lang, tts_speakers, tts_rate):
    tts_model = models.silero_tts_load(localization=tts_lang)
    tts_model.save_wav(text=tts_input, speaker=tts_speakers, sample_rate=tts_rate)
    
    src_path = 'test.wav'
    dest_dir = 'outputs/tts'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    current_time = datetime.now().strftime("%d.%m.%y_%H-%M")
    new_filename = f"tts_{current_time}.wav"
    dest_path = os.path.join(dest_dir, new_filename)
    sh.move(src_path, dest_path)
    
    wav_file = dest_path
    
    CODC_clear(silent=True)
    return wav_file

def tts_clear():
    outputs_dir1 = os.path.join(config.current_directory, "outputs/tts")
    sh.rmtree(outputs_dir1)
    folder_path1 = "outputs/tts"
    os.makedirs(folder_path1)
    file = open(f"{folder_path1}/outputs will be here.txt", "w")
    file.close()
    gr.Info("TTS outputs cleared")    

##################################################################################################################################

def CODC_clear(silent):
    try:
        if not silent:
            gr.Info("Clearing cache...")
        
        torch.cuda.empty_cache()
        
        clear_on_device_caches()
        
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()
        
        gc.collect()
         
        if not silent:
            gr.Info("All cache cleared!")   
    except Exception:
        gr.Warning("Something wrong in cache clearing. Contact dev.")
        gr.Info("All cache cleared?...")
        
def CODC_clear_app():
    try:
        gr.Info("Clearing cache...")
        
        torch.cuda.empty_cache()
        
        clear_on_device_caches()
        
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()
        
        gc.collect()
         
        gr.Info("All cache cleared!")   
    except Exception:
        gr.Warning("Something wrong in cache clearing. Contact dev.")
        gr.Info("All cache cleared?...")
