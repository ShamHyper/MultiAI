import gradio as gr

from nsfw_detector import predict
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from clip_interrogator import Config, Interrogator
from keras.models import load_model
from transformers import AutoModelForImageClassification, ViTImageProcessor
import torch

import os
import urllib

from config import current_directory
import config

modelname = "nsfw_mobilenet2.224x224.h5"
url = "https://vdmstudios.ru/server_archive/nsfw_mobilenet2.224x224.h5"

modelname_h5 = "model_2.0.h5"
url_h5 = "https://vdmstudios.ru/server_archive/model_2.0.h5"

models_directory = os.path.join(current_directory, "models")

if not os.path.exists(models_directory):
    os.makedirs(models_directory)

def check_file(filename):
    file_path = os.path.join(models_directory, filename)
    files_in_directory = os.listdir(models_directory)

    if filename in files_in_directory:
        if config.debug: 
            gr.Info("NSFW Model detected")
    else:
        if config.debug: 
            gr.Info("NSFW Model undetected. Downloading...")
        urllib.request.urlretrieve(url, file_path)

def checkfile_h5(filename):
    file_path = os.path.join(models_directory, filename)
    files_in_directory = os.listdir(models_directory)

    if filename in files_in_directory:
        if config.debug:
            gr.Info("H5 Model detected")
    else:
        if config.debug: 
            gr.Info("H5 Model undetected. Downloading...")
        urllib.request.urlretrieve(url_h5, file_path)
 
def nsfw_load():
    try:
        if config.debug:
            gr.Info("Loading NSFW model...")
        check_file(modelname)
        model_nsfw = predict.load_model("models/nsfw_mobilenet2.224x224.h5")
    except NameError:
        gr.Error("Error in nsfw_load!")
    return model_nsfw

def tokenizer_load():
    try:
        if config.debug:
            gr.Info("Loading Tokenizer model...")
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model_tokinezer = GPT2LMHeadModel.from_pretrained('FredZhang7/anime-anything-promptgen-v2')
    except NameError:
        gr.Error("Error in tokenizer_load!")
    return tokenizer, model_tokinezer

def ci_load(clip_chunk_size):
    try:
        if config.debug:
            gr.Info("Loading CLIP model...")  
            
        ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k", chunk_size=clip_chunk_size, flavor_intermediate_count=clip_chunk_size))
        
        if config.debug:
            gr.Info("Using laion2b_s32b_b79k model")
            gr.Info(f"Chunk size: {clip_chunk_size}")          
    except (NameError, RuntimeError, UnboundLocalError):
        gr.Error("Error in ci_load!")
        ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k", chunk_size=clip_chunk_size, flavor_intermediate_count=clip_chunk_size))
    return ci

def h5_load():
    try:
        if config.debug:
            gr.Info("Loading H5 model...")
        checkfile_h5(modelname_h5)
        model_h5 = load_model('models/model_2.0.h5')
    except NameError:
        gr.Error("Error in h5_load!")
    return model_h5

def nsfw_ng_load():
    try:
        if config.debug:
            gr.Info("Loading nsfw_ng model...")
        model_nsfw_ng = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
        processor_nsfw_ng = ViTImageProcessor.from_pretrained('Falconsai/nsfw_image_detection')
    except NameError:
        gr.Error("Error in nsfw_ng_load!")
    return model_nsfw_ng, processor_nsfw_ng

def silero_tts_load(localization):
    try:
        if config.debug:
            gr.Info("Loading TTS model...")
        
        device = config.check_gpu()
        torch.set_num_threads(4)
        
        if localization == "ru":
            local_file = os.path.join(models_directory, 'v4_ru.pt')
            url = 'https://models.silero.ai/models/tts/ru/v4_ru.pt'
        elif localization == "en":
            local_file = os.path.join(models_directory, 'v3_en.pt')
            url = 'https://models.silero.ai/models/tts/en/v3_en.pt'
        else:
            raise ValueError("Unsupported localization")

        if not os.path.isfile(local_file):
            torch.hub.download_url_to_file(url, local_file)
        
        tts_model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
        tts_model.to(device)
    
    except Exception as e:
        if config.debug:
            gr.Error(f"Error in silero_tts_load: {str(e)}")
        tts_model = None

    return tts_model
        
voices_ru = ['random', 'aidar', 'baya', 'kseniya', 'xenia', 'eugene']

voices_en = [
    'random', 'en_0', 'en_1', 'en_2', 'en_3', 'en_4', 'en_5', 'en_6', 'en_7', 'en_8', 'en_9',
    'en_10', 'en_11', 'en_12', 'en_13', 'en_14', 'en_15', 'en_16', 'en_17', 'en_18', 'en_19',
    'en_20', 'en_21', 'en_22', 'en_23', 'en_24', 'en_25', 'en_26', 'en_27', 'en_28', 'en_29',
    'en_30', 'en_31', 'en_32', 'en_33', 'en_34', 'en_35', 'en_36', 'en_37', 'en_38', 'en_39',
    'en_40', 'en_41', 'en_42', 'en_43', 'en_44', 'en_45', 'en_46', 'en_47', 'en_48', 'en_49',
    'en_50', 'en_51', 'en_52', 'en_53', 'en_54', 'en_55', 'en_56', 'en_57', 'en_58', 'en_59',
    'en_60', 'en_61', 'en_62', 'en_63', 'en_64', 'en_65', 'en_66', 'en_67', 'en_68', 'en_69',
    'en_70', 'en_71', 'en_72', 'en_73', 'en_74', 'en_75', 'en_76', 'en_77', 'en_78', 'en_79',
    'en_80', 'en_81', 'en_82', 'en_83', 'en_84', 'en_85', 'en_86', 'en_87', 'en_88', 'en_89',
    'en_90', 'en_91', 'en_92', 'en_93', 'en_94', 'en_95', 'en_96', 'en_97', 'en_98', 'en_99',
    'en_100', 'en_101', 'en_102', 'en_103', 'en_104', 'en_105', 'en_106', 'en_107', 'en_108', 
    'en_109', 'en_110', 'en_111', 'en_112', 'en_113', 'en_114', 'en_115', 'en_116', 'en_117'
]

sample_rates = [8000, 24000, 48000]