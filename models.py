import gradio as gr

from nsfw_detector import predict
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from clip_interrogator import Config, Interrogator
from keras.models import load_model
from transformers import AutoModelForImageClassification, ViTImageProcessor

import os
import urllib

from config import current_directory
import config

modelname = "nsfw_mobilenet2.224x224.h5"
url = "https://vdmstudios.ru/server_archive/nsfw_mobilenet2.224x224.h5"

modelname_h5 = "model_2.0.h5"
url_h5 = "https://vdmstudios.ru/server_archive/model_2.0.h5"

def check_file(filename):
    files_in_directory = os.listdir(current_directory)

    if filename in files_in_directory:
        if config.debug: 
            gr.Info("NSFW Model detected")
    else:
        if config.debug: 
            gr.Info("NSFW Model undected. Downloading...")
        urllib.request.urlretrieve(url, modelname)

def checkfile_h5(filename):
    files_in_directory = os.listdir(current_directory)
    
    if filename in files_in_directory:
        if config.debug:
            gr.Info("H5 Model detected")
    else:
        if config.debug: 
            gr.Info("H5 Model undected. Downloading...")
        urllib.request.urlretrieve(url_h5, modelname_h5)
 
def nsfw_load():
    try:
        if config.debug:
            gr.Info("Loading NSFW model...")
        check_file(modelname)
        model_nsfw = predict.load_model("nsfw_mobilenet2.224x224.h5")
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
        model_h5 = load_model('model_2.0.h5')
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
        
        
