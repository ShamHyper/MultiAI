import gradio as gr

from nsfw_detector import predict
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from clip_interrogator import Config, Interrogator
from keras.models import load_model

import main
import config
 
def nsfw_load():
    try:
        if config.debug:
            gr.Info("Loading NSFW model...")
        main.check_file(main.modelname)
        model_nsfw = predict.load_model("nsfw_mobilenet2.224x224.h5")
    except NameError:
        gr.Error("Error in nsfw_load!")
        main.check_file(main.modelname)
        model_nsfw = predict.load_model("nsfw_mobilenet2.224x224.h5")
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
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model_tokinezer = GPT2LMHeadModel.from_pretrained('FredZhang7/anime-anything-promptgen-v2')
    return tokenizer, model_tokinezer

def ci_load(clip_chunk_size):
    try:
        if config.debug:
            gr.Info("Loading CLIP model...")  
        ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k", chunk_size=clip_chunk_size, flavor_intermediate_count=clip_chunk_size))
        if config.debug:
            gr.Info("Using laion2b_s32b_b79k model")
            gr.Info(f"Chunk size: {clip_chunk_size}")          
    except (NameError, RuntimeError):
        gr.Error("Error in ci_load!")
        ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k", chunk_size=clip_chunk_size, flavor_intermediate_count=clip_chunk_size))
        if config.debug: 
            gr.Info("Using laion2b_s32b_b79k model")
            gr.Info(f"Chunk size: {clip_chunk_size}")
    return ci

def h5_load():
    try:
        if config.debug:
            gr.Info("Loading H5 model...")
        main.checkfile_h5(main.modelname_h5)
        model_h5 = load_model('model_2.0.h5')
    except NameError:
        gr.Error("Error in h5_load!")
        main.checkfile_h5(main.modelname_h5)
        model_h5 = load_model('model_2.0.h5')
    return model_h5