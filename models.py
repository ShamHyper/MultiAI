import gradio as gr

from nsfw_detector import predict
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from clip_interrogator import Config, Interrogator
from keras.models import load_model

import main
import config

def nsfw_load():
    try:
        if nsfw_status is not True:
            main.check_file(main.modelname)
            model_nsfw = predict.load_model("nsfw_mobilenet2.224x224.h5")
            nsfw_status = True
        elif nsfw_status is True:
            if config.debug: 
                gr.Info("NSFW model already loaded!")
    except NameError:
            main.check_file(main.modelname)
            model_nsfw = predict.load_model("nsfw_mobilenet2.224x224.h5")
            nsfw_status = True
    return model_nsfw, nsfw_status

def tokenizer_load():
    try:
        if tokenizer_status is not True:
            tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model_tokinezer = GPT2LMHeadModel.from_pretrained('FredZhang7/anime-anything-promptgen-v2')
            tokenizer_status = True
        elif tokenizer_status is True:
            if config.debug: 
                gr.Info("Tokinezer already loaded!")
    except NameError:
            tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model_tokinezer = GPT2LMHeadModel.from_pretrained('FredZhang7/anime-anything-promptgen-v2')
            tokenizer_status = True
    return tokenizer, tokenizer_status, model_tokinezer

def ci_load():
    try:
        if ci_status is not True:
            ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k"))
            ci_status = True
        elif ci_status is True:
            if config.debug: 
                gr.Info("CLIP already loaded!")
    except NameError:
            ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k"))
            ci_status = True
    return ci, ci_status

def h5_load():
    global model_h5, h5_status
    try:
        if h5_status is not True:
            main.checkfile_h5(main.modelname_h5)
            model_h5 = load_model('model_2.0.h5')
            h5_status = True
        elif h5_status is True:
            if config.debug: 
                gr.Info("H5 model already loaded!")
    except NameError:
            main.checkfile_h5(main.modelname_h5)
            model_h5 = load_model('model_2.0.h5')
            h5_status = True
    return model_h5, h5_status