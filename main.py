import shutil as sh
import os

import urllib.request

import gradio as gr

import config
import models

import torch

version = "MultiAI | v1.14.0-b2"

##################################################################################################################################

ver = version

print(f"Initializing {ver} launch...")

current_directory = os.path.dirname(os.path.abspath(__file__))

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

def delete_tmp_pngs():
    output_dir = "tmp"
    try:
        rm_tmp = os.path.join(current_directory, output_dir)
        sh.rmtree(rm_tmp)
    except (PermissionError, FileNotFoundError, FileExistsError, Exception):
        pass
    
    tmp_file = "tmp.png"
    
    try:
        os.remove(tmp_file)
    except FileNotFoundError as e:
        gr.Error(f"Error: {e}")
        pass

def preloader():
    print("Preloading models...")
    models.ci_load()
    models.nsfw_load()
    models.tokenizer_load()  
    models.h5_load()
    preloaded_tb = "Done!"
    return preloaded_tb

def clear_all():
    import multi
    multi.BgRemoverLite_Clear()
    multi.NSFWDetector_Clear()
    multi.VideoAnalyzerBatch_Clear()
    multi.AID_Clear()
    gr.Info("All outputs cleared!")
    clear_all_tb = "All outputs deleted!"
    return clear_all_tb

def check_gpu():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config.debug:
        gr.Info(f"Device: {device}")
        
    if torch.cuda.is_available():
        gr.Info("CUDA available! Working on")
    elif not torch.cuda.is_available():
        gr.Warning("CUDA is not available, using CPU. Warning: this will be very slow!")
    return device

