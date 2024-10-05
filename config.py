import os
import json
import gradio as gr
import torch
import torchvision
import shutil as sh

def save_config_gr(settings_debug_mode, settings_start_in_browser, settings_share_gradio, settings_clear_on_start, settings_use_proxy):
    settings = {
        "debug_mode": str(settings_debug_mode),
        "start_in_browser": str(settings_start_in_browser),
        "share_gradio": str(settings_share_gradio),
        "clear_on_start": str(settings_clear_on_start),
        "use_proxy": str(settings_use_proxy)
    }
    
    if "dev_config.json" in os.listdir("settings"):
        json_file = "dev_config.json"
    elif "config.json" in os.listdir("settings"):
        json_file = "config.json"
        
    json_file = "settings/" + json_file

    with open(json_file, 'w') as file:
        json.dump(settings, file, indent=4)
        
    settings_save_progress_toast = f"Settings saved to [{json_file}]. Restart MultiAI!"
    settings_save_progress = "Done!"
    
    gr.Info(settings_save_progress_toast)
    
    return settings_save_progress

if "dev_config.json" in os.listdir("settings"):
    with open("settings/dev_config.json") as json_file:
        data = json.load(json_file)
        print("dev_config.json loaded")
elif "config.json" in os.listdir("settings"):
    with open("settings/config.json") as json_file:
        data = json.load(json_file)
        print("config.json loaded")

debug = data.get('debug_mode', 'False').lower() == 'true'
inbrowser = data.get('start_in_browser', 'False').lower() == 'true'
share_gradio = data.get('share_gradio', 'False').lower() == 'true'
clear_on_start = data.get('clear_on_start', 'False').lower() == 'true'
use_proxy = data.get('use_proxy', 'False').lower() == 'true'

if not (debug or inbrowser or share_gradio or clear_on_start or use_proxy):
    print("Something wrong in config.json. Check them out!")
    
cuda_version = torch.version.cuda
cudnn_version = torch.backends.cudnn.version()
torch_version = torch.__version__
torchvision_version = torchvision.__version__
current_directory = os.path.dirname(os.path.abspath(__file__))

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

def clear_all():
    import multi
    multi.BgRemoverLite_Clear()
    multi.NSFWDetector_Clear()
    multi.VideoAnalyzerBatch_Clear()
    multi.AID_Clear()
    multi.tts_clear()
    gr.Info("All outputs cleared!")

def check_gpu():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    try:
        if debug:
            gr.Info(f"Device: {device}")
            print(f"Allocated memory: {torch.cuda.memory_allocated()} bytes")
            print(f"Reserved memory: {torch.cuda.memory_reserved()} bytes")
            print("")
        if torch.cuda.is_available():
            gr.Info("CUDA available! Working on")
        elif not torch.cuda.is_available():
            gr.Warning("CUDA is not available, using CPU. Warning: this will be very slow!")
    except Exception as e:
        print(f"ERROR IN CHECK GPU: {e}")
    return device

