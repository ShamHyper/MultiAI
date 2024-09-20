import os
import json
import gradio as gr

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
preload_models = data.get('preload_models', 'False').lower() == 'true'
clear_on_start = data.get('clear_on_start', 'False').lower() == 'true'

if not (debug or inbrowser or share_gradio or preload_models or clear_on_start):
    print("Something wrong in config.json. Check them out!")