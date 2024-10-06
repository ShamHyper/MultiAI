import gradio as gr
from clear import clear
from upscalers import available_models

import config
import multi
import models

import os
import sys
import time
from PIL import Image

if config.use_proxy is False:
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

##################################################################################################################################

VERSION = "MultiAI v1.16.2"
SERVER_PORT = 7891
SERVER_NAME = '127.0.0.1'

##################################################################################################################################

print(f"Initializing {VERSION} launch...")

print("")

with open("app/css/.css", "r") as file:
    CSS = file.read()
    print("CSS loaded!")
    
with open("app/js/preloader.js", "r") as file:
    JS_SCRIPT_PRELOADER = file.read()
    print("JS_SCRIPT_PRELOADER loaded!")

with open("app/js/script.js", "r") as file:
    JS_SCRIPT = file.read()
    print("JS_SCRIPT loaded!")
    
with open("app/html/.html", "r") as file:
    HTML_FILE = file.read()
    print("HTML_FILE loaded!")
    
def restart_ui():
    gr.Info("Reloading...") 
    time.sleep(0.5)
    os.execv(sys.executable, ['python'] + sys.argv)
    
def update_speakers(lang):
    if lang == "en":
        return gr.Dropdown(choices=models.voices_en, value="random", label="Speaker")
    elif lang == "ru":
        return gr.Dropdown(choices=models.voices_ru, value="random", label="Speaker")

def LIFF(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            try:
                img = Image.open(img_path)
                images.append(img)
            except Exception:
                pass
    return images

def refresh_gallery():
    return LIFF(aid_ai), LIFF(aid_human), LIFF(detector_outputs_nsfw), LIFF(detector_outputs_plain), LIFF(rembg_outputs)
    
##################################################################################################################################


with gr.Blocks(title=VERSION, theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"), css=CSS, js=JS_SCRIPT) as multiai:
    md_text = f'{VERSION} • Torch {config.torch_version} • Torchvision {config.torchvision_version} • CUDA {config.cuda_version} • cuDNN {config.cudnn_version}'
    gr.Markdown(md_text)
    
    
##################################################################################################################################
    
    with gr.Tab("🎗️BgRemoverLite"):
        with gr.Row():
            gr.Label("Remove background from single image")
        with gr.Row():
            image_input = gr.Image(width=200, height=400)
            image_output = gr.Image(width=200, height=400, format="png")
        with gr.Row():
            rembg_button = gr.Button("🖼️ Remove one background")
        with gr.Row():
            gr.Label("Remove background from all images in dir")
        with gr.Row():
            image_input_dir = gr.Textbox(
                label="Enter dir", placeholder="Enter dir like this: D:\Python\MultiAI")
            image_output_dir = gr.Textbox(
                label="Output", placeholder="BrRemoverLite outputs will be here")
        with gr.Row():
            rembg_batch_button = gr.Button("🖼️ Remove some backgrounds")
        with gr.Row():
            gr.Label("Clear outputs")
        with gr.Row():
            clearp_bgr_button = gr.Button("🧹 Clear outputs")
            clearp_bgr = gr.Textbox(label="Clearing progress")
      
##################################################################################################################################
            
    with gr.Tab("⤴️Upscaler"):
        with gr.Row():
            gr.Label("Upscale image up to 10x size")
        with gr.Row():
            upsc_image_input = gr.Image(width=200, height=400)
            upsc_image_output = gr.Image(width=200, height=400, format="png")
        with gr.Row():
            model_ups = gr.Dropdown(label="Model", choices=available_models(), value="None")
            scale_factor = gr.Slider(
                value=4.0,
                label="Scale factor (4x factor max recommended)",
                minimum=2,
                maximum=10.0,
                step=1
            )
            upsc_button = gr.Button("📈 Start upscaling")
            
##################################################################################################################################
            
    with gr.Tab("🔎Image Analyzer"):
        with gr.Row():
            gr.Label("Analyze image")
        with gr.Row():
            file_spc = gr.Image(width=200, height=400)
            spc_output = gr.Textbox(label="Stats", placeholder="Press start to get specifications of image")
        with gr.Row():
            clip_checked = gr.Checkbox(value=False, label="Use CLIP for generate prompt (slow if a weak PC)")
            clip_chunk_size = gr.Slider(value=512, label="Batch size for CLIP, use smaller for lower VRAM", maximum=2048, minimum=512, step=512)
        with gr.Row():
            spc_button = gr.Button("👟 Click here to start")
            
        with gr.Row():
            gr.Label("Detect NSFW images from dir")
        with gr.Row():
            detector_input = gr.Textbox(label="Enter dir", placeholder="Enter dir like this: D:\Python\MultiAI")
            detector_output = gr.Textbox(label="Output", placeholder="NSFW Detector outputs will be here")
        with gr.Row():
            detector_button = gr.Button("👟 Click here to start")
        with gr.Row():
            gr.Label("Clear outputs")
        with gr.Row():
            detector_clear_button = gr.Button("🧹 Clear outputs")
            clearp = gr.Textbox(label="Clearing progress")
            
##################################################################################################################################
            
    with gr.Tab("🎥Video Analyzer"):
        with gr.Row():
            gr.Label("Analyze Video")
        with gr.Row():
            file_Vspc = gr.Video(sources="upload", width=200, height=400)
            Vspc_output = gr.Textbox(label="Stats", placeholder="Press start to get specifications of Video")
        with gr.Row():
            Vspc_button = gr.Button("👟 Click here to start")   
        with gr.Row():
            gr.Label("Analyze Videos in dir")     
        with gr.Row():
            video_dir = gr.Textbox(label="Videos dir", placeholder="Enter dir like this: D:\Python\MultiAI")
            bth_Vspc_output = gr.Textbox(label="Output", placeholder="Output will be here...")
        with gr.Row():
            vbth_slider = gr.Slider(
            value=80,
            step=1,
            label="Frame-Skip (larger number = simpler detection | smaller number = stricter one)",
            minimum=1,
            maximum=100,
            info="Set 1 to turn off. If an error appears, decrease the value"
        )  
            threshold_Vspc_slider = gr.Slider(
            value=20,
            minimum=2,
            maximum=98,
            step=1,
            label="Threshold (larger number = simpler detection | smaller number = stricter one)"
        )
        with gr.Row():
            start_dir_videos = gr.Button("⭐ Start")
        with gr.Row():
            clear_videos = gr.Button("🧹 Clear outputs")
            bth_Vspc_clear_output = gr.Textbox(label="Clearing progress", placeholder="Clear output will be here...")
            
##################################################################################################################################
            
    with gr.Tab("💬Prompt Generator"):
        with gr.Row():
            gr.Label("Generate prompt from your input")
        with gr.Row():
            prompt_input = gr.Textbox(label="Your input", placeholder="Type something...")
            promptgen_output = gr.Textbox(label="Output prompts", placeholder="Your output will be here...")
        with gr.Row():
            randomize_temp = gr.Checkbox(value=True, label="Randomize prompt temperature")
            pg_prompts = gr.Slider(
                value=1,
                label="Prompts",
                minimum=1,
                maximum=1000
            )
            pg_max_length = gr.Slider(
                value=76,
                label="Max Length of prompt",
                minimum=1,
                maximum=1000
            )
        with gr.Row():
            promptgen_button = gr.Button("⭐ Start")
            
##################################################################################################################################            
            
    with gr.Tab("🤖AI Detector"):
        with gr.Row():
            gr.Label("Analyze image")
        with gr.Row():
            aid_input_single = gr.Image(width=200, height=400)
            aid_output_single = gr.Textbox(label="Result", placeholder="Press start to get result")
        with gr.Row():
            aid_single_button = gr.Button("👟 Click here to start")                

        with gr.Row():
            gr.Label("Detect AI/Human images from dir")
        with gr.Row():
            aid_input_batch = gr.Textbox(label="Enter dir", placeholder="Enter dir like this: D:\Python\MultiAI")
            aid_output_batch = gr.Textbox(label="Output", placeholder="AI Detector outputs will be here")
        with gr.Row():
            aid_batch_button = gr.Button("👟 Click here to start")
                    
        with gr.Row():
            gr.Label("Clear outputs")
        with gr.Row():
            aid_clear_button = gr.Button("🧹 Clear outputs")
            aid_clearp = gr.Textbox(label="Clearing progress")
                  
##################################################################################################################################

    with gr.Tab("🔊TTS"):
        with gr.Row():
            gr.Label("Text input")
            gr.Label("Audio output")
        with gr.Row():
            tts_input = gr.Textbox(label="Your input", placeholder="Type something...")
            tts_audio = gr.Audio(label="Audio")
        with gr.Row():
            tts_lang = gr.Dropdown(label="Language", choices=["en", "ru"], value="en")
            tts_speakers = gr.Dropdown(label="Speaker", choices=models.voices_en, value="random")
            tts_rate = gr.Dropdown(label="Sample rate", choices=models.sample_rates, value=48000)
        with gr.Row():
            tts_button = gr.Button("👟 Click here to start")
            tts_clear = gr.Button("🧹 Clear outputs")

##################################################################################################################################

    with gr.Tab("🖼️Outputs"):
        with gr.Row():
            refresh_button = gr.Button("🔄Reload gallery")
        with gr.Row():
            gr.Label("AI Detector - AI")
        with gr.Row():
            aid_ai = "outputs/aid_ai"
            gallery_aid_ai = gr.Gallery(value=LIFF(aid_ai), format="png", interactive=False, columns=8, container=False)
        with gr.Row():
            gr.Label("AI Detector - HUMAN")
        with gr.Row():
            aid_human = "outputs/aid_human"
            gallery_aid_human = gr.Gallery(value=LIFF(aid_human), format="png", interactive=False, columns=8, container=False)
        with gr.Row():
            gr.Label("Analyzer - NSFW")
        with gr.Row():
            detector_outputs_nsfw = "outputs/detector_outputs_nsfw"
            gallery_detector_outputs_nsfw = gr.Gallery(value=LIFF(detector_outputs_nsfw), format="png", interactive=False, columns=8, container=False)
        with gr.Row():
            gr.Label("Analyzer - PLAIN")
        with gr.Row():
            detector_outputs_plain = "outputs/detector_outputs_plain"
            gallery_detector_outputs_plain = gr.Gallery(value=LIFF(detector_outputs_plain), format="png", interactive=False, columns=8, container=False)
        with gr.Row():
            gr.Label("Rembg")
        with gr.Row():
            rembg_outputs = "outputs/rembg_outputs"
            gallery_rembg_outputs = gr.Gallery(value=LIFF(rembg_outputs), format="png", interactive=False, columns=8, container=False)

##################################################################################################################################

    with gr.Tab("⚙️Settings"):
        with gr.Row():
            settings_debug_mode = gr.Checkbox(value=config.debug, label="Enable debug mode (write debug info)")
        with gr.Row():
            settings_start_in_browser = gr.Checkbox(value=config.inbrowser, label="Enable MultiAI starting in browser")
        with gr.Row():
            settings_share_gradio = gr.Checkbox(value=config.share_gradio, label="Enable MultiAI starting with share link")
        with gr.Row():
            settings_clear_on_start = gr.Checkbox(value=config.clear_on_start, label="Enable clear all outputs on MultiAI start")
        with gr.Row():
            settings_use_proxy = gr.Checkbox(value=config.use_proxy, label="Enable using system proxy for connect to WebUI. NEED TO FULL RESTART (CLOSING CMD AND RUNNING .BAT)")
        with gr.Row():
            json_files = gr.Label("Saving in [../settings/config.json]")
            settings_save = gr.Button("🗃️ Save settings")
            settings_save_progress = gr.Textbox(label="Saving progress", placeholder="Your saving progress will be here")
        with gr.Row():
            gr.Label("Creating new ones .json files in ../settings will not give any effect.")
        with gr.Row():
            clear_all_button = gr.Button("⭐ Clear all outputs")
            upsc_clear_cache = gr.Button("🧹 Clear torch, cuda and models cache") 
            check_torch = gr.Button("👾 Check cuda avaible")
            btn_refresh = gr.Button(value="🔁 Restart MultiAI")
            
##################################################################################################################################
                     
    rembg_button.click(multi.BgRemoverLite, inputs=image_input, outputs=image_output)
    rembg_batch_button.click(multi.BgRemoverLiteBatch, inputs=image_input_dir, outputs=image_output_dir)
    clearp_bgr_button.click(multi.BgRemoverLite_Clear, outputs=clearp_bgr)

    detector_button.click(multi.NSFW_Detector, inputs=detector_input, outputs=detector_output)
    detector_clear_button.click(multi.NSFWDetector_Clear, outputs=clearp)
    
    upsc_button.click(multi.Upscaler, inputs=[upsc_image_input, scale_factor, model_ups], outputs=upsc_image_output)
    upsc_clear_cache.click(multi.CODC_clear_app)
    
    spc_button.click(multi.ImageAnalyzer, inputs=[file_spc, clip_checked, clip_chunk_size], outputs=spc_output)
       
    Vspc_button.click(multi.VideoAnalyzer, inputs=file_Vspc, outputs=Vspc_output)
    start_dir_videos.click(multi.VideoAnalyzerBatch, inputs=[video_dir, vbth_slider, threshold_Vspc_slider], outputs=bth_Vspc_output)   
    clear_videos.click(multi.VideoAnalyzerBatch_Clear, outputs=bth_Vspc_clear_output)   
    
    promptgen_button.click(multi.PromptGenetator, inputs=[prompt_input, pg_prompts, pg_max_length, randomize_temp], outputs=promptgen_output)
    
    aid_single_button.click(multi.AiDetector_single, inputs=aid_input_single, outputs=aid_output_single)
    aid_batch_button.click(multi.AiDetector_batch, inputs=aid_input_batch, outputs=aid_output_batch)
    aid_clear_button.click(multi.AID_Clear, outputs=aid_clearp)
    
    clear_all_button.click(config.clear_all)
    
    check_torch.click(config.check_gpu)
    
    settings_save.click(config.save_config_gr, inputs=[settings_debug_mode, settings_start_in_browser, settings_share_gradio, settings_clear_on_start, settings_use_proxy]
                        , outputs=settings_save_progress)
    
    btn_refresh.click(restart_ui, js=JS_SCRIPT_PRELOADER)
    
    tts_lang.change(update_speakers, inputs=tts_lang, outputs=tts_speakers)
    tts_button.click(multi.silero_tts, inputs=[tts_input, tts_lang, tts_speakers, tts_rate], outputs=tts_audio)
    tts_clear.click(multi.tts_clear)
    
    refresh_button.click(refresh_gallery, outputs=[gallery_aid_ai, gallery_aid_human, gallery_detector_outputs_nsfw, gallery_detector_outputs_plain, gallery_rembg_outputs])
    
if config.clear_on_start is True:
    config.clear_all()

config.delete_tmp_pngs()

clear()

config.check_gpu()

multiai.queue()

multiai.launch(inbrowser=config.inbrowser, share=config.share_gradio, server_port=SERVER_PORT, server_name=SERVER_NAME)