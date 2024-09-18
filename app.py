import time
from multiai import init, config, multi
import gradio as gr
from clear import clear
from upscalers import available_models, clear_on_device_caches

start_time = time.time()

def CODC_log():
    clear_on_device_caches
    gr.Info("Cache cleared!")

with open("settings/.css", "r") as file:
    CSS = file.read()

with gr.Blocks(css=CSS, title=init.ver, theme=gr.themes.Soft(
    primary_hue="purple", 
    secondary_hue="blue")) as multiai:
    gr.Markdown(init.ver)
    with gr.Tab("BgRemoverLite"):
        with gr.Row():
            gr.Label("Remove background from single image")
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image(width=200, height=240)
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
    with gr.Tab("Upscaler"):
        with gr.Row():
            gr.Label("Upscale image up to 10x size")
        with gr.Row():
            upsc_image_input = gr.Image()
            upsc_image_output = gr.Image(width=200, height=240)
        with gr.Row():
            model_ups = gr.Dropdown(label="Model", choices=available_models(), value="None")
            scale_factor = gr.Slider(
                value=4.0,
                label="Scale factor (4x factor max recommended)",
                minimum=1.1,
                maximum=10.0,
            )
            upsc_button = gr.Button("📈 Start upscaling")
            
##################################################################################################################################
            
    with gr.Tab("NSFW Detector"):
        with gr.Row():
            gr.Label("Detect NSFW images from dir")
        with gr.Row():
            detector_input = gr.Textbox(label="Enter dir", placeholder="Enter dir like this: D:\Python\MultiAI")
            detector_slider = gr.Slider(
                value=0.36,
                label="Threshold (larger number = simpler detection | smaller number = stricter one)",
                minimum=0.0001,
                maximum=0.9999,
            )
        with gr.Row():
            detector_button = gr.Button("👟 Click here to start")
            detector_output = gr.Textbox(label="Output", placeholder="NSFW Detector outputs will be here")
        with gr.Row():
            gr.Label("Clear outputs")
        with gr.Row():
            detector_clear_button = gr.Button("🧹 Clear outputs")
            clearp = gr.Textbox(label="Clearing progress")
    with gr.Tab("Image Analyzer"):
        with gr.Row():
            gr.Label("Analyze images")
        with gr.Row():
            file_spc = gr.Image()
            spc_output = gr.Textbox(label="Stats", placeholder="Press start to get specifications of image")
        with gr.Row():
            clip_checked = gr.Checkbox(value=False, label="Use CLIP for generate prompt (slow if a weak PC)")
            spc_button = gr.Button("👟 Click here to start")
    with gr.Tab("Video Analyzer"):
        with gr.Row():
            gr.Label("Analyze images")
        with gr.Row():
            file_Vspc = gr.Video()
            Vspc_output = gr.Textbox(label="Stats", placeholder="Press start to get specifications of Video")
        with gr.Row():
            Vspc_button = gr.Button("👟 Click here to start")        
    with gr.Tab("Prompt Generator"):
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
                maximum=1000,
            )
            pg_max_length = gr.Slider(
                value=76,
                label="Max Length of prompt",
                minimum=1,
                maximum=1000,
            )
        with gr.Row():
            promptgen_button = gr.Button("⭐ Start")
            
##################################################################################################################################            
            
    with gr.Tab("[BETA-0.2]AI Detector"):
        with gr.Row():
            gr.Label("Detector threshold")
            aid_slider = gr.Slider(
                value=1.000,
                label="Threshold (it is better not to change, it is left for tests)",
                minimum=0.001,
                maximum=1.001
            ) 
        with gr.Row():
            gr.Label("Analyze image")
        with gr.Row():
            aid_input_single = gr.Image(width=200, height=400)
            aid_output_single = gr.Textbox(label="Result", placeholder="Press start to get result")
        with gr.Row():
            aid_single_button = gr.Button("👟 Click here to start")

    rembg_button.click(multi.rem_bg_def, inputs=image_input, outputs=image_output)
    rembg_batch_button.click(multi.rem_bg_def_batch, inputs=image_input_dir, outputs=image_output_dir)
    clearp_bgr_button.click(multi.clearp_bgr_def, outputs=clearp_bgr)

        with gr.Row():
            gr.Label("Detect AI/Human images from dir")
        with gr.Row():
            aid_input_batch = gr.Textbox(label="Enter dir", placeholder="Enter dir like this: D:\Python\MultiAI")
            aid_output_batch = gr.Textbox(label="Output", placeholder="AI Detector outputs will be here")
        with gr.Row():
            aid_batch_button = gr.Button("👟 Click here to start")
            
####                           ####                           ####                           ####                           ####              
        with gr.Row():
            gr.Label("Clear outputs")
        with gr.Row():
            aid_clear_button = gr.Button("🧹 Clear outputs")
            aid_clearp = gr.Textbox(label="Clearing progress")
                  
##################################################################################################################################
            
    with gr.Tab("Clearing"):
        with gr.Row():
            gr.Label("Clear all outputs")
            clear_all_button = gr.Button("⭐ Start")
            clear_all_tb = gr.Textbox(label="Result")
        with gr.Row():
            upsc_clear_cache = gr.Button("🧹 Clear torch, cuda and models cache")
            
##################################################################################################################################

    with gr.Tab("Settings"):
        with gr.Row():
            settings_debug_mode = gr.Checkbox(value=config.debug, label="Enable debug mode (write debug info)")
        with gr.Row():
            settings_start_in_browser = gr.Checkbox(value=config.inbrowser, label="Enable MultiAI starting in browser")
        with gr.Row():
            settings_share_gradio = gr.Checkbox(value=config.share_gradio, label="Enable MultiAI starting with share link")
        with gr.Row():
            settings_preload_models = gr.Checkbox(value=config.preload_models, label="Enable preloading AI models")
        with gr.Row():
            settings_clear_on_start = gr.Checkbox(value=config.clear_on_start, label="Enable clear all outputs on MultiAI start")
        with gr.Row():
            json_files = gr.Label("Saving in [../settings/config.json]")
            settings_save = gr.Button("🗃️ Save settings")
            settings_save_progress = gr.Textbox(label="Saving progress", placeholder="Your saving progress will be here")
        with gr.Row():
            gr.Label("Creating new ones .json files in ../settings will not give any effect.")
            
##################################################################################################################################
                     
    rembg_button.click(multi.BgRemoverLite, inputs=image_input, outputs=image_output)
    rembg_batch_button.click(multi.BgRemoverLiteBatch, inputs=image_input_dir, outputs=image_output_dir)
    clearp_bgr_button.click(multi.BgRemoverLite_Clear, outputs=clearp_bgr)

    detector_button.click(multi.NSFW_Detector, inputs=[detector_input, detector_slider, detector_skeep_dr, drawings_threshold], outputs=detector_output)
    detector_clear_button.click(multi.NSFWDetector_Clear, outputs=clearp)
    
    upsc_button.click(multi.Upscaler, inputs=[upsc_image_input, scale_factor, model_ups], outputs=upsc_image_output)
    upsc_clear_cache.click(CODC_log)
    
    spc_button.click(multi.spc, inputs=[file_spc, clip_checked], outputs=spc_output)
    
    Vspc_button.click(multi.Vspc, inputs=file_Vspc, outputs=Vspc_output)
    
    promptgen_button.click(multi.prompt_generator, inputs=[prompt_input, pg_prompts, pg_max_length, randomize_temp], outputs=promptgen_output)

<<<<<<< Updated upstream
if init.debug is True:
    if init.preload_models is True:
        ci_load()
        nsfw_load()
        tokenizer_load()  
    end_time = time.time()
    total_time = round(end_time - start_time)
    clear()
    print(f"Executing init time: {total_time}s")
    multiai.queue()
    multiai.launch(inbrowser=init.inbrowser, share=init.share_gradio)
elif init.debug is False:
    if init.preload_models is True:
        ci_load()
        nsfw_load()
        tokenizer_load()  
    clear()
    multiai.queue()
    multiai.launch(inbrowser=init.inbrowser, share=init.share_gradio)
=======
clear()

if config.clear_on_start is True:
    init.clear_all()

if config.preload_models is True:
    init.preloader()
    
init.delete_tmp_pngs()
   
multiai.queue()

clear()

end_time = time.time()
total_time = round(end_time - start_time, 2)
print(f"Executing time: {total_time}s")

multiai.launch(inbrowser=config.inbrowser, share=config.share_gradio)
>>>>>>> Stashed changes
