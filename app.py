import time
start_time = time.time()
from multiai import *
import gradio as gr
from clear import clear
from upscalers import available_models, clear_on_device_caches

with gr.Blocks(title=init.ver, theme=gr.themes.Soft(primary_hue="red", secondary_hue="orange")) as multiai:
    gr.Markdown(init.ver)
    
##################################################################################################################################
    
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
      
##################################################################################################################################
            
    with gr.Tab("Upscaler"):
        with gr.Row():
            gr.Label("Upscale image up to 10x size")
        with gr.Row():
            upsc_image_input = gr.Image()
            upsc_image_output = gr.Image(width=200, height=240)
        with gr.Row():
            model_ups = gr.Dropdown(label="Model", choices=available_models())
            scale_factor = gr.Slider(
                value=4.0,
                label="Scale factor (4x factor max recommended)",
                minimum=1.1,
                maximum=10.0,
            )
            upsc_button = gr.Button("📈 Start upscaling")
        with gr.Row():
            upsc_clear_cache = gr.Button("🧹 Clear torch, cuda and models cache")
            
##################################################################################################################################
            
    with gr.Tab("NSFW Detector"):
        with gr.Row():
            gr.Label("Detect NSFW images from dir")
        with gr.Row():
            detector_input = gr.Textbox(label="Enter dir", placeholder="Enter dir like this: D:\Python\MultiAI")
            detector_output = gr.Textbox(label="Output", placeholder="NSFW Detector outputs will be here")
        with gr.Row():
            detector_slider = gr.Slider(
                value=0.36,
                label="Threshold (larger number = simpler detection | smaller number = stricter one)",
                minimum=0.01,
                maximum=0.98
            )
        with gr.Row():
            detector_skeep_dr = gr.Checkbox(value=False, label="🎨 Skip drawings or anime")
            drawings_threshold = gr.Slider(
                value=0.10,
                label="Threshold for 🎨 Skip drawings or anime",
                minimum=0.01,
                maximum=0.98
            )
        with gr.Row():
            detector_button = gr.Button("👟 Click here to start")
        with gr.Row():
            gr.Label("Clear outputs")
        with gr.Row():
            detector_clear_button = gr.Button("🧹 Clear outputs")
            clearp = gr.Textbox(label="Clearing progress")
            
##################################################################################################################################            
            
    with gr.Tab("Image Analyzer"):
        with gr.Row():
            gr.Label("Analyze images")
        with gr.Row():
            file_spc = gr.Image()
            spc_output = gr.Textbox(label="Stats", placeholder="Press start to get specifications of image")
        with gr.Row():
            clip_checked = gr.Checkbox(value=False, label="Use CLIP for generate prompt (slow if a weak PC)")
            spc_button = gr.Button("👟 Click here to start")
            
##################################################################################################################################
            
    with gr.Tab("Video Analyzer"):
        with gr.Row():
            gr.Label("Analyze Video")
        with gr.Row():
            file_Vspc = gr.Video()
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
            
    with gr.Tab("Clear all outputs"):
        with gr.Row():
            clear_all_button = gr.Button("⭐ Start")
            clear_all_tb = gr.Textbox(label="Result")
            
##################################################################################################################################

    with gr.Tab("Settings"):
        with gr.Row():
            settings_debug_mode = gr.Checkbox(value=True, label="Enable debug mode (write debug info)")
        with gr.Row():
            settings_start_in_browser = gr.Checkbox(value=True, label="Enable MultiAI starting in browser")
        with gr.Row():
            settings_share_gradio = gr.Checkbox(value=False, label="Enable MultiAI starting with share link")
        with gr.Row():
            settings_preload_models = gr.Checkbox(value=False, label="Enable preloading AI models")
        with gr.Row():
            settings_clear_on_start = gr.Checkbox(value=False, label="Enable clear all outputs on MultiAI start")
        with gr.Row():
            json_files = gr.Dropdown(label="Pick .json file to write settings", choices=config.list_json_files())
            settings_save = gr.Button("🗃️ Save settings")
            settings_save_progress = gr.Textbox(label="Saving progress", placeholder="Your saving progress will be here")
        with gr.Row():
            gr.Label("Creating new ones .json files in ../settings will not give any effect. Choose only config.json in dropdown above!")
            
            
    rembg_button.click(multi.BgRemoverLite, inputs=image_input, outputs=image_output)
    rembg_batch_button.click(multi.BgRemoverLiteBatch, inputs=image_input_dir, outputs=image_output_dir)
    clearp_bgr_button.click(multi.BgRemoverLite_Clear, outputs=clearp_bgr)

    detector_button.click(multi.NSFW_Detector, inputs=[detector_input, detector_slider, detector_skeep_dr, drawings_threshold], outputs=detector_output)
    detector_clear_button.click(multi.NSFWDetector_Clear, outputs=clearp)
    
    upsc_button.click(multi.Upscaler, inputs=[upsc_image_input, scale_factor, model_ups], outputs=upsc_image_output)
    upsc_clear_cache.click(clear_on_device_caches)
    
    spc_button.click(multi.ImageAnalyzer, inputs=[file_spc, clip_checked], outputs=spc_output)
    
    Vspc_button.click(multi.VideoAnalyzer, inputs=file_Vspc, outputs=Vspc_output)
    start_dir_videos.click(multi.VideoAnalyzerBatch, inputs=[video_dir, vbth_slider, threshold_Vspc_slider], outputs=bth_Vspc_output)   
    clear_videos.click(multi.VideoAnalyzerBatch_Clear, outputs=bth_Vspc_clear_output)   
    
    promptgen_button.click(multi.PromptGenetator, inputs=[prompt_input, pg_prompts, pg_max_length, randomize_temp], outputs=promptgen_output)
    
    clear_all_button.click(init.clear_all, outputs=clear_all_tb)
    
    settings_save.click(config.save_config_gr, inputs=[settings_debug_mode, settings_start_in_browser, 
                                                       settings_share_gradio, settings_preload_models, 
                                                       settings_clear_on_start, json_files], outputs=settings_save_progress)

clear()

if config.clear_on_start == True:
    init.clear_all()

if config.preload_models == True:
    init.preloader()
    
init.delete_tmp_pngs()
   
multiai.queue()

end_time = time.time()
total_time = round(end_time - start_time, 2)
print(f"Executing time: {total_time}s")
print("")

multiai.launch(inbrowser=config.inbrowser, share=config.share_gradio)