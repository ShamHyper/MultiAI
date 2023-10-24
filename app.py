from multiai import *

with gr.Blocks(
    title=init.ver, theme=gr.themes.Soft(primary_hue="red", secondary_hue="orange")) as multiai:
    ic()
    ic("Staring gradio...")
    gr.Markdown(init.ver)
    with gr.Tab("BgRemoverLite"):
        with gr.Row():
            gr.Label("Remove background from single image")
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image(width=200, height=240)
        with gr.Row():
            rembg_button = gr.Button("Remove one background")
        with gr.Row():
            gr.Label("Remove background from all images in dir")
        with gr.Row():
            image_input_dir = gr.Textbox(
                label="Enter dir", placeholder="Enter dir like this: D:\Python\MultiAI")
            image_output_dir = gr.Textbox(
                label="Output", placeholder="BrRemoverLite outputs will be here")
        with gr.Row():
            rembg_batch_button = gr.Button("Remove some backgrounds")
        with gr.Row():
            gr.Label("Clear outputs")
        with gr.Row():
            clearp_bgr_button = gr.Button("Clear outputs")
            clearp_bgr = gr.Textbox(label="Clearing progress")
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
            upsc_button = gr.Button("Start upscaling")
        with gr.Row():
            upsc_clear_cache = gr.Button("Clear torch, cuda and models cache")
    with gr.Tab("NSFW Detector"):
        with gr.Row():
            gr.Label("Detect NSFW images from dir")
        with gr.Row():
            detector_input = gr.Textbox(
                label="Enter dir", placeholder="Enter dir like this: D:\Python\MultiAI")
            detector_slider = gr.Slider(
                value=0.36,
                label="Threshold (larger number = simpler detection | smaller number = stricter one)",
                minimum=0.0001,
                maximum=0.9999,
            )
        with gr.Row():
            detector_button = gr.Button("Click here to start")
            detector_output = gr.Textbox(
                label="Output", placeholder="NSFW Detector outputs will be here")
        with gr.Row():
            gr.Label("Clear outputs")
        with gr.Row():
            detector_clear_button = gr.Button("Clear outputs")
            clearp = gr.Textbox(label="Clearing progress")
            

    rembg_button.click(multi.rem_bg_def, inputs=image_input, outputs=image_output)
    rembg_batch_button.click(multi.rem_bg_def_batch, inputs=image_input_dir, outputs=image_output_dir)
    clearp_bgr_button.click(multi.clearp_bgr_def, outputs=clearp_bgr)

    detector_button.click(multi.detector, inputs=[detector_input, detector_slider], outputs=detector_output)
    detector_clear_button.click(multi.detector_clear, outputs=clearp)
    
    upsc_button.click(multi.uspc, inputs=[upsc_image_input, scale_factor, model_ups], outputs=upsc_image_output)
    upsc_clear_cache.click(clear_on_device_caches)
    
  
if init.debug is True:
    ic(multiai.queue())
    ic(multiai.launch(inbrowser=init.inbrowser))
elif init.debug is False:
    multiai.queue()
    multiai.launch(inbrowser=init.inbrowser)