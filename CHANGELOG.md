# Changelog
## 1.14.x
- 
## 1.13.x
- Improved code readability
- Changes recommended by Ruff have been made
- The log system has been rewritten
- Added tooltips instead of some logs
- Fixed a bug that prevented the launch of the Video analyzer
- Now None is selected in Upscaler according to the standard, which removed the error at startup
- Release of AI Detector
- Full new h5 model for detector, trained by me - https://github.com/ShamHyper/Classificator_HUMAN_VS_AI
- QOL fixes
## 1.12.x
- Added [BETA]AI Detector - detecting AI or Human created images. Very crude function, a lot of false definitions
- Some fixes of AI Detector
- Added threshold for AI Detector
- New H5 model
- Now all models downloading from our studio server
- QOL updates
## 1.11.x
- QOL updates
- Libs versions update in requirements
- Removed logs
## 1.10.x
- Fixed live settings for #3 issue
## 1.9.x
- Fixed all bugs of version **1.8**
- Updating for requirements.txt old-version libs
- Fixed requirements.txt (gfpgan requirements compability) 
- Fixed bug with tmp.png in MultiAI/.. folder
- Gradio update
- Fixed single video **Video Analyzer** (idk why and how I broke this function...)
- New color theme - with .css file for custom theme in ../settings
- Clear log in clearing tab
## 1.8.x
- Fixed all bugs of version **1.7**
- All defs renamed for simpler use
- Progress textbox for **Video Analyzer** cleaner
- Added Settings tab. Now you don't have to go into config.json and change something with your hands. This tab will do everything for you!
- Logging for *Debug mode* (../logs)
## 1.7.x
- Fixed all bugs of version **1.6**
- Work on optimization has been carried out
- Now all outputs are in the same folders for ease of use
- Fixed a bug with an uncleaned *tmp_pngs* folder that appeared if the batch process of **Video Analyzer** was interrupted
- Added separate class for models
- Added *Clear all outputs* tab
- Code optimizations for config
- New lines for debug (config.json)
- Threshold slider for *Skip drawings or anime* checkbox
## 1.6.x
- New AI - **Video Analyzer**. Analyze Video like in **Image Analyzer**, but analyze per-frame of video and give you avg. percentages specs. of them
- Fixes for debug launch-timer
- Removed IceCream logging for GitHub. Now this only dev-version function
- Better model loading
- Better console outputs
- "preload_models" option in config.json
- Added batch for **Video Analyzer**
- Lazy mode batch **Video Analyzer** (skiping per x frames)
- Threshold slider for **Video Analyzer**
## 1.5.x
- *New* system of loading models (35s preloading, but faster working)
- Code optimization
- Fixed lying debug timer
- Debug-timer optimizations
## 1.4.x
- Deleted preload models because this function is corrupted and unuseful
- Added checkbox in **Prompt Generator** - randomize temperature of prompt. Temperature controls the level of "creativity" of the AI on a scale from 0 to 1, the default value is 0.7. The lower, the more impersonal the text will be. The higher — the more unusual the text will be. I set the value from 0.4 to 0.9 when enabling the "randomizer". This will help to avoid duplicate prompts when generating with the same text.
- Removed clearing function
- Removed clear_need in config.json
- Code optimizations (20% improve)
- Loading timer for debug mode
- Added emojis for buttons to better understand the interface
- News about Python 3.11+, details in the README
- Fixed console warnings that slowed down the promt generator
- Added a new AI - **prompt generator**. With it, you can generate prompt, complementing your own (examples in the screenshots in the README)
## 1.3.x
- Optimization of the detector functions has been carried out
- Added the ability to choose whether to load AI models during program startup or during the execution of a specific AI. For more information, see config.json
## 1.2.x
- Optimization of the image analysis function
- Fixed several inconspicuous bugs
- Added the ability to enable asynchronous model loading. Later it was cut out in 1.3+
## 1.1.x
- Improved the output of prompts in Image Analyzer
- Removed spam in the console associated with the previously added debugger for gradio
- NSFW specifier renamed to Image Analyzer
- Added CLIP interrogator to Image Analyzer to generate an image promo
## 1.0.x
- Added NSFW specifier for evaluating images as a percentage (porn, hentai, drawings, netural, etc.)
- Release. Beta changelog will be introduced later