![0](https://i.imgur.com/Ui4dvUh.png?raw=true)

[![1](https://i.imgur.com/8Hva6OY.png)](https://www.donationalerts.com/r/shamhyper0)

[![wakatime](https://wakatime.com/badge/github/ShamHyper/MultiAI.svg)](https://wakatime.com/badge/github/ShamHyper/MultiAI)
## AIs
**1. [BgRemoverLite](https://github.com/ShamHyper/BgRemoverLite)**

**2. Image upscaler**

**3. NSFW Detector**

**4. Image Analyzer**

**5. Prompt Generator**
## System requirements
**1. Windows 10-11**

**2. NVIDIA GPU with CUDA 11.8 support**

**3. 16gb RAM**
## Installation (legacy)
**1. Install [Python 3.10.x](https://www.python.org/downloads/)**

**2. Install [Git](https://git-scm.com/downloads)**

**3. Install [FFmpeg](https://ffmpeg.org/download.html)**

**4. Install [CUDA 11.8](https://developer.nvidia.com/cuda-toolkit)**

**5. Restart your PC**

**6. Download using "Code --> Download ZIP"** ([or click here](https://github.com/ShamHyper/MultiAI/archive/refs/heads/main.zip))

**7. Unarchive ZIP**

**8. Run *install.bat* for first time. Next time run *run.bat***
## Installation (recommended)
**1. Install [winget](https://learn.microsoft.com/windows/package-manager/winget/#install-winget)**

**2. Install Python**
```py
winget install -e --id Python.Python.3.10
```
**3. Install Git**
```py
winget install -e --id Git.Git
```
**4. Install FFmpeg**
```py
winget install -e --id Gyan.FFmpeg
```
**5. Install [CUDA 11.8](https://developer.nvidia.com/cuda-toolkit)**

**6. Restart your PC**

**7. Clone repository**
```git
git clone https://github.com/ShamHyper/MultiAI.git
```
**8. Run *install.bat* for first time. Next time run *run.bat***
## Configuration (config.json)
```js
    {
        "debug_mode": "True", 
        // Boolean ("True" or "False")
        // Enable debug mode (write debug info)
        
        "start_in_browser": "True",
        // Boolean ("True" or "False")
        // Enable MultiAI starting in browser

        "share_gradio": "False",
        // Boolean ("True" or "False")
        // Enable MultiAI starting with share link

        "clear_need": "True",
        // Boolean ("True" or "False")
        // Enable cache clear function on MultiAI start

        "preload_models" : "False"
        // Boolean ("True" or "False")
        // Load models on start of run.bat (True) or load models on running Image Analyzer (False)
    }
```
## Support
1. If you see errors like this: ```ModuleNotFoundError: No module named "certifi"```, run install.bat for fix them. If that didn't help, open a new issue
2. If you encounter errors during the execution of the program, open a new issue
3. If your console window freezes during the install process.bat, restart it with administrator rights. It is also highly recommended to run run.bat from the administrator.
## Gallery
**1. [BgRemoverLite](https://github.com/ShamHyper/BgRemoverLite)**
![2](https://i.imgur.com/mIkIOMB.png?raw=true)
**2. Image upscaler**
![3](https://i.imgur.com/4OQmALL.png?raw=true)
**3. NSFW Detector**
![4](https://i.imgur.com/zveO3a7.png?raw=true)
**4. Image Analyzer**
![5](https://i.imgur.com/wR1fGIn.png?raw=true)
**5. Prompt Generator**
![6](https://i.imgur.com/hRVhMKa.png?raw=true)
## About Python 3.11 - 3.12
*So far, most of the libraries used in my project have not been updated to 3.11 - 3.12 or are working incorrectly. I tried to adapt the project for the new version, but for now I advise you to stay within 3.10 (in extreme cases, 3.9 will also work)*
## Credits
*Built with Gradio - https://www.gradio.app/*

*u2net model - https://github.com/xuebinqin/U-2-Net*

*RemBG library - https://github.com/danielgatis/rembg*

*NSFW Detection - https://github.com/GantMan/nsfw_model*

*Upscaling - https://github.com/kmewhort/upscalers*

*Clip-interrogator - https://github.com/pharmapsychotic/clip-interrogator*

*Fast Anime PromptGen - https://huggingface.co/FredZhang7/anime-anything-promptgen-v2*