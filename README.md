[![wakatime](https://wakatime.com/badge/github/ShamHyper/MultiAI.svg)](https://wakatime.com/badge/github/ShamHyper/MultiAI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ShamHyper/MultiAI/blob/main/LICENSE.txt)
## Donate
**https://www.donationalerts.com/r/shamhyper0**

**USDT(TRC20): TAaBpuhdoPGNv9xnyne2nXQ5gLQLNUrY6V**
## AIs
**1. [BgRemoverLite](https://github.com/ShamHyper/BgRemoverLite)**

**2. Image upscaler**

**3. NSFW Detector**

**4. Image Analyzer**

**5. Video Analyzer**

**6. Prompt Generator**

**7. AI Detector**
## System requirements
**1. Windows 10-11**

**2. NVIDIA GPU with CUDA 12.1 support**

**3. 16gb RAM**

**4. 8gb+ free SSD (NOT HDD) space**
## Installation for Windows (recommended)
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
**5. Install [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows)**

**6. Install [cuDNN v8.9.7 (December 5th, 2023), for CUDA 12.x | Local Installer for Windows (Zip)](https://developer.nvidia.com/rdp/cudnn-archive)**

*You need to unpack archive, go to folder cudnn-windows-x86_64-8.9.7.29_cuda12-archive and copy all (bin, inlcude, lib, LICENSE) to "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1" or in your CUDA 12.1 folder*

**7. Restart your PC**

**8. Clone repository**
```git
git clone https://github.com/ShamHyper/MultiAI.git
```
**8. Run *install.bat* for first time. Next time run *run.bat***
## Installation for Windows (legacy)
**1. Install [Python 3.10.x](https://www.python.org/downloads/)**

**2. Install [Git](https://git-scm.com/downloads)**

**3. Install [FFmpeg](https://ffmpeg.org/download.html)**

**4. Install [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows)**

**5. Install [cuDNN v8.9.7 (December 5th, 2023), for CUDA 12.x | Local Installer for Windows (Zip)](https://developer.nvidia.com/rdp/cudnn-archive)**

*You need to unpack archive, go to folder cudnn-windows-x86_64-8.9.7.29_cuda12-archive and copy all (bin, inlcude, lib, LICENSE) to "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1" or in your CUDA 12.1 folder*

**6. Restart your PC**

**7. Download using "Code --> Download ZIP"** ([or click here](https://github.com/ShamHyper/MultiAI/archive/refs/heads/main.zip))

**8. Unarchive ZIP**

**8. Run *install.bat* for first time. Next time run *run.bat***
## Settings tab (recommended)
**Now you don't have to go into config.json and change something with your hands. This tab will do everything for you!**
![cfg](https://i.imgur.com/Pw00z0a.png?raw=true)
1. Choose necessary checkboxes
2. Press *Save settings*
3. Restart **MultiAI**
## Settings (legacy, ../settings/config.json)
```js
    {
        "debug_mode": "True", 
        // Boolean ("True" or "False")
        // Enable debug mode (write debug info)
        // Enable logging
        
        "start_in_browser": "True",
        // Boolean ("True" or "False")
        // Enable MultiAI starting in browser

        "share_gradio": "False",
        // Boolean ("True" or "False")
        // Enable MultiAI starting with share link

        "clear_on_start": "False"
        // Boolean ("True" or "False")
        // Clear all outputs on start
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
**5. Video Analyzer**
![6](https://i.imgur.com/cssEq5K.png?raw=true)
![6.1](https://i.imgur.com/z8aOPXj.png?raw=true)
**6. Prompt Generator**
![7](https://i.imgur.com/hRVhMKa.png?raw=true)
**7. AI Detector**
![8](https://i.imgur.com/qYRg0AS.png?raw=true)
## About Python 3.11 - 3.12
*So far, most of the libraries used in my project have not been updated to 3.11 - 3.12 or are working incorrectly. I tried to adapt the project for the new version, but for now I advise you to stay within 3.10. **After v1.10.x, discontinued support for 3.9, only 3.10 now***
## Credits
- Built with **Gradio** - https://www.gradio.app/
- Using **u2net model** for *BgRemoverLite* - https://github.com/xuebinqin/U-2-Net
- **RemBG** library for *BgRemoverLite* - https://github.com/danielgatis/rembg
- **NSFW Detection** - https://github.com/GantMan/nsfw_model
- **NSFW Detection 2.0** - https://huggingface.co/Falconsai/nsfw_image_detection 
- **Upscaling** - https://github.com/kmewhort/upscalers
- **Clip-interrogator** for *Image Analyzer* - https://github.com/pharmapsychotic/clip-interrogator
- **Fast Prompt Generator** - https://huggingface.co/FredZhang7/anime-anything-promptgen-v2
- **H5 Keras Model** - https://github.com/ShamHyper/Classificator_HUMAN_VS_AI
## License
MIT license | Copyright (c) 2023-2024 ShamHyper
