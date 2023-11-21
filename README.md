## MultiAI
[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)](./LICENSE.md)

![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![PowerShell](https://img.shields.io/badge/PowerShell-%235391FE.svg?style=for-the-badge&logo=powershell&logoColor=white)

![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
## Donate
[![1](https://i.imgur.com/8Hva6OY.png)](https://www.donationalerts.com/r/shamhyper0)

**ETH: 0x634907686f395570eF91cd0a4e694C3Df810D9B0**
## Time spent on the project
[![wakatime](https://wakatime.com/badge/github/ShamHyper/MultiAI.svg)](https://wakatime.com/badge/github/ShamHyper/MultiAI)
## Our discord servers
![Discord Banner SH](https://discordapp.com/api/guilds/1091587110542266501/widget.png?style=banner2)
![Discord Banner VDM](https://discordapp.com/api/guilds/1158378744982020169/widget.png?style=banner2)
## AIs
**1. [BgRemoverLite](https://github.com/ShamHyper/BgRemoverLite)**

**2. Image upscaler**

**3. NSFW Detector**

**4. Image Analyzer**

**5. Video Analyzer**

**6. Prompt Generator**
## System requirements
**1. Windows 10-11**

**2. NVIDIA GPU with CUDA 11.8 support**

**3. 16gb RAM**
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
**5. Install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows)**

**6. Restart your PC**

**7. Clone repository**
```git
git clone https://github.com/ShamHyper/MultiAI.git
```
**8. Run *install.bat* for first time. Next time run *run.bat***
## Installation for Windows (legacy)
**1. Install [Python 3.10.x](https://www.python.org/downloads/)**

**2. Install [Git](https://git-scm.com/downloads)**

**3. Install [FFmpeg](https://ffmpeg.org/download.html)**

**4. Install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows)**

**5. Restart your PC**

**6. Download using "Code --> Download ZIP"** ([or click here](https://github.com/ShamHyper/MultiAI/archive/refs/heads/main.zip))

**7. Unarchive ZIP**

**8. Run *install.bat* for first time. Next time run *run.bat***
## [BETA]Installation for Linux (tested on Ubuntu)
**1. Install Python**
```sh
sudo apt install python3
```
**2. Install PIP**
```sh
sudo apt install python3-pip
```
**3. Install Git**
```py
sudo apt install git
```
**4. Install FFmpeg**
```py
sudo apt install FFmpeg
```
**5. Install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux)**

**6. Restart your PC**

**7. Clone repository**
```git
git clone https://github.com/ShamHyper/MultiAI.git
```
**8. In terminal, type for first time:**
```sh
pip3 install -r requirements.txt
```
**9. In terminal, type for run MultiAI:**
```sh
python3 app.py
```

*This is a very BETA version of the installation for Linux. I have briefly checked the functionality and do not really understand how to set up a venv for this OS yet.*

| BgRemoverLite | Image upscaler | NSFW Detector | Image Analyzer  | Video Analyzer | Prompt Generator |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Fully functional  | Partially functional | Not tested | Not tested | Not tested | Not tested |
## Settings tab (recommended)
### Now you don't have to go into config.json and change something with your hands. This tab will do everything for you!
![cfg](https://i.imgur.com/DzCcfnu.png?raw=true)
1. Choose necessary checkboxes
2. Pick config.json
3. Press *Save settings*
4. Restart **MultiAI**
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

        "preload_models": "False",
        // Boolean ("True" or "False")
        // Preloading AI models
        // False loading time: ~5-10s
        // True loading time: ~40-60s

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
## About Python 3.11 - 3.12
*So far, most of the libraries used in my project have not been updated to 3.11 - 3.12 or are working incorrectly. I tried to adapt the project for the new version, but for now I advise you to stay within 3.10. **After v1.10.x, discontinued support for 3.9, only 3.10 now***
## Credits
- Built with **Gradio** - https://www.gradio.app/
- Using **u2net model** for *BgRemoverLite* - https://github.com/xuebinqin/U-2-Net
- **RemBG** library for *BgRemoverLite* - https://github.com/danielgatis/rembg
- **NSFW Detection** - https://github.com/GantMan/nsfw_model
- **Upscaling** - https://github.com/kmewhort/upscalers
- **Clip-interrogator** for *Image Analyzer* - https://github.com/pharmapsychotic/clip-interrogator
- **Fast Prompt Generator** - https://huggingface.co/FredZhang7/anime-anything-promptgen-v2
## License
MIT license | Copyright (c) 2023 ShamHyper