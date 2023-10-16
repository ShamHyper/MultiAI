import librosa as lba
import numpy as np
import shutil as sh
import os
from clear import clear
from colors import *
import soundfile as sf

clear()
color_start()

input_dir = "input"
output_dir = "output"
count = 0
current_directory = os.path.dirname(os.path.abspath(__file__))

for filename in os.listdir(input_dir):
    if filename.endswith(".mp3") or filename.endswith(".wav"):
        count += 1
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + "_boosted.wav"
        output_path = os.path.join(output_dir, output_filename)

        input_audio, sr = sf.read(input_path)
        
        normed_audio = lba.util.normalize(input_audio)
        
        boost_factor = 1
        boosted_audio = normed_audio * boost_factor

        sf.write(output_path, boosted_audio, sr)
    else:
        print(f"{filename} is not an MP3 or WAV file.")

if count == 0:
    print("No such files in directory. Bye.")
else:
    print("All ready!")
    
input("Press Enter to close program...")

pycache_directory = os.path.join(current_directory, '__pycache__')
if os.path.exists(pycache_directory):
    sh.rmtree(pycache_directory)