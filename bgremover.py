import os
import shutil
from clear import clear
from PIL import Image, UnidentifiedImageError
from rembg import remove
from colors import *

color_start()
clear()

input_dir = "input"
output_dir = "output"
ask_auto_mode = ""
ask_lang = ""
auto_mode_array = ["+", "-"]
ask_lang_array = ["1", "2"]
current_directory = os.path.dirname(os.path.abspath(__file__))

print(f"{b_b}{w}Выбор языка | Language choice")
print(f"{b_b}{w}[1]Русский")
print(f"{b_b}{w}[2]English")

while ask_lang not in ask_lang_array:
    print(f"{b_b}{w}")
    ask_lang = input(f"{b_b}{g}1|2: ")
    
if ask_lang == "1":
    from local_ru import *
if ask_lang == "2":
    from local_eng import *



def rem_bg_def(): 
    pics = 0
    for filename in os.listdir(input_dir):
        try:
            pics += 1
            print(f"{b_b}{y}[{loc_log}|{loc_pic} #{pics}]{loc_st_del}")

            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{filename[:-4]}_output.png")

            if not filename.endswith(".png"):
                input_image = Image.open(input_path).convert("RGB")
                input_path = os.path.join(output_dir, f"{filename[:-4]}_input.png")
                input_image.save(input_path)
                print(f"{b_b}{y}[{loc_imp}|{loc_pic} #{pics}] {loc_jpg}")

            input_image = Image.open(input_path)
            output_image = remove(input_image)
            output_image.save(output_path)

            print(f"{b_b}{y}[{loc_log}|{loc_pic} #{pics}]{loc_del}")

        except PermissionError:
            pics -= 1
            print(f"{r_b}{w}[{loc_err}|{loc_pic} #{pics}]{loc_perm_err}")
            pass
        except FileNotFoundError:
            pics -= 1
            print(f"{r_b}{w}[{loc_err}|{loc_pic} #{pics}]{loc_nofile_err}")
            pass
        except UnidentifiedImageError:
            pics -= 1
            pass
    return pics

print(f"{b_b}{w}{loc_ask}")

while ask_auto_mode not in auto_mode_array:
    print(f"{b_b}{w}")
    ask_auto_mode = input(f"{b_b}{g}{loc_answ}")

if ask_auto_mode == "+":
    auto_mode = True
if ask_auto_mode == "-":
    auto_mode = False
    print(f"{b_b}{w}")
    your_dir = input(f"{b_b}{w}{loc_dir}")

if auto_mode == True:
    print(f"{b_b}{w}")
    print(f"{b_b}{y}[{loc_log}]{loc_st_del_1}")
    pics = rem_bg_def()
if auto_mode == False:
    input_dir = os.path.abspath(your_dir)
    print(f"{b_b}{w}")
    print(f"{b_b}{y}[{loc_log}]{loc_st_del_2}{your_dir}...")
    pics = rem_bg_def()

if pics == 0: 
    print(f"{b_b}{w}")
    print(f"{r_b}{w}[{loc_err}]{loc_need_pic}")
if pics > 0:
    print(f"{b_b}{w}")
    print(f"{b_b}{g}{loc_count_del}{pics}.")
if pics < 0:
    print(f"{b_b}{w}")
    print(f"{r_b}{w}[{loc_err}]{loc_contact}")

print(f"{b_b}{w}")
input(f"{b_b}{w}{loc_exit}")

pycache_directory = os.path.join(current_directory, '__pycache__') # Удаление кеша
shutil.rmtree(pycache_directory)