from colorama import Fore
from colorama import Back
from colorama import init
from colorama import just_fix_windows_console

def color_start():
    just_fix_windows_console()
    init()
    print(f"{b_b}Colors initialized!")
    
r = Fore.RED
g = Fore.GREEN
y = Fore.YELLOW
w = Fore.WHITE
r_b = Back.RED
g_b = Back.GREEN
y_b = Back.YELLOW
b_b = Back.RESET