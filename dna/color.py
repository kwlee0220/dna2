BLACK = (0,0,0)
WHITE = (255,255,255)
YELLOW = (0,255,255)
RED = (0,0,255)
PURPLE = (128,0,128)
MAGENTA = (255,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
LIGHT_GREY = (211, 211, 211)

import sys
def name_to_color(name: str):
    current_module = sys.modules[__name__]
    return current_module.__dict__[name]