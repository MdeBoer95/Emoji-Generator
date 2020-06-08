from PIL import Image
from os import listdir
from os.path import isfile, join

path = "emojis_root/all_emojis"
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

for p in onlyfiles:
    png = Image.open(path+"/"+p)
    png.load() # required for png.split()

    background = Image.new("RGB", png.size, (0,0,0))
    try:
        background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
    except IndexError:  # No alpha channel found
        background.paste(png)
    finally:
        background.save("emojis_jpg_root/all_emojis/" + p.split(".")[0] + ".jpg", 'JPEG', quality=100)
