from PIL import Image, ImageDraw
import random

from cgan.dataloading import EmojiFaceSegments


def query_cgan(model_path):
    # Load model

    # Query model for each part

    # Create background

    # Paste parts on background

    # Save pic
    pass


def combine_nogan(parts_list):
    # Create background
    background = create_background()

    # Load dataset
    dataset_path = "../segmentation/segmentation_data"
    dataset = EmojiFaceSegments(dataset_path)

    # Choose parts
    for apart in parts_list:
        curr_parts = list(map(lambda x: x[0], filter(lambda x: x[1]==apart, dataset.data)))
        # Pick random part
        pick = Image.fromarray(random.choice(curr_parts))

        pick.save("testing.jpg", "JPEG")

        pick = pick.convert("RGBA")

        pixdata = pick.load()

        width, height = pick.size
        for y in range(height):
            for x in range(width):
                if pixdata[x, y] == (0, 0, 0, 255):
                    pixdata[x, y] = (0, 0, 0, 0)

        pick.save("testing.png", "PNG")
        pick.load()

        # Paste parts and save pic
        try:
            background.paste(pick, mask=pick.split()[3])  # 3 is the alpha channel
        except IndexError:  # No alpha channel found
            background.paste(pick)
    background.save("test_paste.jpg", quality=100)
    print("Saved.")


def create_background():
    background = Image.new("RGB", (64, 64), (0,0,0))
    img = ImageDraw.Draw(background)
    img.ellipse((6, 6, 58, 58), fill=(254,215,50), outline=(220,134,11), width=1)
    background.save("test.jpg", 'JPEG', quality=100)
    return background


# labels = {"ears": 1, "eyebrows": 2, "eyes": 3, "hands": 4, "mouth": 5, "tears": 6}
combine_nogan([2, 3, 5])