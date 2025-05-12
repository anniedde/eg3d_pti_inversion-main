from PIL import Image, ImageEnhance

images = [0, 1, '2-left', 4, 5, '6-left']
for im in images:
    img = Image.open(f"{im}.png")
    enhancer = ImageEnhance.Brightness(img)
    # to reduce brightness by 50%, use factor 0.5
    img = enhancer.enhance(1.5)

    img.show()
    img.save(f"{im}_brighter.png")