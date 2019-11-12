from tkinter import filedialog

import StyleTransferProcessor as stp


def select_picture(title="Select picture"):
    return filedialog.askopenfilename(title=title,
                                      filetypes=(("jpeg files","*.jpeg",),("jpg files","*.jpg")))


style_path = select_picture("Select style picture")
content_path = select_picture("Select content picture")

print("Enter style factor:")
style_factor = int(input())

image = stp.process(style_path, content_path, style_factor)
stp.imshow(image, "Stylized image")
stp.save_image(image, "stylized-image.png")
