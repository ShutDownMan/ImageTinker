import PySimpleGUI as sg
from cv2 import cv2
import numpy as np
from PIL import Image, ImageTk
import io


def main():
    def abrirImg():
        file = sg.popup_get_file('Image to open', save_as=False, file_types = ((".png", ".jpg", "jpeg", ".tiff", ".bmp"),))

        if not file:
            sg.popup_cancel('Cancelling')
            raise SystemExit()
        return file

    def get_img_data(f, maxsize=(400, 300), first=False):
        img = Image.open(f)
        img.thumbnail(maxsize)
        if first:                     # tkinter is inactive the first time
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            del img
            return bio.getvalue()
        return ImageTk.PhotoImage(img)

    filename = abrirImg()
    image_elem_Input = sg.Image(data=get_img_data(filename, first=True), key='-In-')
    image_elem_Output = sg.Image(data=get_img_data(filename, first=True), key='-Out-')
    filename_display_elem = sg.Text(filename, size=(80, 3))

    col = [[filename_display_elem], [image_elem_Input], [image_elem_Output]]

    col_filter = [[sg.Radio('None', 'Radio', True, size=(10, 1))],
        [sg.Radio('threshold', 'Radio', size=(10, 1), key='-THRESH-'),
        sg.Slider((0, 255), 128, 1, orientation='h', size=(40, 15), key='-THRESH SLIDER-')],
        [sg.Radio('GrayScale', 'Radio', size=(10,1), key='-GRAYSCALE-')],
        [sg.Radio('canny', 'Radio', size=(10, 1), key='-CANNY-'),
        sg.Slider((0, 255), 128, 1, orientation='h', size=(20, 15), key='-CANNY SLIDER A-'),
        sg.Slider((0, 255), 128, 1, orientation='h', size=(20, 15), key='-CANNY SLIDER B-')],
        [sg.Radio('blur', 'Radio', size=(10, 1), key='-BLUR-'),
        sg.Slider((1, 11), 1, 1, orientation='h', size=(40, 15), key='-BLUR SLIDER-')],
        [sg.Radio('hue', 'Radio', size=(10, 1), key='-HUE-'),
        sg.Slider((0, 225), 0, 1, orientation='h', size=(40, 15), key='-HUE SLIDER-')],
        [sg.Radio('enhance', 'Radio', size=(10, 1), key='-ENHANCE-'),
        sg.Slider((1, 255), 128, 1, orientation='h', size=(40, 15), key='-ENHANCE SLIDER-')]
        ]

    layout = [[sg.Column(col_filter), sg.Column(col)]]

    window = sg.Window('Filtros', layout, return_keyboard_events=True,
                    location=(0, 0), use_default_focus=False)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

        if values['-THRESH-']:
            img = cv2.cvtColor(image_elem_Input, cv2.COLOR_BGR2LAB)[:, :, 0]
            img = cv2.threshold(img, values['-THRESH SLIDER-'], 255, cv2.THRESH_BINARY)[1]
        
        if values['-GRAYSCALE-']:
            print(image_elem_Input)
            #img = cv2.cvtColor(image_elem_Input, cv2.COLOR_BGR2LAB)[:, :, 0]

        print(values['-GRAYSCALE-'])
                    
        # image_elem_Input.update(img)
        image_elem_Onput.update(img)

        # D:\Documents\GitHub\ImageTinker\lena_std.tif

    window.close()

main()




# D:\Documents\GitHub\ImageTinker\lena_std.tif
