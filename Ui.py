import time
import tkinter as tk
import PySimpleGUI as sg
import os.path

def display_result(emoji_path):
    # make sure to make the emoji small or it breaks this
    sg.Popup('This was the matched emoji', icon=emoji_path)

class ProgressWindow():
    def __init__(self, window):
        try:
            self.window = window
            self.window['-TOUT-'].update("Running code ...")
        except Exception as e:
            print(e)

    def update_text(self, txt):
        self.window.Refresh()
        self.window['-TOUT-'].update(txt)

    def close(self):
        self.window.close()

def process_img(filename, window):
    progress = ProgressWindow(window)

    # run code

    #display_result(filename)

def main():
    file_list_column = [
        [
            sg.Text("Choose image that is png, jpeg or jpg"),
            sg.In(size=(25, 1), enable_events=True, key="-ADDIMAGE-"),
            sg.FileBrowse(),
        ],
        [
            sg.Listbox(
                values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
            )
        ],
    ]

    # For now will only show the name of the file that was chosen
    image_viewer_column = [
        [sg.Text("Click image from list on left to preview:")],
        [sg.Text(size=(40, 1), key="-TOUT-")],
        [sg.Image(key="-IMAGE-")],
    ]

    layout = [
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
        ]
    ]

    window = sg.Window("Image Viewer", layout)

    # Run the Event Loop
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        # Folder name was filled in, make a list of files in the folder
        if event == "-ADDIMAGE-":
            image_path = values["-ADDIMAGE-"]
            img_name = image_path.split('/')[-1]

            if img_name.lower().endswith((".png", ".jpeg", ".jpg")):
                window["-FILE LIST-"].update([image_path])

        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            try:
                filename = values["-FILE LIST-"][0]
                window["-TOUT-"].update(filename)

                response = sg.popup_yes_no('Process this image?')
                if response == 'Yes':
                    process_img(filename, window)
            except:
                sg.popup('There was an issue with reading this image try another one or change type to png')

    window.close()

if __name__ == '__main__':
    main()