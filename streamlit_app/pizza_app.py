from multiprocessing import context
from secrets import choice
from turtle import color, title
import streamlit as st
from tempfile import NamedTemporaryFile
import torch.nn as nn
import torch
from torchvision import io
from PIL import Image
from torchvision import transforms as T
from torchvision import io
import numpy as np
import matplotlib.pyplot as plt

# print("Sreamlit version", st.__version__)


def load_image(image_file):
    img = Image.open(image_file)
    return img


def main():
    st.title("File Upload Tutorial")
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    if image_file is not None:
    #     file_details = {"filename":image_file.name, "filetype":image_file.type,
    #                     "filesize":image_file.size}
    #     st.write(file_details)

        # To View Uploaded Image
        # resize = T.Resize((224, 224))
        # print('Type':, type(image_file))
        dictionar_class = {
            0: 'Не пицца',
            1: 'Пицца'}

        decode = lambda x: dictionar_class[int(x)] 

        resize = T.Resize((224, 224))
        img = resize(load_image(image_file)/255)
        model = torch.load_state_dict(torch.load('/home/anna/cv_project/model_pizza.pt', map_location=torch.device('cpu')))
        model.eval()
        # plt.imshow(torch.permute(img, (1, 2, 0)))
        label = decode(torch.round(model(img.unsqueeze(0)).sigmoid()).item())
        st.image(load_image(image_file), width=250)
        st.title(label)
        

if __name__ == "__main__":
    main()