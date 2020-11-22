import cv2
from PIL import Image
import numpy as np
import torch
import PIL.ImageFont as ImageFont
from torchvision import transforms as tfs

from point5.dataset import draw_ann, de_normal_anns
from point5.model import CnnAlign

MODEL_FACE_ALIGN  = "./output/alignment800.pt"

font_size = 8
font1 = ImageFont.truetype(r'./Ubuntu-B.ttf', font_size)


def cv2image(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image

def image2cv(image):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image

def get_pytorch_model(path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    model = CnnAlign().to(device)
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['net'])
    model.eval()
    return model

input_size = 64

if __name__ == "__main__":
    face_align_model = get_pytorch_model(MODEL_FACE_ALIGN)
    to_tensor = tfs.ToTensor()
    cap = cv2.VideoCapture(0)
    while(1):
        # get a frame
        ret, img = cap.read()
        image = cv2image(img)
        image = image.resize((input_size, input_size))
        output = face_align_model(to_tensor(image).view(-1, 3, input_size, input_size))
        anns = output[0].cpu().detach().numpy()
        anns = np.resize(anns, (10, 2))
        anns = de_normal_anns(anns, input_size, input_size)
        draw_ann(image, anns, font1, font_size)
        image = image.resize((480, 480))
        img = image2cv(image)
        cv2.imshow("test", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()