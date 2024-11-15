import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from U2Net.model import U2NET
import cv2

def load_model():
    model_path = 'U2Net/saved_models/u2net/u2net.pth'
    model = U2NET()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def remove_background(image_path, output_path):
    model = load_model()

    img = Image.open(image_path).convert("RGB")
    original_size = img.size

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)[0][0]

    mask = output.squeeze().cpu().numpy()
    mask = (mask > 0.2).astype(np.uint8) * 255
    mask = Image.fromarray(mask).resize(original_size, Image.LANCZOS)

    img_np = np.array(img)
    mask_np = np.array(mask)
    img_np[mask_np == 0] = 255

    result_img = Image.fromarray(img_np)
    result_img.save(output_path)
    return result_img
