import math
import torch
import torchvision
import time
from PIL import Image
from torchvision.transforms import functional as F
import torchvision.transforms as T
import cv2

class HumanDetection:
    def __init__(self) -> None:
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model = self.model.cuda()  # Move model to GPU
        self.model.eval()  # Evaluation mode

    def get_human_bbox(self, cv2_image):
        image_cv2_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        transform = T.Compose([T.ToTensor()])
        image_tensor = transform(image_cv2_rgb).unsqueeze(0).cuda()

        with torch.no_grad():
            prediction = self.model(image_tensor)

        confidence_threshold = 0.9

        filtered_boxes = []
        for box, score, label in zip(prediction[0]['boxes'], prediction[0]['scores'], prediction[0]['labels']):  # prediction of model for 1st image
            if score >= confidence_threshold and label == 1:  # Filtering for humans with high confidence
                filtered_box = box.cpu().numpy().tolist()
                filtered_box = [math.ceil(pos) for pos in filtered_box]
                filtered_boxes.append(filtered_box)

        return filtered_boxes
    
    def draw_bbox(self, bbox, image):
        top_left = (bbox[0], bbox[1])  # (x1, y1)
        bottom_right = (bbox[2], bbox[3])  # (x2, y2)

        color = (0, 255, 0)  # Green
        thickness = 2  # Set to -1 for a filled rectangle
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

# image_path = '/home/dai/MCGaze/240.jpg'
# image_cv2 = cv2.imread(image_path)
# detector = HumanDetection()
# filtered_boxes = detector.get_human_bbox(image_cv2)
# for box in filtered_boxes:
#     detector.draw_bbox(box, image_cv2)
# cv2.imwrite('/home/dai/MCGaze/240_test.jpg', image_cv2)
# print(type(filtered_boxes), len(filtered_boxes), type(filtered_boxes[0]))