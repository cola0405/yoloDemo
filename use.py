from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import requests
import cv2
import numpy as np

def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
width, height = image.size

# load model
feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')


# extract features from image, then send it to model
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)


# deal with output logits
import torch
pred = torch.nn.functional.softmax(outputs.logits, dim=-1)
pred = pred.cpu().detach().numpy()[0]
d = {
    "0": "N/A",
    "1": "person",
    "2": "bicycle",
    "3": "car",
    "4": "motorcycle",
    "5": "airplane",
    "6": "bus",
    "7": "train",
    "8": "truck",
    "9": "boat",
    "10": "traffic light",
    "11": "fire hydrant",
    "12": "N/A",
    "13": "stop sign",
    "14": "parking meter",
    "15": "bench",
    "16": "bird",
    "17": "cat",
    "18": "dog",
    "19": "horse",
    "20": "sheep",
    "21": "cow",
    "22": "elephant",
    "23": "bear",
    "24": "zebra",
    "25": "giraffe",
    "26": "N/A",
    "27": "backpack",
    "28": "umbrella",
    "29": "N/A",
    "30": "N/A",
    "31": "handbag",
    "32": "tie",
    "33": "suitcase",
    "34": "frisbee",
    "35": "skis",
    "36": "snowboard",
    "37": "sports ball",
    "38": "kite",
    "39": "baseball bat",
    "40": "baseball glove",
    "41": "skateboard",
    "42": "surfboard",
    "43": "tennis racket",
    "44": "bottle",
    "45": "N/A",
    "46": "wine glass",
    "47": "cup",
    "48": "fork",
    "49": "knife",
    "50": "spoon",
    "51": "bowl",
    "52": "banana",
    "53": "apple",
    "54": "sandwich",
    "55": "orange",
    "56": "broccoli",
    "57": "carrot",
    "58": "hot dog",
    "59": "pizza",
    "60": "donut",
    "61": "cake",
    "62": "chair",
    "63": "couch",
    "64": "potted plant",
    "65": "bed",
    "66": "N/A",
    "67": "dining table",
    "68": "N/A",
    "69": "N/A",
    "70": "toilet",
    "71": "N/A",
    "72": "tv",
    "73": "laptop",
    "74": "mouse",
    "75": "remote",
    "76": "keyboard",
    "77": "cell phone",
    "78": "microwave",
    "79": "oven",
    "80": "toaster",
    "81": "sink",
    "82": "refrigerator",
    "83": "N/A",
    "84": "book",
    "85": "clock",
    "86": "vase",
    "87": "scissors",
    "88": "teddy bear",
    "89": "hair drier",
    "90": "toothbrush",
    "91": "N/A",
    "92": "N/A"
  }


# get pred boxes
boxes = outputs.pred_boxes.detach().numpy()[0]


image = pil2cv(image)

# loop through all prediction
for i in range(len(pred)):
    # filter p>0.5
    if max(pred[i]) > 0.5:
        res = d[str(list(pred[i]).index(max(pred[i])))]
        if res != 'N/A':
            # convert pred box coordinate and draw it
            import pybboxes as pbx
            box_voc = pbx.convert_bbox(boxes[i], from_type="yolo", to_type="voc", image_size=(width, height))
            cv2.rectangle(image, (box_voc[0], box_voc[1]), (box_voc[2], box_voc[3]), (0, 0, 255), 1)

            # put label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, '{} {:.3f}'.format(res, max(pred[i])), (box_voc[2], box_voc[1]), font, 0.5, (0, 255, 255), 1)


cv2.imshow('img', image)
cv2.waitKey(1000000)
cv2.destroyAllWindows()
