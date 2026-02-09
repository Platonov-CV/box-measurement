import cv2
import numpy as np
import torch
from ultralytics import YOLO
from models.Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2


FOCAL_LENGTH = 0.004
SENSOR_SIZE = (0.00376, 0.0028)

# prepare detection model
det_model = YOLO('models/yolo11n_object365.pt')

# prepare depth model
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vits' # or 'vits', 'vitb'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
depth_model.load_state_dict(torch.load(f'models/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
depth_model.eval()

# prepare video capture
cap = cv2.VideoCapture(0)
frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(SENSOR_SIZE[0] / SENSOR_SIZE[1], frame_size[0] / frame_size[1])

# main loop
while True:
    ret, raw_frame = cap.read()

    depth_frame = depth_model.infer_image(raw_frame)
    out_frame = cv2.cvtColor(depth_frame / 20, cv2.COLOR_GRAY2BGR)

    results = det_model.predict(raw_frame)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        names = result.names

        for box, class_index in zip(boxes, classes):
            name = names[int(class_index)]
            if name != "Storage box": continue

            xmin, ymin, xmax, ymax = box

            norm_bbox_size = tuple(x / frame_side for x, frame_side in zip((xmax - xmin, ymax - ymin), frame_size))
            sensor_bbox_size = tuple(x * sensor_side for x, sensor_side in zip(norm_bbox_size, SENSOR_SIZE))

            xmin, ymin, xmax, ymax = [int(x) for x in [xmin, ymin, xmax, ymax]]

            # calculate real bbox size
            median_box_depth = np.median(depth_frame[ymin:ymax, xmin:xmax])
            real_bbox_size = tuple(x * median_box_depth / FOCAL_LENGTH for x in sensor_bbox_size)

            out_frame[ymin:ymax, xmin:xmax, :] = raw_frame[ymin:ymax, xmin:xmax, :] / 255

            # draw size labels
            width_text_org = [int((xmax + xmin) / 2), ymin]
            height_text_org = [xmax, int((ymax + ymin) / 2)]
            string_box_size = tuple(str(round(x, 2)) + "m" for x in real_bbox_size)
            out_frame = cv2.putText(
                img=out_frame, text=string_box_size[0], org=width_text_org, fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1, thickness=1, color=(255, 255, 255)
            )
            out_frame = cv2.putText(
                img=out_frame, text=string_box_size[1], org=height_text_org, fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1, thickness=1, color=(255, 255, 255)
            )

    cv2.imshow('frame', out_frame)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()