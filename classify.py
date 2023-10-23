from poprt import runtime
import numpy as np
import cv2 as cv
from poprt import Backend
import onnx

labels = {}

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_labels():
    labels_file = 'label.txt'
    with open(labels_file, 'r') as f:
        label_id = 0
        for line in f:
            label_name = line.strip().split('\t')
            labels[label_id] = label_name
            label_id += 1



get_labels()

# opencv data process
image = cv.imread("cat.jpeg")

height,width = image.shape[0:2]

img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
img = cv.resize(img,(224,224))
img = img.astype(np.float32) / 255.0
img_norm = np.transpose(img,(2,0,1))

img_norm[:,:,0] = (img_norm[:,:,0] - mean[0]) / std[0]
img_norm[:,:,1] = (img_norm[:,:,1] - mean[1]) / std[1]
img_norm[:,:,2] = (img_norm[:,:,2] - mean[2]) / std[2]

input_batch = np.expand_dims(img_norm,axis=0)
input_batch = np.ascontiguousarray(input_batch)


# # 创建 ModelRunner 实例, 并加载 PopEF 文件
runner = runtime.ModelRunner('../executable.popef')

# 获取模型输出信息
outputs = runner.get_model_outputs()

# 创建输入输出数据
input_dict = {
    "data": input_batch.astype(np.float16),
}

output_dict = {x.name: np.zeros(x.shape).astype(x.numpy_data_type()) for x in outputs}

# 运行 PopEF3
runner.execute(input_dict, output_dict)
output_list = np.squeeze(output_dict['resnetv17_dense0_fwd'])
output_class_index = np.argmax(output_list)
output_class = labels[output_class_index][0]

new_height = int(height*0.1)
new_width = int(width*0.9)


cv.putText(image,output_class,(new_width, new_height),cv.FONT_HERSHEY_COMPLEX,2.0,(100, 200, 200),5)
cv.imwrite('out.jpg',image)