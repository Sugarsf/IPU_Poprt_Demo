import cv2
import numpy as np
import onnxruntime as ort


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
# 加载模型
model_path = '../models/onnx_official/resnet50-v1-7.onnx'
sess = ort.InferenceSession(model_path)

# 准备输入数据
img_path = 'cat.jpeg'
img = cv2.imread(img_path)

# 预处理输入数据
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.resize(img, (224, 224))
img = img.astype(np.float32) / 255.0
img_norm= np.transpose(img, (2, 0, 1))

img_norm[:,:,0] = (img_norm[:,:,0] - mean[0]) / std[0]
img_norm[:,:,1] = (img_norm[:,:,1] - mean[1]) / std[1]
img_norm[:,:,2] = (img_norm[:,:,2] - mean[2]) / std[2]

input_batch = np.expand_dims(img_norm,axis=0)
input_batch = np.ascontiguousarray(input_batch)


# 将数据输入到模型中
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
outputs = sess.run([output_name], {input_name: input_batch})

# 后处理输出结果
predictions = np.squeeze(outputs[0])
predicted_class = np.argmax(predictions)

print (labels[predicted_class])
