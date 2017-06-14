# coding=utf-8

from model.load import init_model
from flask import Flask, request, render_template
from scipy.misc import imread, imresize
import numpy as np
import re
import base64
import argparse

app = Flask(__name__)

global model, graph
model, graph = init_model()


def convert_image(img_data):
    """将base64编码的解码保存图片"""
    img_str = re.search(b'base64,(.*)', img_data).group(1)
    with open('draw.png', 'wb') as output:
        output.write(base64.decodebytes(img_str))


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    img_data = request.get_data()
    convert_image(img_data)
    x = imread('draw.png', mode='L')
    # 训练的MNIST图片是黑底白字，所以需要将读取图片颜色翻转
    x = np.invert(x)
    x = imresize(x, (28, 28))
    # 训练模型输入数据是做了归一化的，predict也需要，这点非常重要
    x = x.astype('float32')
    x /= 255
    x = x.reshape(1, 28, 28, 1)
    with graph.as_default():
        out = model.predict(x)
        response = np.array_str(np.argmax(out, axis=1))
        return response


if __name__ == '__main__':

    parser = argparse.ArgumentParser('some server parameters')
    parser.add_argument('-p', help='server port', dest='port', type=int, default=5000)
    args = parser.parse_args()

    port = args.port
    # 监听所有网络
    app.run(host='0.0.0.0')
