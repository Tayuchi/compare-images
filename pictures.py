# 南無阿弥陀仏
#
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |# '.
#                 / \\|||  :  |||# \
#                / _||||| -:- |||||- \
#               |   | \\\  -  #/ |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永无BUG
#
#
from flask import Flask, request, jsonify, send_file
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from scipy.spatial import distance
from flask_cors import CORS, cross_origin
import base64
import logging
from logging.handlers import RotatingFileHandler
import os

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ResNet50モデルをロード（事前学習済み）
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()  # 推論モード

app = Flask(__name__)
CORS(app, resources={r"/compare-images": {"origins": "*"}})

# 画像の前処理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 画像データから特徴ベクトルを取得する関数
def get_vector(image_data):
    # image_data が bytes であれば、BytesIO を作成
    if isinstance(image_data, bytes):
        img = Image.open(BytesIO(image_data))
    else:
        # すでに BytesIO オブジェクトの場合はそのまま使う
        img = Image.open(image_data)

    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        features = model(batch_t)
    return features.numpy().flatten()

@app.route('/compare-images', methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def compare_images():
    data = request.get_json()
    if not data or 'image_url1' not in data or 'image_url2' not in data:
        logger.error("Both image URLs are required.")
        return jsonify({"error": "Both image URLs are required."}), 400
    logger.info("Completed checking the recieve data structure!")

    image_url1 = data['image_url1']
    try:
        response1 = requests.get(image_url1)
        response1.raise_for_status()
        image_data1 = response1.content
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image_url1: {str(e)}")
        return jsonify({"error": f"Failed to download image_url1: {str(e)}"}), 500
    logger.info("Completed downloading image_url1!")

    image_url2 = data['image_url2']
    try:
        base64_encoded_data = image_url2.split(',')[1]
        image_data2 = base64.b64decode(base64_encoded_data)
    except Exception as e:
        logger.error(f"Failed to decode or process image_url2: {str(e)}")
        return jsonify({"error": f"Failed to decode or process image_url2: {str(e)}"}), 400
    logger.info("Completed decoding and process image_url2!")

    try:
        img_vec1 = get_vector(BytesIO(image_data1))
        img_vec2 = get_vector(BytesIO(image_data2))
    except Exception as e:
        logger.error(f"Failed to process images: {str(e)}")
        return jsonify({"error": f"Failed to process images: {str(e)}"}), 500
    logger.info("Completed processing images!")

    dist = distance.euclidean(img_vec1, img_vec2)
    logger.info("Completed calcuating similarity!")
    return jsonify({"similarity_score": round(dist, 2)})

@app.route('/logs')
def logs():
    try:
        return send_file('app.log')
    except Exception as e:
        logger.error(f"Failed to retrieve log file: {str(e)}")
        return str(e), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
