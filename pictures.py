from flask import Flask, request, jsonify
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from scipy.spatial import distance
from flask_cors import CORS
import base64

# ResNet50モデルをロード（事前学習済み）
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()  # 推論モード

app = Flask(__name__)
# photopickle からのリクエストのみ許可
CORS(app, resources={r"/compare-images": {"origins": "https://photo-pickle.vercel.app/"}})

# 画像の前処理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 画像から特徴ベクトルを抽出する関数
def get_vector(image_data):
    img = Image.open(image_data)
    # RGBA画像をRGBに変換
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        features = model(batch_t)
    return features.numpy().flatten()

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = 'https://photo-pickle.vercel.app/'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

@app.route('/compare-images', methods=['POST'])
def compare_images():
    data = request.get_json()
    if not data or 'image_url1' not in data or 'image_url2' not in data:
        return jsonify({"error": "Both image URLs are required."}), 400

    image_url1 = data['image_url1']
    try:
        response1 = requests.get(image_url1)
        response1.raise_for_status()
        image_data1 = response1.content
    except requests.exceptions.RequestException:
        return jsonify({"error": "Failed to download image_url1."}), 500

    try:
        base64_encoded_data = data['image_url2'].split(',')[1]
        image_data2 = base64.b64decode(base64_encoded_data)
    except Exception:
        return jsonify({"error": "Failed to decode or process image_url2."}), 400

    try:
        img_vec1 = get_vector(BytesIO(image_data1))
        img_vec2 = get_vector(BytesIO(image_data2))
    except Exception:
        return jsonify({"error": "Failed to process images."}), 500

    dist = distance.euclidean(img_vec1, img_vec2)
    return jsonify({"similarity_score": round(dist, 2)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8080", debug=True)
