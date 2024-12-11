import os
from flask import Flask, request, render_template, redirect, url_for, Response, send_from_directory
import json
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import faiss
from PIL import Image
import pandas as pd
import pickle


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
CUR_DIR = os.getcwd()
OLD_PREFIX = "/kaggle/working/"
NEW_PREFIX = CUR_DIR + "/static/flickr8k/"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


image = None

def adjust_paths_in_database(database_dict, old_prefix, new_prefix):
    new_database_dict = {}
    for path, embedding in database_dict.items():
        if path.startswith(old_prefix):
            new_path = path.replace(old_prefix, new_prefix)
            new_database_dict[new_path] = embedding
        else:
            new_database_dict[path] = embedding
    return new_database_dict

def generate_image_embedding(image_path, is_path=False):
    image = image_path
    if is_path:
        image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy().flatten()


def retrieve_similar_images(query_image_path, database_dict, top_k=5):
    query_embedding = generate_image_embedding(query_image_path)

    database_paths = list(database_dict.keys())
    database_embeddings = np.array(list(database_dict.values()), dtype=np.float32)

    dimension = database_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(database_embeddings)

    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    similar_images = [database_paths[i] for i in indices[0]]
    similar_scores = (1 / (1 + distances[0])).tolist()

    return similar_images, similar_scores

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-result', methods=['POST'])
def handle_request():
    action = request.form.get('action')
    similar_images = []
    scores = []
    results = []

    if action == 'cbir':
        similar_images, scores = retrieve_similar_images(image, database_dict, top_k=10)

    elif action == 'cir':
        pass

    for i, image_path in enumerate(similar_images):
        relative_path = image_path.replace(CUR_DIR + "/static/", "")
        image_url = url_for('static', filename=relative_path)
        results.append({
            'similarity_score': scores[i],
            'image_path': image_url
        })

    return Response(response=json.dumps(results), mimetype='application/json')


@app.route('/upload', methods=['POST'])
def upload_file():
    global image
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    file.save(file_path)
    image = Image.open(file_path).convert('RGB')
    return redirect(url_for('index', filename=filename))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    with open("database_mini.pkl", "rb") as f:
        database_dict = pickle.load(f)

    database_dict = adjust_paths_in_database(database_dict, OLD_PREFIX, NEW_PREFIX)

    df = pd.read_csv(f'{CUR_DIR}/static/flickr8k/captions.txt')
    df['image'] = f'{CUR_DIR}/static/flickr8k/Images/' + df['image']
    flickr_images = list(df['image'].unique())

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir=CUR_DIR)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir=CUR_DIR)

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run()
