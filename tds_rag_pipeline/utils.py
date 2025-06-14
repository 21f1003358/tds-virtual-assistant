import os
import zipfile
import json
import pandas as pd
from bs4 import BeautifulSoup

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def load_markdown_texts(folder_path):
    md_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    md_texts.append(content)
    return md_texts

def load_discourse_json(json_dir):
    all_posts = []
    for file in os.listdir(json_dir):
        if file.endswith(".json"):
            with open(os.path.join(json_dir, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                posts = data.get("post_stream", {}).get("posts", [])
                all_posts.extend(posts)
    return pd.DataFrame(all_posts)

def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text().strip()

