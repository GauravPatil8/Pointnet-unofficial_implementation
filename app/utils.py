import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import random
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.constants import CONFIG_PATH, PROJECT_ROOT
from src.utils.common import read_yaml
from src.entities.pointnet import PointNet
from src.utils.data import read_off, get_classes

def load_local_css(file_path: str):
    """Inject local CSS into Streamlit app."""
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def classify_point_cloud(model, points, class_names):
    """Classify a point cloud using the PointNet model"""

    model.eval()
    
    with torch.no_grad():
        logits = model(points)
        probabilities = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        confidence = probabilities[0][pred_class].item()
    return class_names[pred_class], confidence, probabilities[0].numpy()

def plot_3d_point_cloud(points, title="3D Point Cloud"):
    """Create an interactive 3D plot of the point cloud"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=points[:, 2],  # Color by z-coordinate
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="Z-coordinate")
        ),
        hovertemplate='<b>Point</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=18, color='#2c3e50')
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            bgcolor='rgba(248,249,250,0.8)',
            xaxis=dict(backgroundcolor="white"),
            yaxis=dict(backgroundcolor="white"),
            zaxis=dict(backgroundcolor="white"),
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

@st.cache_resource
def load_model_and_data():
    "loads PointNet model and testing data"

    config = read_yaml(os.path.join(PROJECT_ROOT, CONFIG_PATH))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PointNet().to(device)
    model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, config['trainer']['model_path']), map_location=device))

    classes = get_classes(os.path.join(PROJECT_ROOT, config["data_ingestion"]["extract_path"],"ModelNet10"))

    test_objects = dict()

    for cls in classes:
        test_objects[cls] = get_file(cls)

    return model, classes, test_objects


def preprocess_pointcloud(file_path, num_points=1024):
    pc = read_off(file_path, num_points)

    if pc.shape[0] > num_points:
        choice = np.random.choice(pc.shape[0], num_points, replace=False)
    else:
        choice = np.random.choice(pc.shape[0], num_points, replace=True)

    pc = pc[choice]
    pc = pc - np.mean(pc, axis=0)
    pc = pc / np.max(np.linalg.norm(pc, axis=1))

    return torch.from_numpy(pc).float().unsqueeze(0)  # [1, N, 3]

def get_file(model_class):
    config = read_yaml(os.path.join(PROJECT_ROOT, CONFIG_PATH))
    data_dir = os.path.join(PROJECT_ROOT, config["data_ingestion"]["extract_path"], "ModelNet10")
    folder_path = os.path.join(data_dir, model_class.lower(), "test")
    model_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    if not model_files:
        raise FileNotFoundError("No files found in the directory.")
    
    return os.path.join(folder_path, random.choice(model_files))

def get_labels(data_dir):
    classes = get_classes(data_dir)
    labels = []
    for i, _ in enumerate(classes):
        labels.append(i)
    return labels
    
if __name__ == '__main__':
    config = read_yaml(os.path.join(PROJECT_ROOT, CONFIG_PATH))
    data_dir = (os.path.join(PROJECT_ROOT,config["data_ingestion"]["extract_path"], "ModelNet10"))
    print(get_classes(data_dir))
    print(get_labels(data_dir))

