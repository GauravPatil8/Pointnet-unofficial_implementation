import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import streamlit as st
import plotly.express as px
from utils import load_model_and_data, preprocess_pointcloud, plot_3d_point_cloud, classify_point_cloud, load_local_css
from src.utils.data import read_off

def main():
    st.set_page_config(
        page_title="PointNet 3D Classifier",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    load_local_css(os.path.join(os.path.dirname(__file__),"styles.html"))

    st.markdown('<h1 class="title-header"> PointNet 3D classification </h1>', unsafe_allow_html=True)

    model, classes, test_objects = load_model_and_data()

    col1, col2 = st.columns([1,1], gap='large')

    with col1:
        st.markdown('<h1 class="section-header">3D Model Selection</h1>', unsafe_allow_html=True)

        selected_object = st.selectbox(
            label="Select 3D model",
            options=test_objects.keys(),
            help="Select a 3D point cloud model to visualize and classify"
        )

        test_object = test_objects[selected_object]
        sampled_points = read_off(test_object, 2048)
        points = preprocess_pointcloud(test_object, num_points=2048)

        st.markdown('<h2 class="section-header">Model Visualization</h2>', unsafe_allow_html=True)
        fig = plot_3d_point_cloud(sampled_points, f"{selected_object} Point Cloud")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #3498db; margin-bottom: 0.5rem;">Model Information</h3>
            <p><strong>Selected Model:</strong> {selected_object}</p>
            <p><strong>Number of Points:</strong> {len(points):,}</p>
            <p><strong>Dimensions:</strong> {points.shape[1]}D</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<h2 class="section-header">Classification</h2>', unsafe_allow_html=True)

        if st.button("Classify 3D Model", type="primary"):
            with st.spinner("Analyzing 3D model..."):

                
                predicted_class, confidence, all_probabilities = classify_point_cloud(
                    model, points, classes
                )
                
                
                st.markdown(f"""
                <div class="classification-result">
                    <h3 style="color: #27ae60; margin-bottom: 1rem;">Classification Result</h3>
                    <h2 style="color: #2c3e50; margin-bottom: 0.5rem;">{predicted_class}</h2>
                    <p style="font-size: 1.1rem; color: #7f8c8d;">
                        Confidence: <strong>{confidence:.2%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                
                st.markdown('<h3 class="section-header">Probability Distribution</h3>', unsafe_allow_html=True)
                
                prob_df = pd.DataFrame({
                    'Class': classes,
                    'Probability': all_probabilities
                })

                # Optional: sort only for better visualization, do not limit to top-5
                prob_df = prob_df.sort_values(by='Probability', ascending=True)

                fig_bar = px.bar(
                    prob_df, 
                    x='Probability', 
                    y='Class',
                    orientation='h',
                    color='Probability',
                    color_continuous_scale='Greys',
                    title="Classification Probabilities",
                    text=prob_df['Probability'].round(3)  # Show probability on the bars
                )

                # Update layout
                fig_bar.update_layout(
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    title=dict(
                        x=0.5,
                        font=dict(size=16, color='#1a1a1a')
                    ),
                    xaxis_title="Probability",
                    yaxis_title="Class",
                    font=dict(color='#1a1a1a'),
                    height=400,
                    showlegend=False
                )

                # Optional: Increase bar text visibility
                fig_bar.update_traces(
                    textposition='inside',
                    insidetextanchor='start',
                    cliponaxis=False
                )

                # Show plot
                st.plotly_chart(fig_bar, use_container_width=True)
                
                
                with st.expander("Model Architecture Details"):
                    st.markdown("""
                    **PointNet Architecture:**
                    - Input: Point cloud with N points × 3 coordinates
                    - Feature extraction: 3 → 64 → 128 → 1024 dimensions
                    - Global max pooling for permutation invariance
                    - Classification: 1024 → 512 → 256 → 4 classes
                    - Batch normalization and dropout for regularization
                    
                    **Classes:** Cube, Sphere, Cylinder, Cone
                    """)

if __name__ == '__main__':
    main()
        