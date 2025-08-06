# Base image
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

#working dir
WORKDIR /pointnet

#installing requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#copying local code
COPY app/ ./app/
COPY artifacts ./artifacts/
COPY src/ ./src/
COPY config.yaml .
COPY run.py .

#setting root path
ENV PYTHONPATH=/pointnet

#command to run streamlit
CMD ["streamlit", "run", "app/main.py"]




