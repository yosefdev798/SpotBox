FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xfixes0
RUN pip install --no-cache-dir -r requirements.txt
COPY project_code/ ./project_code/
COPY photos/ ./photos/
EXPOSE 8000
CMD ["uvicorn", "project_code.app:app", "--host", "0.0.0.0", "--port", "8000"]