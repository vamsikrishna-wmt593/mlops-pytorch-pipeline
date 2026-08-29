# mlops-pytorch-pipeline

A production-style MLOps pipeline that trains a PyTorch image classifier (ResNet-18 on CIFAR-10) and serves predictions via a REST API — containerized with Docker and orchestrated on Kubernetes.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Git / CI                             │
│   feature branch → PR → develop → main                     │
│   GitHub Actions: lint + test on every push                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
          ┌─────────────▼──────────────┐
          │      Docker Images         │
          │  mlops-train:v1            │
          │  mlops-serve:v1            │
          └─────────────┬──────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                   Kubernetes (ml-training ns)                │
│                                                             │
│  ┌──────────────────────┐    ┌───────────────────────────┐  │
│  │   Job: model-training│    │ Deployment: model-serving │  │
│  │  - ResNet-18 training│    │  - 2 replicas             │  │
│  │  - CIFAR-10 dataset  │    │  - FastAPI /predict       │  │
│  │  - ConfigMap config  │    │  - GET /health probes     │  │
│  │  - PVC: /app/data    │    │  - HPA: 2-6 replicas      │  │
│  │  - PVC: /app/ckpts   │    │  - PVC: /app/checkpoints  │  │
│  └──────────┬───────────┘    └───────────────────────────┘  │
│             │  checkpoint.pt        ▲                        │
│             └───────────────────────┘                        │
│                                                             │
│  ClusterIP Service: port 80 → container 8080                │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- Python 3.10+
- Docker
- kubectl
- Minikube (or any Kubernetes cluster)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/mlops-pytorch-pipeline.git
cd mlops-pytorch-pipeline
```

### 2. Install dependencies locally

```bash
pip install -r requirements/train.txt   # for training
pip install -r requirements/serve.txt   # for serving
```

### 3. Build Docker images

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

### 4. Run training locally

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1
```

### 5. Run serving locally

```bash
docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1
```

Test the endpoints:

```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

---

## Kubernetes Deployment

### Start Minikube

```bash
export MINIKUBE_HOME=/path/to/writable/dir/.minikube
export KUBECONFIG=/path/to/writable/dir/.kube/config
minikube start --driver=docker
```

### Load images into Minikube

```bash
minikube image load mlops-train:v1
minikube image load mlops-serve:v1
```

### Apply manifests

```bash
# namespace + config
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml

# run training job
kubectl apply -f k8s/training-job.yaml

# wait for training to complete
kubectl wait --for=condition=complete job/model-training -n ml-training --timeout=3600s

# deploy serving
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
```

### Monitor training

```bash
kubectl logs -f job/model-training -n ml-training
```

### Test prediction endpoint

```bash
kubectl port-forward svc/model-serving 8080:80 -n ml-training

curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

---

## Configuration

All training hyperparameters are in `configs/training_config.yaml`:

| Parameter | Default |
|---|---|
| architecture | resnet18 |
| num_classes | 10 |
| epochs | 10 |
| batch_size | 64 |
| learning_rate | 0.001 |
| early_stopping_patience | 3 |

---

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push:
- Lint with `flake8`
- Unit tests with `pytest`
