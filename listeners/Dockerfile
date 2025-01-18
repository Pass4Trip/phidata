FROM python:3.11-slim

WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
COPY postgres_rabbitmq_listener.py .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Commande de démarrage
CMD ["python", "postgres_rabbitmq_listener.py"]
