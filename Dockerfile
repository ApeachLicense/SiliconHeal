FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

ENV PORT=8080
ENV HOST=0.0.0.0
ENV LC_ALL=C
ENV LANG=C

EXPOSE 8080

CMD ["python", "backend/server.py"]
