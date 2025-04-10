FROM python:3.12-slim AS builder

# Instalar dependencias de compilación
RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    libeigen3-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bindings.cpp .
RUN pip install pybind11
RUN python3 -m pybind11 --includes > pybind11_includes.txt
RUN g++ -O3 -Wall -shared -std=c++11 -fPIC $(cat pybind11_includes.txt) -I/usr/include/eigen3 bindings.cpp -o cosine_module.so

FROM python:3.12-slim

WORKDIR /app

# Copiar dependencias y archivos compilados
COPY --from=builder /app/cosine_module.so .
COPY --from=builder /app/requirements.txt .
COPY main.py .

# Instalar dependencias de runtime
RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Configurar caché
RUN mkdir -p /app/cache && chmod -R 777 /app/cache

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]