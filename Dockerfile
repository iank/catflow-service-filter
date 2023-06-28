FROM python:3.10 as base

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir .

CMD ["catflow-service-filter"]
