FROM python:3.10

ENV PYTHONUNBUFFERED=1

ENV HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /code
COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /code

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
