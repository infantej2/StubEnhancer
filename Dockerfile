FROM python:3.10.8-bullseye
EXPOSE 8050

WORKDIR /

RUN git clone https://github.com/infantej2/StubEnhancer
WORKDIR /StubEnhancer/

RUN pip install -r requirements.txt
CMD ["python3", "app.py"]