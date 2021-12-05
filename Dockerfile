# syntax=docker/dockerfile:1

FROM python:3.8.12-buster

WORKDIR /code

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip \
    && pip install --upgrade setuptools \
    && pip --no-cache-dir install -r requirements.txt

COPY . .