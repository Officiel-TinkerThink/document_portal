# Use official python image
FROM python:3.12-slim

# set environment varaible
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# set workdir
WORKDIR /app

# install OS dependencies
RUN apt-get update && apt-get install -y build-essential poppler-utils && rm