from fastapi import FastAPI

from .app import get_app

app: FastAPI = get_app()
