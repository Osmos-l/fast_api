from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.cifar.predict import router as cifar_router
from api.mnist.predict import router as mnist_router

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://jordanlcq.vercel.app",
    "http://jordanlcq.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cifar_router, prefix="/cifar10")
app.include_router(mnist_router, prefix="/mnist")
