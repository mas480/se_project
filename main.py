from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()
pipe = pipeline(model="ai-forever/rugpt3large_based_on_gpt2")

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict/")
def predict(item: Item):
    return pipe(item.text)[0]['generated_text']
