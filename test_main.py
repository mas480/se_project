import io
from stream import load_pipe
import streamlit as st


def load_model():
    model_gpt2 = load_model()
    return model_gpt2


def test_connect_streamlit():
    x = requests.get('https://streamlit.io/')
    assert x.status_code == 200


def test_type_model(load_model):
    assert str(type(load_model)) == "<class 'transformers.pipelines.text_generation.TextGenerationPipeline'>"


def test_example_1(load_model):
    x = load_model('Сегодня')[0]['generated_text']
    assert x == "Сегодня, в день рождения, я хочу поздравить всех, кто любит и ценит свою Родину, кто"


def test_example_2(load_model):
    x = load_model('Завтра')[0]['generated_text']
    assert x == "Утром, когда я проснулся, я увидел, что в комнате горит свет. Я подумал, что это"
    
