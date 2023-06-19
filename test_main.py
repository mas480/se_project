import requests
from main import load_model

model_gpt2 = load_model()


def test_connect_streamlit():
    x = requests.get('https://streamlit.io/')
    assert x.status_code == 200


def test_type_model(model_gpt2):
    assert str(type(model_gpt2)) == "<class 'transformers.pipelines.text_generation.TextGenerationPipeline'>"


def test_example_1(model_gpt2):
    x = model_gpt2('Сегодня')[0]['generated_text']
    assert x == "Сегодня, в день рождения, я хочу поздравить всех, кто любит и ценит свою Родину, кто"


def test_example_2(model_gpt2):
    x = model_gpt2('Завтра')[0]['generated_text']
    assert x == "Утром, когда я проснулся, я увидел, что в комнате горит свет. Я подумал, что это"
