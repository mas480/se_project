import streamlit as st
from transformers import pipeline


# @st.cache(allow_output_mutation=True)


def load_model():
    return pipeline(model="ai-forever/rugpt3large_based_on_gpt2")


def get_text():
    """Загрузка изображения средствами Streamlit"""
    text_in = st.text_input(label='Наберите текст')
    if text_in:
        st.write("You entered: ", text_in)
        return text_in


pipe = load_model()

"""Выводим заголовок страницы средствами Streamlit"""
st.title('Приложение, генерирующее продолжение фразы')
"""Вызываем функцию для набора текста"""
txt = get_text()

result = st.button('Сгенерировать продолжение')


def print_predictions(txt: str):
    return pipe(txt)[0]['generated_text']


if result:
    for_print = print_predictions(txt)
    st.write('Результат: ', '\n', for_print)
