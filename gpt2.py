from transformers import pipeline

pipe = pipeline(model="ai-forever/rugpt3large_based_on_gpt2")

print(pipe("�ਢ��")[0]['generated_text'])
