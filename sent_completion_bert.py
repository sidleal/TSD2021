import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import io
import pandas as pd
from google.colab import files


# Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased')

# Load pre-trained model (weights)
# model = BertForMaskedLM.from_pretrained('neuralmind/bert-base-portuguese-cased')
model = BertForMaskedLM.from_pretrained('neuralmind/bert-large-portuguese-cased')
model.eval()


uploaded = files.upload()

df = pd.read_csv(io.BytesIO(uploaded['dataset_v1.tsv']), sep="\t", header=0)
dfa = pd.read_csv(io.BytesIO(uploaded['dataset_v1_answers.tsv']), sep="\t", header=0)

df_full = pd.merge(df, dfa, on='id')
df_full.head()



# df_human = df_full.query('id in [6,3,8,16,19,26,29,31,49,50,60,91,79,77,55,54,68,110,119,116]')
# df_human.head()



def bert_prediction(text, candidates):
  text = '[CLS] %s [SEP]' % text[:-1]
  text = text.replace('__________', '[MASK]')
  print(text)
  tokenized_text = tokenizer.tokenize(text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

  # Create the segments tensors.
  segments_ids = [0] * len(tokenized_text)

  # Convert inputs to PyTorch tensors
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])

  # Predict all tokens
  with torch.no_grad():
      predictions = model(tokens_tensor, segments_tensors)[0]

  masked_index = tokenized_text.index('[MASK]') 

  candidates_ids = tokenizer.convert_tokens_to_ids(candidates)

  predictions_candidates = predictions[0, masked_index, candidates_ids]
  predicted_index = torch.argmax(predictions_candidates).item()
  predicted_token = candidates[predicted_index]
  # print("----------->", predicted_token)

  return predicted_token




hits = 0
for index, row in df_full.iterrows():
# for index, row in df_human.iterrows():
    print(index, row['question'], row['answer'])
    candidates = row[['a', 'b', 'c', 'd', 'e']]
    predicted = bert_prediction(row['question'], candidates)
    target = row[row['answer']]
    print("Candidates", list(candidates))
    print('-----> Target:', target, "=> Predicted:", predicted, "\n")
    if target == predicted:
      hits += 1

print('Total hits: ', hits)


