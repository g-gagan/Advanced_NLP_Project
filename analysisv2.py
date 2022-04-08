
!pip install transformers

import torch
from transformers import AutoTokenizer,BertTokenizerFast, BertForQuestionAnswering


# Define the bert tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load the fine-tuned modeol
# model = torch.load("/content/drive/MyDrive/Spring22/CS769/Project/model.pt",map_location=torch.device('cpu'))
model = BertForQuestionAnswering.from_pretrained('kaporter/bert-base-uncased-finetuned-squad')
model.eval()


def predict(context,query):
  inputs = tokenizer.encode_plus(query, context, return_tensors='pt')

  outputs = model(**inputs)
  answer_start = torch.argmax(outputs[0])  # get the most likely beginning of answer with the argmax of the score
  answer_end = torch.argmax(outputs[1]) + 1 

  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

  return answer

def normalize_text(s):
  import string, re

  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
  pred_tokens = normalize_text(prediction).split()
  truth_tokens = normalize_text(truth).split()

  if len(pred_tokens) == 0 or len(truth_tokens) == 0:
    return int(pred_tokens == truth_tokens)
  
  common_tokens = set(pred_tokens) & set(truth_tokens)
  
  if len(common_tokens) == 0:
    return 0
  
  prec = len(common_tokens) / len(pred_tokens)
  rec = len(common_tokens) / len(truth_tokens)
  
  return 2 * (prec * rec) / (prec + rec)

def give_an_answer(context,query,answer):

  prediction = predict(context,query)
  em_score = compute_exact_match(prediction, answer)
  f1_score = compute_f1(prediction, answer)

  print(f"Question: {query}")
  print(f"Prediction: {prediction}")
  print(f"True Answer: {answer}")
  print(f"EM: {em_score}")
  print(f"F1: {f1_score}")
  print("\n")


import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import time

# Give the path for validation data
!wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O dev-v2.0.json

path = Path('dev-v2.0.json')

# Open .json file
with open(path, 'rb') as f:
    squad_dict = json.load(f)

texts = []
queries = []
answers = []
qa_dict = {'what':[] , 'where': [], 'how': [], 'why':[], 'when': [], 'which':[],  'misc': [], 'who' : []}
qa_keys = ['what', 'where', 'how', 'why', 'when', 'which', 'who']

# Search for each passage, its question and its answer
for group in squad_dict['data']:
    for passage in group['paragraphs']:
        context = passage['context']
        for qa in passage['qas']:
            question = qa['question']
            id = qa['id']
            misc_flag = True
            for key in qa_keys:
                if key in question.lower():
                    qa_dict[key].append((id , question, context, [i['text'] for i in qa['answers']]))
                    misc_flag = False
                    break
            if misc_flag == True:
                qa_dict['misc'].append((id , question, context, [i['text'] for i in qa['answers']]))
            
                # Store every passage, query and its answer to the lists
                # texts.append(context)
                # queries.append(question)
                # answers.append(answer)

# val_texts, val_queries, val_answers = texts, queries, answers
qa_dict['how'][0]

[(key , len(qa_dict[key])) for key in qa_dict.keys()]
#DEV

#TRAIN
[(key , len(qa_dict[key])) for key in qa_dict.keys()]

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

pred_dict = {}
ctr = 0
q = 0
for key in qa_dict.keys():
  for element in qa_dict[key]:
    q+=1
    for x in element[3]:
      if has_numbers(x):
        ctr+=1

print(ctr, q-ctr, ctr/q*100)

len(pred_dict)

count = 0
for element in qa_dict['how']:
  print(pred_dict[element[0]], ':', element[3], end = '\t')
  if pred_dict[element[0]] in [x.lower() for x in element[3]]:
    count += 1
    print('matched')
  elif pred_dict[element[0]] == '' and len(element[3]) == 0:
    count += 1
    print('matched')
  else:
    print('')
print(count / len(pred_dict))

print(sum(([len(qa_dict[key]) for key in qa_dict.keys()])) , len(pred_dict))

from google.colab import drive
drive.mount('/content/drive')


with open('kaporter_predictions.json', 'w') as fp:
    json.dump(pred_dict, fp)