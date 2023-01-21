import pandas as pd
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.layers import *
import tensorflow_hub as hub
from keras.models import Model
from sklearn.model_selection import train_test_split
import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel,Seq2SeqArgs
import json
from nltk.translate.meteor_score import meteor_score
from evaluate import load
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import evaluate

df_1 = pd.read_csv('dataset_csv_format.csv')
df_1.head()
doc_sug = {}
for i in range(len(df_1)):
  doc_sug[df_1['Video Link'][i]] = df_1['Doctor Suggestion Summary'][i]

model_args = Seq2SeqArgs()
model_args.num_train_epochs = 30
model_args.no_save = True
model_args.evaluate_generated_text = False
model_args.evaluate_during_training = False
model_args.evaluate_during_training_verbose = False
model_args.max_length=1500

# Initialize model
model = Seq2SeqModel(encoder_decoder_type="bart", 
                     encoder_decoder_name="facebook/bart-large", 
                     args=model_args, 
                     use_cuda=True, 
                     max_length=25, 
                     batch_size=2)

with open('/content/mmcs_with_foc.json', 'r') as json_file:
    data = json.load(json_file)

transcript=[]
pcs=[]
img_f=[]
aud_f=[]
intent=[]
age=[]
gender=[]
ds=[]

for i in range(len(list(data.keys()))):
    transcript.append('[INT ]' + data[str(i+1)]['intent'] + ' [AGE] ' + data[str(i+1)]['patient_age_group'] + ' [GEN] ' + data[str(i+1)]['patient_gender'] + ' [TRN] ' + data[str(i+1)]['transcript'])
    pcs.append(data[str(i+1)]['mmcs'])
    img_f.append(np.asarray(data[str(i+1)]['image_features'], dtype=float))
    aud_f.append(np.asarray(data[str(i+1)]['audio_features'], dtype=float))
    age.append(data[str(i+1)]['patient_age_group'])
    gender.append(data[str(i+1)]['patient_gender'])
    intent.append(data[str(i+1)]['intent'])

    ds.append(doc_sug[data[str(i+1)]['url']])

df = pd.DataFrame(list(zip(transcript, pcs, img_f, aud_f, age, gender, intent, ds)),
               columns =['transcript', 'pcs', 'image_features', 'audio_features', 'age_group', 'gender', 'intent', 'doctor_suggestion'])

for i in range(len(df)):
    if df['intent'][i].strip() == 'Affect':
        df['intent'][i] = 'Affect'
    if df['intent'][i] == 'Prevention':
        df['intent'][i] = 'Suggestion'

x_trn, x_test, y_train, y_test = train_test_split(df, 
                                                df['intent'], 
                                                stratify=df['intent'],
                                                test_size=0.15, 
                                                random_state=0)

trn_df = x_trn[['transcript', 'doctor_suggestion']]
val_df = x_test[['transcript', 'doctor_suggestion']]

trn_df.columns=['input_text', 'target_text']
val_df.columns=['input_text', 'target_text']

trn_df

# Train the model
model.train_model(trn_df, eval_data=val_df)

val_df = val_df.reset_index()
generated=[]
actual=[]
patient_dialogue=[]
for i in range(len(val_df)):
  actual.append(val_df['target_text'][i])
  patient_dialogue.append(val_df['input_text'][i])
  pred = model.predict([val_df['input_text'][i]])[0]

  generated.append(pred)

new = pd.DataFrame(list(zip(patient_dialogue, actual, generated)),
               columns =['Input text', 'Actual', 'Generated'])


bertscore = load("bertscore")
meteor = evaluate.load('meteor')
  
rouge = Rouge()
bert_scores=0
one_gram=0
two_gram=0
three_gram=0
four_gram=0
rouge_1=0
rouge_2=0
rouge_l=0
meteor_score=0

df = new

for i in range(len(df)):
  predictions = df['Generated'][i]
  references  = df['Actual'][i].replace('\n', '')


  #for BLEU score
  one_gram = one_gram + sentence_bleu([references.split()], predictions.split(), weights=(1, 0, 0, 0))
  two_gram = two_gram + sentence_bleu([references.split()], predictions.split(), weights=(0.5, 0.5, 0, 0))
  three_gram = three_gram + sentence_bleu([references.split()], predictions.split(), weights=(0.33, 0.33, 0.33, 0))
  four_gram = four_gram + sentence_bleu([references.split()], predictions.split(), weights=(0.25, 0.25, 0.25, 0.25))

  #for ROUGE score
  if references == '.':
    references='empty'
  rouge_scores = rouge.get_scores(predictions, references)
  rouge_1 = rouge_1 + rouge_scores[0]['rouge-1']['f']
  rouge_2 = rouge_2 + rouge_scores[0]['rouge-2']['f']
  rouge_l = rouge_l + rouge_scores[0]['rouge-l']['f']

  #for METEOR score
  meteor_score = meteor_score + meteor.compute(predictions=[predictions], references=[references])['meteor']

#print('bert_scores : ', bert_scores/len(df))
print('BLEU 1      : ', one_gram/len(df))
print('BLEU 2      : ', two_gram/len(df))
print('BLEU 3      : ', three_gram/len(df))
print('BLEU 4      : ', four_gram/len(df))
print('ROUGE 1     : ', rouge_1/len(df))
print('ROUGE 2     : ', rouge_2/len(df))
print('ROUGE L     : ', rouge_l/len(df))
print('METEOR      : ', meteor_score/len(df))

new.to_csv('generated.csv', index=False)