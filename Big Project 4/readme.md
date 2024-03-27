The outlining of how to use the code: 

First you must install the respective packages which are needed for the code to run on both the simulation.py file and the training.py file.

All these packages and libraries must be imported:
csv
random
nltk
from nltk.corpus wordnet as wn
from nltk.tokenize word_tokenize
from nltk.corpus stopwords
from itertools chain
re
pandas as pd
json
os 
numpy as np
sklearn.model_selection import train_test_split
transformers import BertTokenizer, BertForSequenceClassification
torch
torch.utils.data import TensorDataset, DataLoader
transformers import AdamW, get_linear_schedule_with_warmup
ast
sklearn.metrics import classification_report
sklearn.metrics import confusion_matrix

The training.py file if you run the command python3 training.py, you train the original dataset and all the code from there will run 
with all the different tests on evaluation metrics also.
The simulation.py file if you run the command python3 simulation.py, you will create all the different transformed datasets. 

The first lines here: 

# Read the jsonl file into a list
# with open('/Users/arhamazhary/Downloads/train (1).jsonl') as f:
#     data = [json.loads(line) for line in f]

# # Convert the list to a pandas DataFrame
# df = pd.json_normalize(data)

# # Save the DataFrame as a CSV file
# df.to_csv('data.csv', index=False)

# print(df)

These are commented to help produce the original dataset so it can be manipulated. 
The data.csv file is already created and present so commented out. 
