import csv
import random
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from itertools import chain
import re
import pandas as pd
import json
import os 

# # Read the jsonl file into a list
# with open('/Users/arhamazhary/Downloads/train (1).jsonl') as f:
#     data = [json.loads(line) for line in f]

# # Convert the list to a pandas DataFrame
# df = pd.json_normalize(data)

# # Save the DataFrame as a CSV file
# df.to_csv('data.csv', index=False)

# print(df)

input_csv_file = "/Users/arhamazhary/Desktop/Big Project/data.csv"
df = pd.read_csv(input_csv_file) 

label_counts = df['label'].value_counts()
print(label_counts)


nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Synonym transformation
def transform_synonyms(claim, pos_filter=('VB', 'JJ')):
    tokens = nltk.word_tokenize(claim)
    pos_tags = nltk.pos_tag(tokens)

    transformed_tokens = []
    for word, pos in pos_tags:
        if pos.startswith(pos_filter):
            synsets = wn.synsets(word, pos=nltk2wordnet(pos))
            if synsets:
                synonyms = set(chain.from_iterable([word.lemma_names() for word in synsets]))
                if word not in synonyms:
                    synonyms.add(word)
                chosen_synonym = random.choice(list(synonyms))
                transformed_tokens.append(chosen_synonym)
            else:
                transformed_tokens.append(word)
        else:
            transformed_tokens.append(word)

    transformed_claim = ' '.join(transformed_tokens)
    return transformed_claim

# Antonym transformation
def transform_antonyms(claim, pos_filter=('VB', 'JJ')):
    tokens = nltk.word_tokenize(claim)
    pos_tags = nltk.pos_tag(tokens)

    transformed_tokens = []
    for word, pos in pos_tags:
        if pos.startswith(pos_filter):
            synsets = wn.synsets(word, pos=nltk2wordnet(pos))
            if synsets:
                antonyms = set(chain.from_iterable([lem.antonyms() for lem in synsets[0].lemmas()]))
                if antonyms:
                    chosen_antonym = antonyms.pop().name()
                    transformed_tokens.append(chosen_antonym)
                else:
                    transformed_tokens.append(word)
            else:
                transformed_tokens.append(word)
        else:
            transformed_tokens.append(word)

    transformed_claim = ' '.join(transformed_tokens)
    return transformed_claim

# Negation transformation

def get_verb_base_form(verb):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    base_form = lemmatizer.lemmatize(verb, pos='v')
    return base_form

import re

def transform_negation(claim):
    tokens = nltk.word_tokenize(claim)
    pos_tags = nltk.pos_tag(tokens)

    transformed_tokens = []
    for i, (word, pos) in enumerate(pos_tags):
        if pos.startswith('VB'):
            base_form = get_verb_base_form(word)
            if word.lower() in ["is", "am", "are", "was", "were"]:
                transformed_tokens.append(f"{word} not")
            else:
                transformed_tokens.append(f"didn't {base_form}")
        else:
            transformed_tokens.append(word)

    transformed_claim = ' '.join(transformed_tokens)
    return transformed_claim

# Swap adjacent words transformation
def swap_adjacent_words(claim):
    tokens = nltk.word_tokenize(claim)
    stop_words = set(stopwords.words('english'))

    token_indices = list(range(len(tokens) - 1))  # Exclude the last token
    random.shuffle(token_indices)

    for i in token_indices:
        if (i + 1 < len(tokens) and tokens[i] not in stop_words
            and tokens[i + 1] not in stop_words and tokens[i + 1] != '.'):
            tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
            break

    transformed_claim = ' '.join(tokens)
    return transformed_claim

# Insert redundant phrase transformation
def insert_redundant_phrase(claim):
    redundant_phrases = [
        "As we know, ",
        "In fact, ",
        "It is known that ",
        "It is a fact that ",
    ]

    chosen_phrase = random.choice(redundant_phrases)
    transformed_claim = f"{chosen_phrase}{claim}"

    return transformed_claim
def transform_typo(claim):
    words = claim.split()
    word_index = random.randint(0, len(words) - 1)
    char_index = random.randint(0, len(words[word_index]) - 1)

    typo_word = words[word_index][:char_index] + random.choice('abcdefghijklmnopqrstuvwxyz') + words[word_index][char_index + 1:]
    words[word_index] = typo_word

    transformed_claim = ' '.join(words)
    return transformed_claim

# Helper function
def nltk2wordnet(tag):
    if tag.startswith('N'):
        return 'n'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('R'):
        return 'r'
    elif tag.startswith('J'):
        return 'a'
    return None

# Full code for processing the dataset
input_file = "/Users/arhamazhary/Desktop/Big Project/data.csv"
output_files = {
    "synonyms": "output_synonyms.csv",
    "antonyms": "output_antonyms.csv",
    "negation": "output_negation.csv",
    "swap_adjacent_words": "output_swap_adjacent_words.csv",
    "insert_redundant_phrase": "output_insert_redundant_phrase.csv",
    "typo": "output_typo.csv",
}

transformations = {
    "synonyms": transform_synonyms,
    "antonyms": transform_antonyms,
    "negation": transform_negation,
    "swap_adjacent_words": swap_adjacent_words,
    "insert_redundant_phrase": insert_redundant_phrase,
    "typo": transform_typo,
}

for key, output_file in output_files.items():
    with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            transformed_claim = transformations[key](row["claim"])
            row["claim"] = transformed_claim

            if key == "antonyms" or key == "negation":
                if row["label"] == "SUPPORTS":
                    row["label"] = "REFUTES"
                elif row["label"] == "REFUTES":
                    row["label"] = "SUPPORTS"

            writer.writerow(row)