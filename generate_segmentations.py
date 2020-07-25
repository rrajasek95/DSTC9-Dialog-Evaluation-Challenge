import spacy
import json


nlp = spacy.load('en_core_web_sm')
sentencizer = nlp.create_pipe("sentencizer")
def prevent_s_being_start(doc):
    for i, token in enumerate(doc):
        if token.text in ("’s", "'s"):
            doc[i].is_sent_start = False
        elif token.text in ("“", "‘") and i < len(doc) - 1:
            # opening quote
            doc[i+1].is_sent_start = False
        elif token.text in ("”", "’"):
            # closing quote
            doc[i].is_sent_start = False
    return doc

def split_long_sent(doc):
    clause_length = 0
    for i, token in enumerate(doc):
        if clause_length > 20 and token.text == "," and i+1 < len(doc):
            doc[i+1].is_sent_start = True
            clause_length = 0
        else:
            clause_length += 1
    return doc

nlp.add_pipe(sentencizer, first=True)
nlp.add_pipe(prevent_s_being_start, before="parser")
nlp.add_pipe(split_long_sent, before="parser")

files = ['valid_freq', 'valid_rare', 'test_freq', 'test_rare']
for file_name in files:
    path = f"tc_processed/{file_name}_full_anno.json"
    with open(path) as f:
        train = json.load(f)
    keys = list(train.keys())
    j = 0
    for each in keys:
        content = train[each]['content']
        for i in range(len(content)):
            message = content[i]['message'].replace("\n", "")
            doc = nlp(message)
            utterances = list(doc.sents)
            train[each]['content'][i]['segments'] = [{'text': utt.text} for utt in utterances]
        j += 1
        if j % 100 == 0:
            print(j/len(keys))

    with open(f"tc_processed/new_segmentation/{file_name}_new.json", 'w') as f:
        json.dump(train, f)
