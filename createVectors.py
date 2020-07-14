import fasttext
import json
import re
import enchant
def create_text_file(labels):
    """ Takes texts from .jsons concatenates them and saves it as one big text file
        used later in Fasttext model training"""
    whole_text = ""
    dict = enchant.Dict("en_US")
    i = 0
    for label in labels:
        i += 1
        print('Processing [{}]/[{}] of labels dataset.'.format(i,len(labels)))
        with open("train_"+label+ "_full.json", "r") as read_file:
            data = json.load(read_file)
        data_arr = data[label]
        j = 0
        for d in data_arr:
            text = ""
            for e in d['data']:
                text= text + e['text']
            text = text.lower()
            text = string_cleaning(text, dict)
            text = text + " . "
            whole_text = whole_text + text
            j += 1
            if (j + 1) % 100 == 0:
                print("Processing [{}]/[{}] of sentences in current dataset.".format(j,len(data_arr)))
    text_file = open("data_raw.txt", "wt")
    n = text_file.write(whole_text)
    text_file.close()

def create_fasttext_model(labels):
    """Runs Fastettext unsupervised to create a good model based on training set"""
    create_text_file(labels)
    model = fasttext.train_unsupervised('data_raw.txt', model='skipgram', dim=15)
    model.save_model("model_text_raw.bin")

def string_cleaning(string, dict):
    """ Striping from sentences words that are not proper english words
        to reduce noise."""
    string = string.replace('</s>','')
    res = re.findall(r'\w+',string)
    tmp = ""
    for i in range(len(res)):
        if dict.check(res[i]):
           tmp = tmp +' '+ res[i]
    return tmp