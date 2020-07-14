import json
from createVectors import create_fasttext_model, string_cleaning
import fasttext
from ConvolutionalNetwork import *
import enchant
import re
def main():
    labels = ["AddToPlaylist","BookRestaurant","GetWeather","PlayMusic","RateBook","SearchCreativeWork","SearchScreeningEvent"]
    # If you have existing model comment below line
    create_fasttext_model(labels)
    fasttext_model = fasttext.load_model("model_text_raw.bin")
    train_dataset = create_full_dataset(fasttext_model,labels)
    train_samples, train_targets = toTensor(train_dataset)
    test_dataset = create_full_dataset(fasttext_model,labels, mode="validate")
    test_samples, test_targets = toTensor(test_dataset)
    model = ConvolutionalNetwork()
    print(model)
    # Below hyperparameters tunning
    #hyper_tunning(model,train_samples, train_targets)
    train(model, train_samples, train_targets)
    test(model, test_samples, test_targets)

    print("Stop")



def toVectors(model, label, mode):
    """
    Converts .json files into dictionaries where vectors is Fasttext embedding of a word and label is category
    to which data belongs. Output vectors are 15x15 if sentence, text doesn't have 15 words in it is filled with zeros.
    :param model: network model
    :param label: category to which data belongs
    :param mode: validate or train
    :return: Dictionary as dataset
    """
    type = ""
    dictionary = enchant.Dict("en_US")
    if mode=="train":
        type = "_full"
    with open(mode+"_"+label +type+".json", "r") as read_file:
        data = json.load(read_file)
    data_arr = data[label]
    label_dataset = []
    for d in data_arr:
        vectors = []
        i = 0
        s = ""
        for e in d['data']:
            if i >= 15:
                break
            s = s + str(e['text'])
            i += 1
        s = string_cleaning(s,dictionary)
        s = re.findall(r'\w+',s)
        i = 0
        for e in s:
            if i >= 15:
                break
            vectors.append(model[e])
            i += 1
        if len(vectors) < 15:
            for j in range(15-len(vectors)):
                vectors.append(np.zeros(15))
        dict = {"vectors": vectors, "label": label}
        label_dataset.append(dict)
    return label_dataset

def create_full_dataset(model,labels,mode= "train"):
     """ Merges data from every category into one big dataset. """
     whole_dataset = []
     for label in labels:
         whole_dataset = whole_dataset + toVectors(model, label, mode)
     return whole_dataset

def toTensor(dataset):
    """ Splits dataset into samples and targets and then converts it to tensors which are returned."""
    sample = []
    target = []
    for dativum in dataset:
        if dativum['label'] == "AddToPlaylist":
            target.append(0)
        if dativum['label'] == "BookRestaurant":
            target.append(1)
        if dativum['label'] == "GetWeather":
            target.append(2)
        if dativum['label'] == "PlayMusic":
            target.append(3)
        if dativum['label'] == "RateBook":
            target.append(4)
        if dativum['label'] == "SearchCreativeWork":
            target.append(5)
        if dativum['label'] == "SearchScreeningEvent":
            target.append(6)
        sample.append(dativum['vectors'])
    sample = torch.FloatTensor(sample)
    target = torch.Tensor(target)
    return sample, target




if __name__ == "__main__":
    main()