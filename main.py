import sys
import pickle
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import re
import math
from collections import Counter
from naive_bayes_spam_filter import process_text
sys.path


def main():
    actual_message = input('Enter the message:')
    modify = input("Do you want to replace the words with appropriate synonyms (Y/N):")
    modify = modify.lower()
    actual_message_arr = actual_message.split(' ')
    changed_message = actual_message_arr
    result = None
    predictions = []
    if modify == 'y':
        synonym_dict, largest_syn_arr = get_synonym_dict(actual_message_arr)
        print(synonym_dict)
        new_msg_arr = construct_message(actual_message_arr, synonym_dict, largest_syn_arr)
        cos_sim = get_similarities(actual_message, new_msg_arr)
        print(cos_sim)
        for msg in new_msg_arr:
            print(msg)
            result = predict(msg)
            print(result)
            predictions.append(result[0])
        print(predictions)
    elif modify == 'n':
        message = ' '.join(changed_message)
        result = predict(message)
    else:
        print('incorrect choice')
        sys.exit(1)
    print(result)
    sys.exit(0)


def get_synonym_dict(act_msg_arr):
    synonym_dict = {}
    prev_syn_len = 0
    for word in act_msg_arr:
        if len(word) > 2 and word.lower() not in stopwords.words('english'):
            synonyms = get_synonyms(word)
            new_syn_list = []
            for syn in synonyms:
                if syn not in new_syn_list and word.lower() != syn.lower():
                    new_syn_list.append(syn)
            if len(new_syn_list) > prev_syn_len:
                prev_syn_len = len(new_syn_list)
            if word not in synonym_dict.keys():
                synonym_dict[word] = new_syn_list
    return synonym_dict, prev_syn_len


def construct_message(act_msg_arr, syn_dict, rng):
    new_messages = []
    words = syn_dict.keys()
    original_arr = act_msg_arr
    new_msg = []
    for i in range(rng):
        new_msg += original_arr
        for word in words:
            syn_arr = syn_dict[word]
            if len(syn_arr) > 0:
                if len(syn_arr) > i:
                    new_msg[new_msg.index(word)] = syn_arr[i]
                else:
                    new_msg[new_msg.index(word)] = syn_arr[-1]
        new_messages.append(' '.join(new_msg))
        new_msg = []
    return new_messages


def get_similarities(act_msg, new_msg_arr):
    act_msg_vec = text_to_vector(act_msg)
    cos_sim_arr = []
    for new_msg in new_msg_arr:
        new_msg_vec = text_to_vector(new_msg)
        cos_sim_arr.append(get_cosine(act_msg_vec, new_msg_vec))
    return cos_sim_arr


def modify_message(act_msg_arr, syn_index):
    changed_message = act_msg_arr
    for i, word in enumerate(act_msg_arr):
        if len(word) > 2 and word.lower() not in stopwords.words('english'):
            synonyms = get_synonyms(word)
            if len(synonyms) > 0:
                changed_message[i] = synonyms[syn_index]
    return ' '.join(changed_message)


def predict(message):
    f = open('my_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier.predict([message])


def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    WORD = re.compile(r'\w+')
    words = WORD.findall(text)
    return Counter(words)


if __name__ == '__main__':
    main()
