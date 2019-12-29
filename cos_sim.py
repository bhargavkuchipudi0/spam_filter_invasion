import math
import re
from collections import Counter


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


def main():
    msg1 = "Text & meet someone sexy today. U can find a date or even flirt its up to U. Join 4 just 10p. REPLY with " \
           "NAME & AGE eg Sam 25. 18 -msg recd@thirtyeight pence "
    msg2 = "Text & meet someone s e x y today. U can find a date or even f l i r t its up to U. Join 4 just 10p. " \
           "REPLY with NAME & AGE eg Sam 25. 18 -msg recd@thirtyeight pence "
    vec_1 = text_to_vector(msg1)
    vec_2 = text_to_vector(msg2)
    cos_sim = get_cosine(vec_1, vec_2)
    print(cos_sim)


if __name__ == '__main__':
    main()
