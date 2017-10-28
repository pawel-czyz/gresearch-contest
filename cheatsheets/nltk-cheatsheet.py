from nltk.tokenize import sent_tokenize, word_tokenize

some_text = """Hi, nice you are reading this. We are gonna do some simple text operations."""

# Tokenisation
print("Sentence tokenizer: {}".format(sent_tokenize(some_text)))
print("Word tokenizer: {}".format(word_tokenize(some_text)))

# Definition and PoS
from nltk.corpus import wordnet
s = wordnet.synsets('king')[0]
print("Word: king, Name: {}, Definition: {}, Part of Speech: {}".format(s.name(), s.definition(), s.pos()))

# Lemmatisation, synonyms and antonyms
print("Lemmas for a king: ", s.lemmas())
print("Antonyms ", s.lemmas()[0].antonyms())
