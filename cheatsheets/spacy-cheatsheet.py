import spacy
import numpy as np

some_text = """Google's headquaters are in the USA. I wonder is the weather is sunny in the California.
Have Kevin Mitnick ever worked for Google of FB?"""

nlp = spacy.load('en')
parsed = nlp(some_text)

# Lemmatisation, Part of Speech detection, Entity recognition
for word in parsed:
    print("Word: {}\t| Lemma: {}\t| PoS: {}\t| Entity: {}".format(
        word, word.lemma_, word.pos_, word.ent_type_
        )
    )


# Vectorisation
queen, man, woman, king = nlp("queen man woman king")
print("queen ~ woman: {}\nwoman ~ queen: {}\nwoman ~ man: {}".format(
    queen.similarity(woman), woman.similarity(queen), woman.similarity(man)
    )
)

cosine_sim = lambda v, u: np.dot(v,u) / (np.linalg.norm(v) * np.linalg.norm(u))
print("king ~ queen + man - woman: {}".format(cosine_sim(
    king.vector,
    queen.vector + man.vector - woman.vector))
)

# Find synonyms
all_vocab = [w for w in nlp.vocab if w.orth_.islower()]
def similiar(word, n=5):
    s = sorted(all_vocab, key=lambda w: -word.similarity(w))
    s = list(map(lambda x: x.orth_, s))
    return s[:n]
print("Synonyms to king: ", similiar(king))
