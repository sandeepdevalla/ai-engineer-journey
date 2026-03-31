import spacy

nlp = spacy.load("en_core_web_sm")

word1 = nlp("king")
word2 = nlp("queen")
word3 = nlp("apple")

print(word1.similarity(word2))
print(word1.similarity(word3))


print("word1.vector[:5]", word1.vector[:5])