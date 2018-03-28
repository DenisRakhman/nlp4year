import RAKE
import nltk
from collections import Counter

with open('text.txt','r') as f:
    text = f.read().lower()

Rake = RAKE.Rake(nltk.corpus.stopwords.words('english'))
kws = Rake.run(text,maxWords=3)
for kw in kws[:15]:
    print(kw[0])

print('----')

lemmatizer = nltk.stem.WordNetLemmatizer()
lemma_text = ' '.join([lemmatizer.lemmatize(x) for x in nltk.tokenize.word_tokenize(text)])
kws = Rake.run(lemma_text,maxWords=3,minFrequency=2)
for kw in kws[:15]:
    print(kw[0])

print('----')

with open('text_russian.txt','r',encoding='cp1251') as f:
    text = f.read().lower()

Rake = RAKE.Rake(nltk.corpus.stopwords.words('russian'))
kws = Rake.run(text,maxWords=3)
for kw in kws[:15]:
    print(kw[0])
