#Author: Elias Rosenberg
#Date: April 19th, 2021
#Purpose: find word-relationship biases in a language other than english using Word2vec
#input: Premade language word2vec from github. I will be using French ('fr.bin')
#output: Outputs the "top 25" responses for multiple equations (see instructions), as well as a scatter plot for 7
#specific words in the chosen language.



# =======================================================================
# Dartmouth College, LING48/CS72, Spring 2021
# Rolando Coto-Solano (Rolando.A.Coto.Solano@dartmouth.edu)
# Examples for Homework 3.3: Word2Vec
#
# This Python script reads one file, the es.bin file for a pre-trained
# word2vec. .bin files for other languages can be downloaded here:
# https://github.com/Kyubyong/wordvectors
# This example uses the Catalan word2vec.
#
# You can learn more about the details of the code below here:
# https://samyzaf.com/ML/nlp/nlp.html
#
# I got the visualization code from here:
# https://stackoverflow.com/questions/43776572/visualise-word2vec-generated-from-gensim
#
# You can learn more about word2vec here:
# http://jalammar.github.io/illustrated-word2vec/
# =======================================================================

# Import libraries to run word2vec and make charts
import pickle

import matplotlib
from gensim.models import Word2Vec


# Disable warnings. This is not usually recommended, but they are putting out
# an update to the word2vec libraries, and it's throwing out some warnings that
# are not relevant to what we're doing. If you want to see the warnings,
# comment the four lines below and re-run the code.
import sys

from jedi.api.refactoring import inline

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

# Get all of the sentences in the Catalan .bin file.
# This file contains a pretrained word2vec model.

model = Word2Vec.load('fr.bin')
print("Word2Vec downloaded...")

#______________________________________________________________________________________________________________________

#part 3: trouvant les mots

#finding the top 25 words related to "man" and "woman" in french
xy = model.most_similar(positive='homme', topn=25)
xx = model.most_similar(positive='femme', topn=25)
print("top 25 words similar to men:\n")
print(xy)
print("\n")
print("top 25 words similar to women:\n")
print(xx)
print("\n")

#“king-man+woman”
print("results of equation 'king-man+woman':\n")
algebra = model.most_similar(positive=['roi', 'femme'], negative=['homme'], topn=25)
print(algebra)
print("\n")

#man+homme and woman+home"
homeMan = model.most_similar(positive=['maison', 'homme'], topn=25)
homeWoman = model.most_similar(positive=['maison', 'femme'], topn=25)

print("results of equation 'home + man': \n")
print(homeMan)
print("\n")
print("results of equation 'home + woman': \n")
print(homeWoman)
print("\n")


#_____________________________________________________________________________________________________________________
#part 6- visualization of [ 'man', 'woman','king','queen','child','boy','girl' ]

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd


list = ['homme', 'femme', 'roi', 'reine', 'enfant', 'garçon', 'fille'] #list of requested words

vocab = list
X = model[vocab]
#X = vocabDictionary
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X) #shrinking down the vector numbers to a usable dimension

df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y']) #setting up the panda object

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos)

plt.show()
