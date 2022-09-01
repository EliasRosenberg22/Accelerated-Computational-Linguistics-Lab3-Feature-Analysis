#Author: Elias Rosenberg
#Date: April 19th, 2021
#Purpose: A program that takes in star wars scripts and returns the prevailing emotion of Darth Vader in each movie by tokenizing
#his lines and comparing the lines to a sentiment corpus (NRC).
#Inputs: Movie scripts to get Vader's words (sw4-6.txt)
#Outputs: Title of each movie plus Vader's prevailing emotion in that movie.

import nltk
from nltk.tokenize import *


def get_name(file):
    name = ""
    if file == "sw4.txt":
        name = "Episode 4: A New Hope"
    if file == "sw5.txt":
        name = "Episode 5: The Empire Strikes Back"
    if file == "sw6.txt":
        name = "Episode 6: Return of the Jedi"
    return name


def parse_script(file):

    test_file = open(file)
    str = ""
    for line in test_file: #converting the text file to a string so the nltk methods will accept it
        str += line

    sentences = sent_tokenize(str) #tokenizing the script by sentences

    vaderLines = ""
    for sentence in sentences: #finding Vader's lines
        if "VADER" in sentence:
            vaderLines += sentence

    words = word_tokenize(vaderLines)
    #print(words)
    #____________________________________________________________________________________________________


    NRC = open("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt") #loading in the emotions lexicon
    emotions = ""
    for line in NRC:
        emotions += line

    emotionLines = word_tokenize(emotions)
    #print(emotionLines)

    word_dict = {}  # create the dictionary of words to emotions and values
    i = 0  # start counter at index 0
    for i in range(len(emotionLines)-2):  # go until end of list
        word = emotionLines[i]  # word is the first value in the groups of 3
        #print(word)
        emotion = emotionLines[i+1]  # emotion is the second value (i+1)
        #print(emotion)
        value = emotionLines[i+2]  # value is the third value (i+2)
        #print(value)

        if word not in word_dict.keys():  # check if we've added this word to the dictionary and if not
            emotion_dict = {}  # create an emotion dictionary to hold the emotional values for this word
            word_dict[word] = emotion_dict  # add word to the word dictionary with our new emotion dictionary

        word_dict[word][emotion] = value  # add this emotion and value to the emotion dictionary for this word
        i += 3   # move on to next set of values (skip 3)

    key_list = []
    for key in word_dict.keys(): # deleting keys that don't fit within the dictionary format. It literally doesn't work for one word. I blame the NRC :(
        if len(word_dict[key]) is not 10:
            key_list.append(key)
    for k in key_list:
        del word_dict[k]

    emotionsDict = {"positive": 0, "negative": 0, "fear": 0, "anger": 0, "trust": 0, "sadness": 0, "disgust": 0, #all the possible emotions with started values of 0
                    "anticipation": 0, "joy": 0, "surprise": 0}

    for w in words:
        if w in word_dict:
            for e in word_dict[w].keys(): #searching through the emotions dict to get all the values and add them up
                v = int(word_dict[w][e])
                emotionsDict[e] = int(emotionsDict[e])
                emotionsDict[e] += v
 #_________________________________________________________________________________________________________________
    max = 0  #making a fancy print statement
    max2 = 0
    emotion = ""
    emotion2 = ""
    print(emotionsDict)
    for key in emotionsDict.keys():
        value = emotionsDict[key]
        if value >= max:
            max = value
            emotion = key

    for key in emotionsDict.keys():
        value = emotionsDict[key]
        if value >= max2 and key != "negative" and key != "positive":
            max2 = value
            emotion2 = key

    name = get_name(file)


    print ("Vader's emotion in " + name + " was " + emotion + " at" )
    print(max)
    print("followed by " + emotion2 + " at")
    print(max2)


if __name__ == '__main__':
    parse_script("sw4.txt")
    parse_script("sw5.txt")
    parse_script("sw6.txt")
