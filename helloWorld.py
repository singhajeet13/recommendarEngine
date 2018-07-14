#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:32:52 2018

@author: ajeet
"""

from flask import Flask, flash, redirect, render_template, request, session, abort
from flask import jsonify
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io


app = Flask(__name__)

# Reading ratings file
# Ignore the timestamp column
ratings = pd.read_csv('ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])

# Reading users file
users = pd.read_csv('users.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

# Reading movies file
movies = pd.read_csv('movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])


movies['genres'] = movies['genres'].str.split('|')
# Convert genres to string value
movies['genres'] = movies['genres'].fillna("").astype('str')

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies['genres'])
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[:4, :4]

# Build a 1-dimensional array with movie titles
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

def genre_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    movie_indices.remove(idx)
    return titles.iloc[movie_indices]
#genre_recommendations('Good Will Hunting (1997)').head(5)
    


def printMovies(username=None):
    recommMovies = list(genre_recommendations('Good Will Hunting (1997)').head(5))
    print(recommMovies)
    
    print("People who watched {0} also watched ---!".format(username))
    
@app.route('/movie-recomm/<string:username>', methods=['GET', 'POST'])
def hello_world(username=None):
    recommMovies = list(genre_recommendations(username).head(5))
    return('''<h1>People who watched {0} also watched ---</h1>
           <h2>1. {1} </h2>
           <h2>2. {2} </h2>
           <h2>3. {3} </h2>
           <h2>4. {4} </h2>
           <h2>5. {5} </h2>
           '''.format(username,recommMovies[0],recommMovies[1],recommMovies[2],recommMovies[3],recommMovies[4]))

@app.route('/movieName/<string:name>')
def movieName(name=None):   
    return render_template(
        'movies.html',name=name)