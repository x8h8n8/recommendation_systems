# date: 2018/5/13
# creator: Xu haonan
# CF recommendation System

import math
import numpy as np
import pandas as pd
import operator

def load_data():
    movies_path = "./ml-1m/movies.dat"
    ratings_path = "./ml-1m/ratings.dat"
    users_path = "./ml-1m/users.dat"
    movie_columns = ["movie_id", "name", "genre"]
    rating_columns = ["user_id", "movie_id", "rating", "timestamp"]
    user_columns = ["user_id", "gender", "age", "occupation", "zipcode"]
    movies = pd.read_table(movies_path, sep="::", header=None, names=movie_columns)
    ratings = pd.read_table(ratings_path, sep="::", header=None, names=rating_columns)
    users = pd.read_table(users_path, sep="::", header=None, names=user_columns)
    data = pd.merge(pd.merge(ratings, users), movies)
    user_movie = pd.DataFrame(data=data, columns=["user_id", "movie_id", "rating"])
    user_movie.to_csv("./ml-1m/user_movie.csv")
    # print(user_movie.head())

# load_data()

def CFmatrix():
    user_movie = pd.read_csv("./ml-1m/user_movie.csv", dtype=np.int, index_col=0)
    users = set(list(user_movie["user_id"].values))
    movies = set(list(user_movie["movie_id"].values))
    CF_matrix = pd.DataFrame(index=users, columns=movies)

    # fill the CF_matrix
    for i in user_movie.values:
        print(i[2])
        CF_matrix.at[i[0], i[1]] = i[2]

    CF_matrix.to_csv("./ml-1m/CFmatrix.csv")

# CFmatrix()

def user_CF(user_id):
    # function: given a user by user_id, return the similarity users dictionary of it
    CF_matrix = pd.read_csv("./ml-1m/CFmatrix.csv", index_col=0).values

    similarity_list = {}
    for i in range(CF_matrix.shape[0]):
        # similarity_sum : add the difference between users
        # count: the number of movies which both two users have seen
        similarity_sum = 0
        count = 0
        if i != user_id:
            for j in range(CF_matrix.shape[1]):
                if CF_matrix[user_id][j] == CF_matrix[user_id][j] and CF_matrix[i][j] == CF_matrix[i][j]:
                    count += 1
                    similarity_sum += abs(CF_matrix[user_id][j] - CF_matrix[i][j])

        # if the number of movies both users have seen is bigger than 20, we save this user and difference
        # into a dictionary
        if count >= 20:
            similarity_list.setdefault(i, [count, similarity_sum])

    for i in similarity_list.keys():
        similarity = similarity_list[i][1]/similarity_list[i][0] if similarity_list[i][0] > 0 else 100
        similarity_list[i].append(similarity)

    sorted_similarity = sorted(similarity_list.items(), key=lambda item:item[1][2])

    return user_id, sorted_similarity

user_id, user_similarity = user_CF(100)




