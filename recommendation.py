def get_unseen(ratings, movies, userId):
    seen_movies = ratings[ratings['userId'] == userId]['movieId'].tolist()
    total_movies = movies['movieId'].tolist()
    unseen_movies = [movie for movie in total_movies if movie not in seen_movies]
    return unseen_movies

def recommend_movie(algo, movies, userId, unseen_movies, top_n=10):
    predictions = [algo.predict(str(userId), str(movieId)) for movieId in unseen_movies]
    def sortkey_est(pred):
        return pred.est
    predictions.sort(key=sortkey_est, reverse=True)
    top_predictions = predictions[:top_n]
    top_movie_ids = [int(pred.iid) for pred in top_predictions]
    top_movie_rating = [pred.est for pred in top_predictions]
    top_movie_titles = movies[movies.movieId.isin(top_movie_ids)]['title']
    top_movie_preds = [(id, title, rating) for id, title, rating in zip(top_movie_ids, top_movie_titles, top_movie_rating)]
    return top_movie_preds