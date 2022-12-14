from surprise import Reader, Dataset, accuracy, KNNWithMeans
from surprise.model_selection import GridSearchCV, KFold
from surprise.dataset import DatasetAutoFolds
import pandas as pd
import numpy as np
import recommendation


def knn_recommendation(user_id):
    reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))
    data = Dataset.load_from_file('./ml-latest-small/ratings_noh.csv', reader=reader)
    sim_options = {
        'name': ['msd', 'cosine'],
        'min_support': [3, 4, 5, 6, 7, 8, 9, 10],
        'user_based': [False, True]
    }

    param_grid = {'sim_options': sim_options}
    gs = GridSearchCV(KNNWithMeans, param_grid=param_grid, measures=['rmse'], cv=5)
    gs.fit(data)

    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])

    sim_options = {
        'name': 'msd',
        'min_support': 3,
        'user_based': False
    }

    accuracies = []
    kf = KFold(n_splits=5)
    algo = KNNWithMeans(sim_options=sim_options)
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        accuracies.append(accuracy.rmse(predictions, verbose=True))
        print(f'accuracy: {accuracy.rmse(predictions, verbose=True)}')
    print(f"mean accuracy: {np.mean(np.array(accuracies))}")

    reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))
    data_folds = DatasetAutoFolds(ratings_file='./ml-latest-small/ratings_noh.csv', reader=reader)
    trainset = data_folds.build_full_trainset()

    algo = KNNWithMeans(sim_options=sim_options)
    algo.fit(trainset)

    movies = pd.read_csv('./ml-latest-small/qualified_movies.csv')
    ratings = pd.read_csv('./ml-latest-small/ratings_small.csv')
    unseen_movies = recommendation.get_unseen(ratings, movies, user_id)
    top_movies_preds = recommendation.recommend_movie(algo, movies, user_id, unseen_movies, top_n=100)
    titles = [row[1] for row in top_movies_preds]
    return titles
