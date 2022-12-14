from surprise import Reader, Dataset, accuracy, SVD
from surprise.model_selection import GridSearchCV, KFold, cross_validate
from surprise.dataset import DatasetAutoFolds
import pandas as pd
import numpy as np
import recommendation


def svd_recommendation(user_id):
    reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))
    data = Dataset.load_from_file('./ml-latest-small/ratings_noh.csv', reader=reader)
    param_grid = {'n_epochs': [20, 40, 60], 'n_factors': [50, 100, 200]}
    gs = GridSearchCV(SVD, param_grid=param_grid, measures=['rmse'], cv=5)
    gs.fit(data)

    print(f"best score: {gs.best_score['rmse']}")
    print(f"best params: {gs.best_params['rmse']}")

    accuracies = []
    kf = KFold(n_splits=5)
    algo = SVD(n_epochs=20, n_factors=50, random_state=0)

    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        accuracies.append(accuracy.rmse(predictions, verbose=True))
        print(f'accuracy: {accuracy.rmse(predictions, verbose=True)}')
    print(f"mean accuracy: {np.mean(np.array(accuracies))}")

    reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))
    data_folds = DatasetAutoFolds(ratings_file='./ml-latest-small/ratings_noh.csv', reader=reader)
    trainset = data_folds.build_full_trainset()

    algo = SVD(n_epochs=20, n_factors=50, random_state=0)
    algo.fit(trainset)

    movies = pd.read_csv('./ml-latest-small/qualified_movies.csv')
    ratings = pd.read_csv('./ml-latest-small/ratings_small.csv')
    unseen_movies = recommendation.get_unseen(ratings, movies, user_id)
    top_movies_preds = recommendation.recommend_movie(algo, movies, user_id, unseen_movies, top_n=100)
    titles = [row[1] for row in top_movies_preds]
    return titles