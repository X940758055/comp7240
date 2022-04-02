import math
import numpy as np
import pandas as pd
import content_based_function as cbf
import random
 



def content_based_recommend(user_id, movie_feature_df, movie_df, rating_df, k):

    user_ids = [user_id]
    user_rating_df = rating_df[rating_df['USER_ID'] == user_id]
    user_rating_df.reset_index(drop=True)
    if len(user_rating_df) == 0:
        res = []
        for index, row in movie_df.iteritems():
            if random.randint(0,200) == 100:
                res.append(row['MOVIE_ID'])
                if len(res) >= k:
                    return res
    
    
    feature_list = []
    for index, column in movie_feature_df.iteritems():
        feature_list.append(index)
    feature_list = feature_list[1:]
    temp_movie_feature_df = movie_feature_df.copy(deep=True)
    movie_feature_matrix = np.array(temp_movie_feature_df.drop(['MOVIE_ID'], axis=1))
    

    weighted_profile = cbf.build_user_profile(user_id, rating_df, movie_feature_df, feature_list, weighted = True)
    
    
    weighted_rec_result = cbf.generate_recommendation_results(user_id, weighted_profile, movie_feature_df, movie_feature_matrix, movie_df)
    true_rec_list = cbf.get_ground_truth_list(rating_df[rating_df['USER_ID'] == user_id])


    return cbf.get_recommendation_list(weighted_rec_result, k)
    
    