from feature_extraction import feature_extraction
import dataset_preparation
import build_model
import understanding_data

def link_prediction_with_logistic():

    fb_df, node_list_1, node_list_2 = understanding_data.load_data()
    G = understanding_data.create_graph(fb_df)

    data = dataset_preparation.retrieve_unconnected(node_list_1, node_list_2, G)
    omissible_links_index = dataset_preparation.remove_link_connected(fb_df, G)
    data, fb_df_ghost = dataset_preparation.data_for_model_training(fb_df, omissible_links_index, data)

    x = feature_extraction(fb_df, fb_df_ghost, data)

    xtrain, xtest, ytrain, ytest = build_model.split_data(data, x)
    build_model.logistic_regression(xtrain, xtest, ytrain, ytest)



    
