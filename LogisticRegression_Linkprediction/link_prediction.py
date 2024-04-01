from understanding_data import load_data, plot_graph
from dataset_preparation import retrieve_unconnected, remove_link_connected, data_for_model_training
from feature_extraction import feature_extraction
from build_model import split_data, logistic_regression


def link_prediction_with_logistic():

    fb_df, node_list_1, node_list_2 = load_data()
    G = plot_graph(fb_df)

    data = retrieve_unconnected(node_list_1, node_list_2, G)
    omissible_links_index = remove_link_connected(fb_df, G)
    data, fb_df_ghost = data_for_model_training(fb_df, omissible_links_index, data)

    x = feature_extraction(fb_df, fb_df_ghost, data)

    xtrain, xtest, ytrain, ytest = split_data(data, x)
    logistic_regression(xtrain, xtest, ytrain, ytest)



    
