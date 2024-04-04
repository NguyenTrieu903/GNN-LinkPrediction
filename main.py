from LogisticRegression_Linkprediction.model.link_prediction import link_prediction_with_logistic
from SEAL import main_test
import warnings
warnings.filterwarnings("ignore")
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Link prediction with GNN.")
    parser.add_argument("--model", type=str, help="model name.", default="SEAL")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.model == 'Logistic':
        link_prediction_with_logistic()
    if args.model == 'SEAL':
        main_test.seal_for_link_predict()
    if args.model == 'GNN':
        pass