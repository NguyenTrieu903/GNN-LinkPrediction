from LogisticRegression_Linkprediction.model.link_prediction import link_prediction_with_logistic
from SEAL.operators.seal_link_predict import execute
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
        execute(0, 0.1, 128, "auto",0.00001)
    if args.model == 'GNN':
        pass