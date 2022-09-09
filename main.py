import sys
import numpy as np
from typing import Tuple

MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #
from model_vars import Model_Vars
from inference import decode_sequence
from inference import modeVarsLoaded

# Load Variables

max_encoder_seq_length = modeVarsLoaded.max_encoder_seq_length
num_encoder_tokens = modeVarsLoaded.num_encoder_tokens
input_token_index = modeVarsLoaded.input_token_index

# Load Model

def predict(factors: str):
    encoder_input_data_val = np.zeros( (1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    for t, char in enumerate(factors):
        encoder_input_data_val[0, t, input_token_index[char]] = 1.0
    encoder_input_data_val[0, t + 1:, input_token_index[" "]] = 1.0
    decoded_sentence = decode_sequence(encoder_input_data_val)
    return decoded_sentence.strip()


# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str):
    factors, expansions = load_file(filepath)
    pred = [predict(f) for f in factors]
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print(np.mean(scores))

if __name__ == "__main__":
    main("test.txt" if "-t" in sys.argv else "train.txt")