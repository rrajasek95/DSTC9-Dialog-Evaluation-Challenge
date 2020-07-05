import argparse

from config import Config
from nltk import sent_tokenize
from predictors.svm_predictor import SVMPredictor
from tqdm import tqdm
def augment_split(split):
    cfg = Config.from_json('models/Model.SVM/meta.json')


    tagger = SVMPredictor(cfg)

    with open(f'processed_output/{split}.src', 'r') as context_file:
        context_utterances = [line.strip().split("_eos")[:-1] for line in context_file]

    with open(f'processed_output/{split}.src.da', 'r') as context_da_file:
        context_das = [line.strip().split("_eos") for line in context_da_file]

    with open(f'processed_output/{split}.tgt', 'r') as response_file:
        responses = [line.strip() for line in response_file]

    """
    We need to construct additional files as follows:
    1. DA of response
    """

    previous_turn_length = 0

    response_das = []
    for i, (context, response) in tqdm(enumerate(zip(context_utterances, responses))):
        current_turn = len(context)  # 0 indexed.

        if i < len(responses) - 1:
            future_context = context_das[i + 1]
            if len(future_context) - 1 == current_turn:
                # If future context is an extension of previous context,
                # Then the last set of DAs will align with the current turn
                response_das.append(future_context[-1] + '\n')
            else:
                # The next utterance is a new conversation, we cannot extract DA information
                # therefore we *must* predict the response's DA
                turn_das = []
                for sent in sent_tokenize(response):
                    turn_das += [da["communicative_function"] for da in tagger.dialogue_act_tag(sent)]
                response_das.append(" ".join(turn_das) + '\n')

        else:
            # We must predict the response's da
            turn_das = []
            for sent in sent_tokenize(response):
                turn_das += [da["communicative_function"] for da in tagger.dialogue_act_tag(sent)]
            response_das.append(" ".join(turn_das) + '\n')

    with open(f'processed_output/{split}.tgt.da', 'w') as response_da_file:
        response_da_file.writelines(response_das)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default="",
                        help="Split to annotate")
    args = parser.parse_args()
    augment_split(args.split)