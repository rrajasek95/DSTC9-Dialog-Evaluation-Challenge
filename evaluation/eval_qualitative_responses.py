import argparse
from nltk import RegexpTokenizer

tokenizer = RegexpTokenizer(r"\w+")


def check_repetition(response, n):
    tokens = tokenizer.tokenize(response)
    n_grams = []
    for i in range(len(tokens) - n):
        n_grams.append(" ".join(tokens[i:i+n]))
    return len(n_grams) != len(set(n_grams))


def count_repetition(text_file, n_gram, output_file):
    with open(text_file, 'r') as valid_freq_response:
        responses = [line.strip() for line in valid_freq_response]
    count = 0
    repetitions = []
    for response in responses:
        if check_repetition(response, n_gram):
            count += 1
            repetitions.append(response)
    with open(output_file, 'w') as output:
        for each in repetitions:
            output.writelines(each + "\n")
    print(f"Percentage of responses have repetition: {count / len(responses):.2f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--response_file',
                        type=str,
                        default="submissions/kd_pd_nrg_dstc9_valid_freq_heuristic_swbd.txt",
                        help='File containing output responses.')
    parser.add_argument('--n_gram', type=int, default=3, help='The number of n_grams to check the repetition.')
    parser.add_argument('--output_file', type=str,
                        default='submissions/kd_pd_nrg_dstc9_valid_freq_heuristic_swbd_repetitions.txt',
                        help='Output file to write the repeated utterances.')
    args = parser.parse_args()
    count_repetition(args.response_file, args.n_gram, args.output_file)
