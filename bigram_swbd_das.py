import numpy as np


def get_da_counts(filename):
    f = open(filename)
    das = f.readlines()
    f.close()
    das = [each.replace('\n', '') for each in das]
    return [each.split(',') for each in das]


def get_da_to_index(counts):
    da_to_i = {}
    index = 0
    for each in counts:
        if each[0] not in da_to_i:
            da_to_i[each[0]] = index
            index += 1
    return da_to_i


def create_prob_mapping(dtoi, counts):
    prob = np.zeros((len(dtoi), len(dtoi)))
    for each in counts:
        prob[dtoi[each[0]]][dtoi[each[1]]] = each[2]
    prob = prob / prob.sum(axis=1)[:, None]
    return prob


def get_next_das(prob, da, dtoi):
    index = dtoi[da]
    next_da = np.random.choice(list(dtoi.keys()), replace=True, p=prob[index])
    return next_da


def main():
    start_da = "Statement-non-opinion"
    following_seq = []
    path = "tc_processed/SB_V1_Full/das_inter_turn.csv"
    counts = get_da_counts(path)
    dtoi = get_da_to_index(counts)
    prob = create_prob_mapping(dtoi, counts)
    next_state = get_next_das(prob, start_da, dtoi)
    following_seq.append(next_state)
    next_state = get_next_das(prob, next_state, dtoi)
    following_seq.append(next_state)
    print(following_seq)


if __name__ == "__main__":
    main()
