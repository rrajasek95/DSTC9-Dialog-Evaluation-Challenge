import json


def load_data(path, tag):
    with open(path) as f:
        data = json.load(f)
    conv_keys = data.keys()
    src = []
    da = []
    tgt = []
    for each in conv_keys:
        content = data[each]['content']
        s, t, d = get_content(content, tag)
        src.append(s)
        da.append(d)
        tgt.append(t)
    return src, da, tgt




def get_content(content, tag):
    src = ""
    da = ""
    tgt = ""
    da_label = "label"
    for i in range(len(content)):
        segments = content[i]['segments']
        if tag == "mezza_da":
            da_label = "da"
        das = content[i][tag]
        for j in range(len(segments)):
            if i == 0 and j == 0:
                src = src + segments[j]['text'] + " _end"
                da = da + das[j][da_label]
            elif i == len(content) - 1 and j == len(segments) - 1:
                if tgt == "":
                    tgt = das[j][da_label]
                else:
                    tgt = tgt + ' ' + das[j][da_label]
            else:
                src = src + ' ' + segments[j]['text'] + " _end"
                da = da + ' ' + das[j][da_label]
                if tgt == "":
                    tgt = das[j][da_label]
                else:
                    tgt = tgt + ' ' + das[j][da_label]
        if i != len(content) - 1:
            src += ' _eos'
            da += ' _eos'
            if tgt == "":
                tgt = '_eos'
            else:
                tgt += ' _eos'
    return src, tgt, da


def main():
    path = "../tc_processed/SB_V1_Full/valid_freq_anno_switchboard_v1.json"
    # tag = "switchboard_da"
    tag = "mezza_da"
    src, da, tgt = load_data(path, tag)
    with open("valid_freq.src", 'w') as f:
        for each in src:
            temp = each.replace("\n", "")
            f.write(f"{temp}\n")
    with open("valid_freq.src.mezza", 'w') as f:
        for each in da:
            f.write(f'{each}\n')
    with open("valid_freq.tgt.mezza", 'w') as f:
        for each in tgt:
            f.write(f'{each}\n')


if __name__ == "__main__":
    main()

