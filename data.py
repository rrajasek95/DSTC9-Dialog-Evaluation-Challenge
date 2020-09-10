import json

def load_keys(file_path):
    with open(file_path, 'r') as key_file:
        conv_ids = [conv_id.strip() for conv_id in key_file]

    return conv_ids

def prepare_dstc9_conversations():
    valid_freq_keys = load_keys('tc_processed/valid_freq.keys')

    with open('tc_processed/valid_freq_full_anno.json', 'r') as valid_freq_conv_file:
        valid_freq_conv_data = json.load(valid_freq_conv_file)

    with open('tc_processed/train_full_anno.json', 'r') as train_conv_file:
        train_conv_data = json.load(train_conv_file)
    with open('tc_processed/test_freq_full_anno.json', 'r') as test_freq_conv_file:
        test_freq_conv_data = json.load(test_freq_conv_file)
    with open('tc_processed/test_rare_full_anno.json', 'r') as test_rare_conv_file:
        test_rare_conv_data = json.load(test_rare_conv_file)
    with open('tc_processed/valid_rare_full_anno.json', 'r') as valid_rare_conv_file:
        valid_rare_conv_data = json.load(valid_rare_conv_file)

    valid_freq_conv_data.update(train_conv_data)
    valid_freq_conv_data.update(test_freq_conv_data)
    valid_freq_conv_data.update(test_rare_conv_data)
    valid_freq_conv_data.update(valid_rare_conv_data)
    current_conv_id = None
    current_idx = -1

    context = []
    response = []
    examples = []
    for conv_id in valid_freq_keys:
        if current_conv_id != conv_id:
            current_conv_id = conv_id
            current_idx = 0
            context = []

        if current_idx > 0:
            response.append(context[-1] + "\n")

        text = valid_freq_conv_data[current_conv_id]["content"][current_idx]["message"]

        new_example = context + [text.lower()]

        examples.append(" _eos ".join(new_example) + " _eos\n")
        context = new_example

        current_idx += 1

    with open('valid_freq.src', 'w') as src_file:
        src_file.writelines(examples)

if __name__ == '__main__':
    prepare_dstc9_conversations()