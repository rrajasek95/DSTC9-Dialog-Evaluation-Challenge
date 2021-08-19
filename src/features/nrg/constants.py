
def apply_tag_markers(token_list):
    # Applies tag markers when given a list of tokens
    return [f"<{token}>" for token in token_list]


# These are tokens that the model shouldn't condition on. Since the models can generate responses at either a turn or
# sentence level, we have the following special markers:
# <bos>      - beginning of segment : Marks the start of the model input sequence
# <eos>      - end of segment         : Marks the end of the model sequence
# <speaker1> - speaker 1 tag          : Indicates that the tokens that follow are for speaker 1's turn
# <speaker2> - speaker 2 tag          : Indicates that the tokens that follow are for speaker 2's turn
# <end>      - end of sentence marker : Marks the end of sentence in a given turn
# <pad>      - padding token
# <eot>      - end of turn            : token that marks the end of the given turn
SPECIAL_TOKENS = [
    "<bos>", "<eos>", "<speaker1>", "<speaker2>", "<end>", "<pad>", "<eot>"
]

ATTR_TO_SPECIAL_TOKEN = {
    'bos_token': '<bos>',
    'eos_token': '<eos>',
    'pad_token': '<pad>',
    'additional_special_tokens': ["<speaker1>", "<speaker2>", "<end>", "<eot>"]
}

SWBD_ADDITIONAL_TOKENS = apply_tag_markers([
    'affirmative-non-yes-answer', 'open-question', 'statement-non-opinion', 'rhetorical-question', 'wh-question',
    'disagree', 'no-answer', 'conventional-closing', 'backchannel-in-question-form', 'negative-non-no-answer',
    'yes-answer', 'action-directive', 'backchannel', 'conventional-opening', 'thanking', 'agree', 'or-clause',
    'tag-question', 'spoken-artifact', 'apology', 'statement-opinion', 'hedge', 'offers-options-commits', '+',
    'downplayer', 'summarize-reformulate', 'yes-no-question', 'appreciation', 'signal-non-understanding'
]) + ["_fact", "_nofact"]
