import pickle
from itertools import chain
import random

from torch.utils.data import Dataset

from pd_nrg.policies import KnowledgeDependent, KnowledgeIndependentSWBDPolicy


from pd_nrg.ranker import TfIdfRankerRetriever
from pd_nrg.ranker import EmbRankerRetriever


class TopicalChatsDataset(Dataset):
    """
    It's absolutely necessary to create a dataset class since
    the amount of data is huge.
    I wonder if there are other optimization opportunities
    - Rishi
    """

    def __init__(self, dataset, tokenizer, special_tokens, args):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens

        # Args to control memory footprint
        self.max_history = args.max_history
        self.num_candidates = args.num_candidates
        self.max_fact_length = args.max_fact_length

    def __getitem__(self, index):
        # For the baseline implementation, we don't need to consider the DA
        (history, (response, _, fact)) = self.dataset[index]

        # h[0] contains the response
        history = [h[0] for h in history]
        history, fact = self.truncate_sequences(history, fact)

        candidates = self.sample_candidates(self.dataset, index)
        candidates.append(response)

        instances = []
        for j, candidate in enumerate(candidates):
            lm_labels = bool(j == self.num_candidates - 1)
            instance = self.build_input_from_segments(history, candidate, fact, self.tokenizer, lm_labels)
            instances.append(instance)
        return instances

    def __len__(self):
        return len(self.dataset)

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

    def sample_candidates(self, dataset, current_conversation_index):
        # Lets just hope that the number of cases where the true responses gets included in the
        # candidates is vanishingly small
        candidates = [response for (_, (response, _, _)) in random.sample(dataset, self.num_candidates - 1)]

        return candidates

    def build_input_from_segments(self, history, response, fact, tokenizer, lm_labels=False):
        bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids((self.special_tokens[:-1]))

        """
        Input construction (may change):
        <bos> FACT <speaker1/2> UTT1 <speaker1/2> ... <speaker2> RESPONSE
        Considerations for design:
        1. Topical chat examples are created by adding a response every turn
        2. Last turn is always speaker2
        To my knowledge, the position of the fact in input is mostly immaterial due to
        the self-attention mechanism (since all tokens are equidistant). The positional
        embeddings affect only the contextual representation (I think!)
          - Rishi
        """
        sequence = [[bos] + fact] + history + [response + [eos]]

        sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in
                                    enumerate(sequence[1:])]

        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        """
        Explanation:
        lm_labels is token-wise mask that is used to compute language modeling loss 
        We want the language modeling loss to propagate only when we generate
        incorrectly on the true response and not on the distractor responses
        """
        instance["lm_labels"] = [-100] * len(instance["input_ids"])
        if lm_labels:
            instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
        return instance

    def truncate_sequences(self, history, fact):
        # Truncate history turns to reduce memory requirement
        if len(history) > (2 * self.max_history + 1):
            history = history[-(2 * self.max_history + 1):]

        # Truncate facts to decrease overall input length
        trunc_facts = fact[:min(len(fact), self.max_fact_length)]
        return history, trunc_facts

class TopicalChatsDatasetSent(Dataset):
    """
    It's absolutely necessary to create a dataset class since
    the amount of data is huge.
    I wonder if there are other optimization opportunities
    - Rishi
    """

    def __init__(self, dataset, tokenizer, special_tokens, args):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens

        # Args to control memory footprint
        self.max_history = args.max_history
        self.num_candidates = args.num_candidates
        self.max_fact_length = args.max_fact_length

    def __getitem__(self, index):
        # For the baseline implementation, we don't need to consider the DA
        (history, (response, _, fact)) = self.dataset[index]

        # h[0] contains the response
        history = [h[0] for h in history]
        history, fact = self.truncate_sequences(history, fact)

        candidates = self.sample_candidates(self.dataset, index)
        candidates.append(response)

        instances = []
        for j, candidate in enumerate(candidates):
            lm_labels = bool(j == self.num_candidates - 1)
            instance = self.build_input_from_segments(history, candidate, fact, self.tokenizer, lm_labels)
            instances.append(instance)
        return instances

    def __len__(self):
        return len(self.dataset)

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

    def sample_candidates(self, dataset, current_conversation_index):
        # Lets just hope that the number of cases where the true responses gets included in the
        # candidates is vanishingly small
        candidates = [response for (_, (response, _, _)) in random.sample(dataset, self.num_candidates - 1)]

        return candidates

    def build_input_from_segments(self, history, response, fact, tokenizer, lm_labels=False):
        bos, eos, end, speaker1, speaker2 = tokenizer.convert_tokens_to_ids((self.special_tokens[:-2]))
        eot = tokenizer.convert_tokens_to_ids((self.special_tokens[-1]))
        """
        Input construction (may change):
        <bos> FACT <speaker1/2> UTT1 <speaker1/2> ... <speaker2> RESPONSE
        Considerations for design:
        1. Topical chat examples are created by adding a response every turn
        2. Last turn is always speaker2
        To my knowledge, the position of the fact in input is mostly immaterial due to
        the self-attention mechanism (since all tokens are equidistant). The positional
        embeddings affect only the contextual representation (I think!)
          - Rishi
        """
        is_new_turn = len(history) == 0 or history[-1][0] == eot
        if is_new_turn:
            history = history[:-1] # Skip EOT marker

        segmented_history = []
        for i, history_turn in enumerate(history):
            # interleave end of sentence markers between segments
            segments = list(chain.from_iterable(
                [turn_segment + [end] for turn_segment in history_turn[:-1]] + [history_turn[-1]]
            ))

            segmented_history.append(segments)

        sequence = [[bos] + fact] + segmented_history + [response + [eos]]
        if is_new_turn:
            sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in
                                        enumerate(sequence[1:])]
        # if the generated response is still continuing the previous sentence, do not add the speaker token
        # and add a <end> token to the last of previous sentence
        else:
            sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in
                                        enumerate(sequence[1:-1])] + [sequence[-1]]
            sequence[-2] += [end]

        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        if is_new_turn:
            instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
        else:
            # The token type ids for the response segment must match the preceding segment
            # for turn continuation since they belong to the same speaker
            instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[:-1]) for _ in s] \
                + [speaker2 if (len(sequence) - 2) % 2 else speaker1 for _ in sequence[-1]]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        """
        Explanation:
        lm_labels is token-wise mask that is used to compute language modeling loss 
        We want the language modeling loss to propagate only when we generate
        incorrectly on the true response and not on the distractor responses
        """
        instance["lm_labels"] = [-100] * len(instance["input_ids"])
        if lm_labels:
            instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
        return instance

    def truncate_sequences(self, history, fact):
        # Truncate history turns to reduce memory requirement
        if len(history) > (2 * self.max_history + 2) and history[-1][0] == self.tokenizer.encode("<eot>"):
            history = history[-(2 * self.max_history + 2):]
        elif len(history) > (2 * self.max_history + 1):
            history = history[-(2 * self.max_history + 1):]


        # Truncate facts to decrease overall input length
        trunc_facts = fact[:min(len(fact), self.max_fact_length)]
        return history, trunc_facts


class TopicalChatsKDDataset(TopicalChatsDataset):
    def _init_knowledge_index(self, knowledge_index_path, knowledge_policy):
        with open(knowledge_index_path, 'rb') as knowledge_index_file:
            index_data = pickle.load(knowledge_index_file)
        if knowledge_policy == "tf_idf":
            self.ranker_retriever = TfIdfRankerRetriever(index_data)
        else:
            self.ranker_retriever = EmbRankerRetriever(index_data)

    def __init__(self, dataset, tokenizer, special_tokens, args, inference=False):
        self.dialog_policy = KnowledgeDependent()

        # For inference, the model will start executing the
        # heuristic dialog policy and knowledge selection policy
        self.inference = inference
        if self.inference:
            self._init_knowledge_index(args.knowledge_index_path, args.knowledge_policy)
        self.dataset_configuration = args.dataset_configuration
        super().__init__(dataset, tokenizer, special_tokens, args)

    def sample_candidates(self, dataset, current_conversation_index):
        # Lets just hope that the number of cases where the true responses gets included in the
        # candidates is vanishingly small
        candidates = [response for (_, (response, _, _)) in random.sample(dataset, self.num_candidates - 1)]

        return candidates

    def _construct_dialog_state(self, history):
        turn_history = []
        da_history = []
        knowledge_history = [""]  # Hack to always have empty
        for (response, das, past_knowledge) in history:
            turn_history.append(response)
            da_history += das
            if self.inference:
                # Knowledge history only matters during inference
                # this also optimizes running an unnecessary decode
                # during training
                knowledge_history.append(self.tokenizer.decode(past_knowledge))

        dialog_state = {
            "turn_history": turn_history,
            "da_history": da_history,
            "knowledge_history": knowledge_history
        }

        return dialog_state

    def _select_appropriate_knowledge(self, dialog_state):
        turn_history = dialog_state["turn_history"]
        if len(turn_history) == 0:
            return ""
        else:
            last_turn = self.tokenizer.decode(turn_history[-1])
            knowledge, similarity = self.ranker_retriever.get_top_n(last_turn, n=1)[0]

            if similarity > 0.2:
                return knowledge
            else:
                return ""

    def _execute_heuristic_policy(self, dialog_state):
        knowledge = self._select_appropriate_knowledge(dialog_state)
        dialog_state["knowledge"] = knowledge  # Augment dialog state with knowledge
        das, knowledge = self.dialog_policy.get_knowledge_grounded_action(dialog_state)
        return das, self.tokenizer.encode(knowledge)

    def __getitem__(self, index):
        (history, (response, mezza_das, knowledge)) = self.dataset[index]

        dialog_state = self._construct_dialog_state(history)
        das_to_return = []
        if self.inference:
            """
            During inference time, there is no ground truth utterance to 
            choose the appropriate knowledge on. So we use a heuristic policy 
            to "predict" the best knowledge and dialogue act to use for the next turn.
            """
            mezza_das, knowledge = self._execute_heuristic_policy(dialog_state)
            das_to_return = [f"<{da}>" for da in mezza_das]
            mezza_das = self.tokenizer.encode([f"<{da}>" for da in mezza_das])
        history, fact = self.truncate_sequences(dialog_state["turn_history"], knowledge)

        candidates = self.sample_candidates(self.dataset, index)
        candidates.append(response)
        if self.dataset_configuration != "dstc9":
            encoded_das = self.tokenizer.encode([f"<{da['da']}>" for da in mezza_das])
        else:
            encoded_das = mezza_das
        instances = []

        # The action plan must be ground-truth for training and validation
        # However, for inference time, it must follow the policy
        uses_fact = self.tokenizer.encode("_nofact" if len(knowledge) <= 1 else "_fact")
        action_plan = encoded_das + fact + uses_fact
        for j, candidate in enumerate(candidates):
            lm_labels = bool(j == self.num_candidates - 1)
            instance = self.build_input_from_segments(history, candidate, action_plan, self.tokenizer, lm_labels)
            instance['das_to_return'] = das_to_return
            instances.append(instance)

        return instances

class TopicalChatsKDSentDataset(TopicalChatsDatasetSent):
    def _init_knowledge_index(self, knowledge_index_path, knowledge_policy):
        with open(knowledge_index_path, 'rb') as knowledge_index_file:
            index_data = pickle.load(knowledge_index_file)
        if knowledge_policy == "tf_idf":
            self.ranker_retriever = TfIdfRankerRetriever(index_data)
        else:
            self.ranker_retriever = EmbRankerRetriever(index_data)

    def __init__(self, dataset, tokenizer, special_tokens, args, inference=False):
        self.dialog_policy = KnowledgeDependent()

        # For inference, the model will start executing the
        # heuristic dialog policy and knowledge selection policy
        self.inference = inference
        if self.inference:
            self._init_knowledge_index(args.knowledge_index_path, args.knowledge_policy)
        self.dataset_configuration = args.dataset_configuration
        super().__init__(dataset, tokenizer, special_tokens, args)

    def sample_candidates(self, dataset, current_conversation_index):
        # Lets just hope that the number of cases where the true responses gets included in the
        # candidates is vanishingly small
        candidates = [response for (_, (response, _, _)) in random.sample(dataset, self.num_candidates - 1)]

        return candidates

    def _construct_dialog_state(self, history):
        turn_history = []
        da_history = []
        knowledge_history = [""]  # Hack to always have empty
        for (response, das, past_knowledge) in history:
            turn_history.append(response)
            da_history += das
            if self.inference:
                # Knowledge history only matters during inference
                # this also optimizes running an unnecessary decode
                # during training
                knowledge_history.append(self.tokenizer.decode(past_knowledge))

        dialog_state = {
            "turn_history": turn_history,
            "da_history": da_history,
            "knowledge_history": knowledge_history
        }

        return dialog_state

    def _select_appropriate_knowledge(self, dialog_state):
        turn_history = dialog_state["turn_history"]
        if len(turn_history) == 0:
            return ""
        else:
            last_turn = self.tokenizer.decode(turn_history[-1])
            knowledge, similarity = self.ranker_retriever.get_top_n(last_turn, n=1)[0]

            if similarity > 0.2:
                return knowledge
            else:
                return ""

    def _execute_heuristic_policy(self, dialog_state):
        knowledge = self._select_appropriate_knowledge(dialog_state)
        dialog_state["knowledge"] = knowledge  # Augment dialog state with knowledge
        das, knowledge = self.dialog_policy.get_knowledge_grounded_action(dialog_state)
        return das, self.tokenizer.encode(knowledge)

    def __getitem__(self, index):
        (history, (response, mezza_das, knowledge)) = self.dataset[index]

        dialog_state = self._construct_dialog_state(history)
        das_to_return = []
        if self.inference:
            """
            During inference time, there is no ground truth utterance to 
            choose the appropriate knowledge on. So we use a heuristic policy 
            to "predict" the best knowledge and dialogue act to use for the next turn.
            """
            mezza_das, knowledge = self._execute_heuristic_policy(dialog_state)
            das_to_return = [f"<{da}>" for da in mezza_das]
            mezza_das = self.tokenizer.encode([f"<{da}>" for da in mezza_das])
        history, fact = self.truncate_sequences(dialog_state["turn_history"], knowledge)

        candidates = self.sample_candidates(self.dataset, index)
        candidates.append(response)
        if self.dataset_configuration != "dstc9":
            encoded_das = self.tokenizer.encode([f"<{da['da']}>" for da in mezza_das])
        else:
            encoded_das = mezza_das
        instances = []

        # The action plan must be ground-truth for training and validation
        # However, for inference time, it must follow the policy
        uses_fact = self.tokenizer.encode("_nofact" if len(knowledge) <= 1 else "_fact")
        action_plan = encoded_das + fact + uses_fact
        for j, candidate in enumerate(candidates):
            lm_labels = bool(j == self.num_candidates - 1)
            instance = self.build_input_from_segments(history, candidate, action_plan, self.tokenizer, lm_labels)
            instance['das_to_return'] = das_to_return
            instances.append(instance)

        return instances


class TopicalChatsSWBDDataset(TopicalChatsDataset):

    def __init__(self, dataset, tokenizer, special_tokens, args, inference=False):
        self.dialog_policy = KnowledgeIndependentSWBDPolicy()

        # For inference, the model will start executing the
        # heuristic dialog policy
        self.inference = inference
        self.dataset_configuration = args.dataset_configuration
        super().__init__(dataset, tokenizer, special_tokens, args)

    def sample_candidates(self, dataset, current_conversation_index):
        # Lets just hope that the number of cases where the true responses gets included in the
        # candidates is vanishingly small
        candidates = [response for (_, (response, _, _)) in random.sample(dataset, self.num_candidates - 1)]

        return candidates

    def _construct_dialog_state(self, history):
        turn_history = []
        da_history = []
        knowledge_history = [""]  # Hack to always have empty
        inter_turn = True
        for (response, das, past_knowledge) in history:
            turn_history.append(response)
            da_history += das
            if self.inference:
                # Knowledge history only matters during inference
                # this also optimizes running an unnecessary decode
                # during training
                knowledge_history.append(self.tokenizer.decode(past_knowledge))

        dialog_state = {
            "turn_history": turn_history,
            "da_history": da_history,
            "knowledge_history": knowledge_history,
            "inter_turn": inter_turn
        }

        return dialog_state

    def _select_appropriate_knowledge(self, dialog_state):
        turn_history = dialog_state["turn_history"]
        if len(turn_history) == 0:
            return ""
        else:
            last_turn = self.tokenizer.decode(turn_history[-1])
            knowledge, similarity = self.ranker_retriever.get_top_n(last_turn, n=1)[0]

            if similarity > 0.2:
                return knowledge
            else:
                return ""

    def _execute_heuristic_policy(self, dialog_state):
        das = self.dialog_policy.get_action(dialog_state)
        return das

    def __getitem__(self, index):
        (history, (response, mezza_das, knowledge)) = self.dataset[index]

        dialog_state = self._construct_dialog_state(history)
        das_to_return = []

        # h[0] contains the response
        history = [h[0] for h in history]
        history, fact = self.truncate_sequences(history, knowledge)

        if self.inference:
            """
            During inference time, there is no ground truth utterance to 
            choose the appropriate knowledge on. So we use a heuristic policy 
            to "predict" the best knowledge and dialogue act to use for the next turn.
            """

            mezza_das = self._execute_heuristic_policy(dialog_state)
            das_to_return = [f"<{da}>" for da in mezza_das]
            mezza_das = self.tokenizer.encode([f"<{da}>" for da in mezza_das])

        candidates = self.sample_candidates(self.dataset, index)
        candidates.append(response)
        if self.dataset_configuration != "dstc9":
            encoded_das = self.tokenizer.encode([f"<{da['da']}>" for da in mezza_das])
        else:
            encoded_das = mezza_das
        instances = []

        # The action plan must be ground-truth for training and validation
        # However, for inference time, it must follow the policy
        uses_fact = self.tokenizer.encode("_nofact" if len(knowledge) <= 1 else "_fact")
        action_plan = encoded_das + fact + uses_fact
        for j, candidate in enumerate(candidates):
            lm_labels = bool(j == self.num_candidates - 1)
            instance = self.build_input_from_segments(history, candidate, action_plan, self.tokenizer, lm_labels)
            instance['das_to_return'] = das_to_return
            instances.append(instance)

        return instances


class TopicalChatsSentimentDataset(TopicalChatsDataset):

    def __init__(self, dataset, tokenizer, special_tokens, args, inference=False):
        super().__init__(dataset, tokenizer, special_tokens, args)

    def _construct_dialog_state(self, history):
        turn_history = []
        sentiment_history = []
        knowledge_history = [""]  # Hack to always have empty
        for (response, sentiments, past_knowledge) in history:
            turn_history.append(response)
            sentiment_history += sentiments

        dialog_state = {
            "turn_history": turn_history,
            "sentiment_history": sentiment_history,
            "knowledge_history": knowledge_history
        }

        return dialog_state

    def __getitem__(self, index):
        (history, (response, sentiment, knowledge)) = self.dataset[index]

        dialog_state = self._construct_dialog_state(history)
        history, fact = self.truncate_sequences(dialog_state["turn_history"], knowledge)

        candidates = self.sample_candidates(self.dataset, index)
        candidates.append(response)

        instances = []

        # The action plan must be ground-truth for training and validation
        # However, for inference time, it must follow the policy
        uses_fact = self.tokenizer.encode("_nofact" if len(knowledge) <= 1 else "_fact")
        sentiments = []
        for dict in sentiment:
            sentiments.append("<" + dict["label"] + ">")

        sentiment_encoded = self.tokenizer.encode(sentiments)
        action_plan = sentiment_encoded + fact + uses_fact
        for j, candidate in enumerate(candidates):
            lm_labels = bool(j == self.num_candidates - 1)
            instance = self.build_input_from_segments(history, candidate, action_plan, self.tokenizer, lm_labels)
            instances.append(instance)

        return instances