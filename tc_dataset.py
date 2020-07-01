import pickle
from itertools import chain
import random

from torch.utils.data import Dataset

from pd_nrg.policies import KnowledgeDependent

from sklearn.metrics.pairwise import linear_kernel

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
        (history, response, fact) = self.dataset[index]

        # Truncate history turns to reduce memory requirement
        if len(history) > (2 * self.max_history + 1):
            history = history[-(2 * self.max_history + 1):]

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
        candidates = [response for (_, response, _) in random.sample(dataset, self.num_candidates - 1)]

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

        # I have no idea what this part refers to, Cargo Culting for now
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


class TopicalChatsKDDataset(TopicalChatsDataset):
    def _init_knowledge_index(self, knowledge_index_path):
        with open(knowledge_index_path, 'rb') as knowledge_index_file:
            index_data = pickle.load(knowledge_index_file)
        self.knowledge_retriever = index_data["bm25_index"]
        self.tfidf_vec = index_data["tfidf_vec"]
        self.knowledge_sentences = index_data["knowledge_list"]
        self.vectorized_sentences = self.tfidf_vec.transform(self.knowledge_sentences)


    def __init__(self, dataset, tokenizer, special_tokens, args, inference=False):
        self.dialog_policy = KnowledgeDependent()
        self._init_knowledge_index(args.knowledge_index_path)
        # For inference, the model will start executing the
        # heuristic dialog policy and knowledge selection policy
        self.inference = inference
        super().__init__(dataset, tokenizer, special_tokens, args)

    def sample_candidates(self, dataset, current_conversation_index):
        # Lets just hope that the number of cases where the true responses gets included in the
        # candidates is vanishingly small
        candidates = [response for (_, (response, _, _)) in random.sample(dataset, self.num_candidates - 1)]

        return candidates

    def _construct_dialog_state(self, history):
        turn_history = []
        da_history = []
        knowledge_history = []
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
            # closest_sentences = self.knowledge_retriever.get_top_n(last_turn, self.knowledge_sentences)
            tfidf_vecs = self.tfidf_vec.transform([last_turn])
            similarities = linear_kernel(self.vectorized_sentences, tfidf_vecs[0]).flatten()

            closest_knowledge_index = similarities.argsort()[-1]
            if similarities[closest_knowledge_index] > 0.25:
                return self.knowledge_sentences[closest_knowledge_index]
            else:
                return ""
        pass

    def _execute_heuristic_policy(self, dialog_state):
        knowledge = self._select_appropriate_knowledge(dialog_state)
        dialog_state["knowledge"] = knowledge  # Augment dialog state with knowledge
        das, knowledge = self.dialog_policy.get_knowledge_grounded_action(dialog_state)
        return das, self.tokenizer.encode(knowledge)

    def __getitem__(self, index):
        (history, (response, mezza_das, knowledge)) = self.dataset[index]

        dialog_state = self._construct_dialog_state(history)
        if self.inference:
            mezza_das, knowledge = self._execute_heuristic_policy(dialog_state)
            mezza_das = [{"da": da} for da in mezza_das]
            print(knowledge)
        history, fact = self.truncate_sequences(dialog_state["turn_history"], knowledge)

        candidates = self.sample_candidates(self.dataset, index)
        candidates.append(response)

        encoded_das = self.tokenizer.encode([f"<{da['da']}>" for da in mezza_das])
        instances = []

        # The action plan must be ground-truth for training and validation
        # However, for inference time, it must follow the policy
        uses_fact = self.tokenizer.encode("_nofact" if len(knowledge) == 0 else "_fact")
        action_plan = encoded_das + fact + uses_fact
        for j, candidate in enumerate(candidates):
            lm_labels = bool(j == self.num_candidates - 1)
            instance = self.build_input_from_segments(history, candidate, action_plan, self.tokenizer, lm_labels)
            instances.append(instance)
        return instances