import itertools
import operator
import random

# Dialog Act Labels
NO_DIALOGUE_ACT = "NoDialogueAct"
THANKING = "Thanking"
DIRECTIVE = "Directive"
COMMISSIVE = "Commissive"
APOLOGY = "Apology"
CHOICE_Q = "ChoiceQ"
SET_Q = "SetQ"
SALUTATION = "Salutation"
PROP_Q = "PropQ"
STATEMENT = "Statement"
FEEDBACK = "Feedback"


STATEMENT_NON_OPINION = "Statement-non-opinion"
STATEMENT_OPINION = "Statement-opinion"
YES_NO_QUESTION = "Yes-No-Question"
APPRECIATION = "Appreciation"
WH_QUESTION = "Wh-Question"
CONVENTIONAL_CLOSING = "Conventional-closing"
OPEN_QUESTION = "Open-Question"
CONVENTIONAL_OPENING = "Conventional-opening"
DECLARATIVE_WH_QUESTION = "Declarative Wh-Question"
AGREE_ACCEPT = "Agree/Accept"
ACTION_DIRECTIVE = "Action-directive"
BACKCHANNEL_IN_QUESTION_FORM = "Backchannel in question form"
SIGNAL_NON_UNDERSTANDING = "Signal-non-understanding"
HEDGE = "HEDGE"
DECLARATIVE_YES_NO_QUESTION = "Declarative Yes-No-Question"
NEGATIVE_NON_NO_ANSWERS = "Negative non-no answers"
OR_CLAUSE = "Or-Clause"
OFFERS = "Offers, Options, Commits"
MAYBE_ACCEPT_PART = "Maybe/Accept-part"
AFFIRMATIVE_NON_YES_ANSWERS = "Affirmative non-yes answers"
REJECT = "Reject"
OTHER_ANSWERS = "Other answers"
SUMMARIZE = "Summarize/reformulate"
YES_ANSWERS = "Yes answers"
DOWNPLAYER = "Downplayer"
RHETORICAL_QUESTIONS = "Rhetorical-QUestions"
HOLD_BEFORE_ANSWER = "Hold before answer/agreement"
ACKNOWLEDGE = "Acknowledge"
NO_ANSWERS = "No answers"
OTHER = "Other"
NON_VERBAL = "Non-verbal"
UNINTERPRETABLE = "Uninterpretable"
TAG_QUESTION = "Tag-Question"
EQUAL_PLUX = "=+"
COLLABORATIVE_COMPLETION = "Collaborative Completion"
THIRD_PARTY_TALK = "3rd-party-talk"
REPEAT_PHRASE = "Repeat-phrase"
SELF_TALK = "Self-talk"
RESPONSE_ACKNOWLEDGE = "Response Acknowledgement"
QUOTATION = "Quotation"
ABANDONED_OR_TURN_EXIT = "Abandoned or Turn-Exit"
DISPREFRRED_ANSWERS = "Dispreferred answers"



class DialogPolicy(object):
    """
    DialogPolicy

    Implements an abstract knowledge-grounded dialog policy.
    The base class presupposes that the dialog state contains the knowledge already selected
    (i.e. an oracle provides the knowledge). this is not what Hedayatnia et al. 2020
    assumes (it has a knowledge selection policy), but we do so for the sake of simplicity
    """

    def choice(self, acts, weights):
        weights = itertools.accumulate(weights, func=operator.add)

        x = random.random()

        for i, weight in enumerate(weights):
            if x < weight:
                return acts[i]

        # Corner case in case some issue with weights arises
        return acts[-1]

    def retrieve_knowledge(self, dialog_state):
        return dialog_state["knowledge"]

    def _get_action_space(self, dialog_state):
        """
        Given a history of dialog acts,
        get a distribution of possible actions
        :return:
        """
        raise NotImplementedError("Get action space must be implemented")

    def get_action(self, dialog_state):
        actions, weights = self._get_action_space(dialog_state)
        action = self.choice(actions, weights)
        return action

    def _includes_knowledge(self, act):
        raise NotImplementedError("Get knowledge inclusion must be implemented")

    def _include_knowledge_in_acts(self, acts, knowledge, knowledge_history):
        return knowledge and any(self._includes_knowledge(act) for act in acts)

    def get_knowledge_grounded_action(self, dialog_state):
        action = self.get_action(dialog_state)
        knowledge = self.retrieve_knowledge(dialog_state)
        if self._include_knowledge_in_acts(action, knowledge, dialog_state["knowledge_history"]):
            return action, dialog_state["knowledge"]
        else:
            return action, ""


class KnowledgeIndependentSimple(DialogPolicy):
    def __init__(self):
        self.include_knowledge = {
            FEEDBACK: False,
            STATEMENT: True,
            PROP_Q: True,
            SALUTATION: False
        }

    def _get_action_space(self, dialog_state):
        weights = [0.5, 0.5]
        da_history = dialog_state["da_history"]
        if len(da_history) == 0:
            # Initial turn policy
            acts = [[SALUTATION, STATEMENT], [SALUTATION, PROP_Q]]
        else:
            if da_history[-1] == STATEMENT:
                acts = [[FEEDBACK, STATEMENT], [FEEDBACK, PROP_Q]]
            elif da_history[-1] == PROP_Q:
                acts = [[STATEMENT, STATEMENT], [STATEMENT, PROP_Q]]
            else:
                acts = [[STATEMENT, STATEMENT]]
                weights = [1.0]
        return acts, weights

    def _includes_knowledge(self, act):
        return self.include_knowledge.get(act, False)


class KnowledgeIndependentPropQ(DialogPolicy):
    def __init__(self):
        self.include_knowledge = {
            FEEDBACK: False,
            PROP_Q: True,
            SET_Q: True,
            CHOICE_Q: True,
            STATEMENT: True,
            APOLOGY: False,
            COMMISSIVE: False,
            DIRECTIVE: False,
            SALUTATION: False,
            THANKING: False,
            NO_DIALOGUE_ACT: True
        }

    def _get_action_space(self, dialog_state):
        da_history = dialog_state["da_history"]

        if len(da_history) == 0:
            acts = [[SALUTATION, STATEMENT], [SALUTATION, PROP_Q]]
            weights = [0.5, 0.5]
        else:
            acts = [[PROP_Q], [SET_Q], [CHOICE_Q], [FEEDBACK],
                    [STATEMENT],
                    [APOLOGY],
                    [COMMISSIVE],
                    [DIRECTIVE],
                    [SALUTATION],
                    [THANKING],
                    [NO_DIALOGUE_ACT]]
            weights = [0.657, 0.0343, 0.0343, 0.0343, 0.0343, 0.0343, 0.0343, 0.0343,
                       0.0343, 0.0343, 0.0343]
        return acts, weights

    def _includes_knowledge(self, act):
        return self.include_knowledge.get(act, False)

class KnowledgeIndependentAllQ(DialogPolicy):
    def __init__(self):
        self.include_knowledge = {
            FEEDBACK: False,
            PROP_Q: True,
            SET_Q: True,
            CHOICE_Q: True,
            STATEMENT: True,
            APOLOGY: False,
            COMMISSIVE: False,
            DIRECTIVE: False,
            SALUTATION: False,
            THANKING: False,
            NO_DIALOGUE_ACT: True
        }

    def _get_action_space(self, dialog_state):
        da_history = dialog_state["da_history"]

        if len(da_history) == 0:
            acts = [[SALUTATION, STATEMENT], [SALUTATION, PROP_Q]]
            weights = [0.5, 0.5]
        else:
            acts = [[PROP_Q], [SET_Q], [CHOICE_Q], [FEEDBACK],
                    [STATEMENT], [APOLOGY], [COMMISSIVE], [DIRECTIVE],
                    [SALUTATION], [THANKING], [NO_DIALOGUE_ACT]]
            weights = [0.219, 0.219, 0.219,
                       0.042875, 0.042875, 0.042875, 0.042875,
                       0.042875, 0.042875, 0.042875, 0.042875
                       ]
        return acts, weights

    def _includes_knowledge(self, act):
        return self.include_knowledge.get(act, False)


class KnowledgeDependent(DialogPolicy):
    """
    Knowledge Dependent Dialog Policy

    This was sort of implemented like the original paper since there is no
    clean design I could think of in the short term, although I prefer something
    better.
    """
    def get_knowledge_grounded_action(self, dialog_state):
        da_history = dialog_state["da_history"]
        knowledge = self.retrieve_knowledge(dialog_state)

        if len(da_history) == 0:
            acts = [[SALUTATION, STATEMENT], [SALUTATION, PROP_Q]]
            include_knowledge = {STATEMENT: True, PROP_Q: True, SALUTATION: False}
            weights = [0.5, 0.5]
        else:
            knowledge_history = dialog_state["knowledge_history"]
            if knowledge_history[-1] == knowledge:
                if da_history[-1] == STATEMENT:
                    acts = [[FEEDBACK, STATEMENT],
                            [FEEDBACK, PROP_Q]]

                    include_knowledge = {
                        STATEMENT: True,
                        PROP_Q: True,
                        FEEDBACK: False
                    }
                    weights = [0.5, 0.5]

                elif da_history[-1] == PROP_Q:
                    acts = [[STATEMENT, PROP_Q], [FEEDBACK, PROP_Q]]

                    include_knowledge = {
                        STATEMENT: False,
                        PROP_Q: True,
                        FEEDBACK: False
                    }

                    weights = [0.5, 0.5]

                else:
                    acts = [[FEEDBACK, STATEMENT]]
                    include_knowledge = {
                        STATEMENT: False,
                        PROP_Q: True,
                        FEEDBACK: False
                    }

                    weights = [1.0]

            else:
                include_knowledge = {
                    STATEMENT: False,
                    PROP_Q: True,
                    FEEDBACK: False
                }

                weights = [0.5, 0.5]

                if da_history[-1] == STATEMENT:
                    acts = [[FEEDBACK, STATEMENT],
                            [FEEDBACK, PROP_Q]
                            ]
                elif da_history[-1] == PROP_Q:
                    acts = [[STATEMENT, PROP_Q],
                            [FEEDBACK, PROP_Q]
                            ]
                else:
                    acts = [[FEEDBACK, PROP_Q]]
                    weights = [1.0]

        action = self.choice(acts, weights)

        if knowledge and any(include_knowledge.get(act, False) for act in action):
            return action, dialog_state["knowledge"]
        else:
            return action, ""


class KnowledgeIndependentSWBDPolicy(DialogPolicy):
    def __init__(self):
        self.include_knowledge = {
            STATEMENT_NON_OPINION: True,
            STATEMENT_OPINION: True,
            YES_NO_QUESTION: True,
            APPRECIATION: False,
            WH_QUESTION: True,
            CONVENTIONAL_CLOSING: False,
            OPEN_QUESTION: True,
            CONVENTIONAL_OPENING: False,
            DECLARATIVE_WH_QUESTION: True,
            AGREE_ACCEPT: False,
            ACTION_DIRECTIVE: False,
            BACKCHANNEL_IN_QUESTION_FORM: False,
            SIGNAL_NON_UNDERSTANDING: False,
            HEDGE: False,
            DECLARATIVE_YES_NO_QUESTION: True,
            NEGATIVE_NON_NO_ANSWERS: False,
            OR_CLAUSE: False,
            AFFIRMATIVE_NON_YES_ANSWERS: False,
            REJECT: False,
            OTHER_ANSWERS: False,
            SUMMARIZE: False,
            YES_ANSWERS: False,
            DOWNPLAYER: False,
            RHETORICAL_QUESTIONS: True,
            HOLD_BEFORE_ANSWER: False,
            ACKNOWLEDGE: False,
            NO_ANSWERS: False,
            OTHER: False,
            APOLOGY: False,
            NO_DIALOGUE_ACT: True
        }

    def _normalized_weight(self, weights):
        return [each / sum(weights) for each in weights]

    def _get_action_space(self, dialog_state):
        if dialog_state['inter_turn']:
            return self._get_action_space_inter_turn(dialog_state)
        else:
            return self._get_action_space_intra_turn(dialog_state)

    def _get_action_space_inter_turn(self, dialog_state):
        da_history = dialog_state["da_history"]

        if len(da_history) == 0:
            acts = [[CONVENTIONAL_OPENING, STATEMENT_NON_OPINION], [CONVENTIONAL_OPENING, OPEN_QUESTION]]
            weights = [0.5, 0.5]
        else:
            acts = []
            weights = []
            if da_history[-1] == ACKNOWLEDGE:
                acts = [[STATEMENT_OPINION, STATEMENT_NON_OPINION], [YES_NO_QUESTION, OPEN_QUESTION]]
                weights = self._normalized_weight([8.9, 1.3])
            elif da_history[-1] == ACTION_DIRECTIVE:
                acts = [[STATEMENT_NON_OPINION, STATEMENT_OPINION], [YES_NO_QUESTION, STATEMENT_NON_OPINION]]
                weights = self._normalized_weight([12.4, 1.8])
            elif da_history[-1] == AFFIRMATIVE_NON_YES_ANSWERS:
                acts = [[STATEMENT_NON_OPINION, STATEMENT_OPINION], [STATEMENT_OPINION, WH_QUESTION]]
                weights = self._normalized_weight([12.5, 12.5])
            elif da_history[-1] == AGREE_ACCEPT:
                acts = [[YES_NO_QUESTION, STATEMENT_OPINION], [STATEMENT_OPINION, STATEMENT_NON_OPINION]]
                weights = self._normalized_weight([3.5, 17.6])
            elif da_history[-1] == APOLOGY:
                acts = [[STATEMENT_NON_OPINION, STATEMENT_OPINION], [ACKNOWLEDGE, STATEMENT_OPINION]]
                weights = self._normalized_weight(([0.1, 0.1]))
            elif da_history[-1] == BACKCHANNEL_IN_QUESTION_FORM:
                acts = [[STATEMENT_OPINION, STATEMENT_NON_OPINION], [STATEMENT_OPINION, WH_QUESTION]]
                weights = self._normalized_weight([16.7, 16.7])
            elif da_history[-1] == CONVENTIONAL_CLOSING:
                acts = [[CONVENTIONAL_CLOSING, STATEMENT_OPINION], [STATEMENT_NON_OPINION, CONVENTIONAL_CLOSING]]
                weights = self._normalized_weight([1.5, 8.3])
            elif da_history[-1] == CONVENTIONAL_OPENING:
                acts = [[CONVENTIONAL_OPENING, YES_NO_QUESTION], [CONVENTIONAL_OPENING, OPEN_QUESTION]]
                weights = self._normalized_weight([4.2, 4.2])
            elif da_history[-1] == DECLARATIVE_YES_NO_QUESTION:
                acts = [[STATEMENT_OPINION, STATEMENT_NON_OPINION], [YES_NO_QUESTION, STATEMENT_OPINION]]
                weights = self._normalized_weight([15.9, 1.6])
            elif da_history[-1] == HEDGE:
                acts = [[YES_NO_QUESTION, STATEMENT_NON_OPINION], [STATEMENT_OPINION, STATEMENT_NON_OPINION]]
                weights = self._normalized_weight([12.5, 12.5])
            elif da_history[-1] == NEGATIVE_NON_NO_ANSWERS:
                acts = [[STATEMENT_NON_OPINION, STATEMENT_OPINION], [STATEMENT_OPINION, STATEMENT_OPINION]]
                weights = self._normalized_weight([23.1, 11.5])
            elif da_history[-1] == NO_ANSWERS:
                acts = [[STATEMENT_OPINION, STATEMENT_NON_OPINION], [STATEMENT_OPINION, STATEMENT_OPINION]]
                weights = self._normalized_weight([15.4, 7.7])
            elif da_history[-1] == OPEN_QUESTION:
                acts = [[STATEMENT_OPINION, STATEMENT_NON_OPINION], [STATEMENT_OPINION, STATEMENT_OPINION]]
                weights = self._normalized_weight([15.3, 16.8])
            elif da_history[-1] == OR_CLAUSE:
                acts = [[STATEMENT_NON_OPINION, STATEMENT_OPINION], [STATEMENT_OPINION, YES_NO_QUESTION]]
                weights = self._normalized_weight([23.1, 15.4])
            elif da_history[-1] == REJECT:
                acts = [[STATEMENT_OPINION, STATEMENT_NON_OPINION], [STATEMENT_OPINION, STATEMENT_OPINION]]
                weights = self._normalized_weight([14.3, 14.3])
            elif da_history[-1] == SIGNAL_NON_UNDERSTANDING:
                acts = [[STATEMENT_NON_OPINION, STATEMENT_OPINION], [STATEMENT_OPINION, STATEMENT_OPINION]]
                weights = self._normalized_weight([12, 21])
            elif da_history[-1] == STATEMENT_NON_OPINION:
                acts = [[YES_NO_QUESTION, STATEMENT_OPINION], [OPEN_QUESTION, STATEMENT_NON_OPINION]]
                weights = self._normalized_weight([1.6, 3.2])
            elif da_history[-1] == STATEMENT_OPINION:
                acts = [[STATEMENT_OPINION, YES_NO_QUESTION], [STATEMENT_NON_OPINION, YES_NO_QUESTION]]
                weights = self._normalized_weight([6.5, 2.8])
            elif da_history[-1] == THANKING:
                acts = [[STATEMENT_OPINION, CONVENTIONAL_CLOSING], [STATEMENT_OPINION, STATEMENT_NON_OPINION]]
                weights = self._normalized_weight([6.9, 15.3])
            elif da_history[-1] == WH_QUESTION:
                acts = [[STATEMENT_OPINION, YES_NO_QUESTION], [STATEMENT_OPINION, STATEMENT_NON_OPINION]]
                weights = self._normalized_weight([5.3, 13.4])
            elif da_history[-1] == YES_ANSWERS:
                acts = [[STATEMENT_OPINION, STATEMENT_NON_OPINION], [STATEMENT_OPINION, OPEN_QUESTION]]
                weights = self._normalized_weight([13.3, 13.3])
            elif da_history[-1] == YES_NO_QUESTION:
                acts = [[YES_ANSWERS, STATEMENT_OPINION], [NO_ANSWERS, STATEMENT_OPINION]]
                weights = self._normalized_weight([0.5, 0.5])
        return acts, weights

    def _get_action_space_intra_turn(self, dialog_state):
        da_history = dialog_state["da_history"]

        if len(da_history) == 0:
            acts = [[CONVENTIONAL_OPENING, STATEMENT_NON_OPINION], [CONVENTIONAL_OPENING, OPEN_QUESTION]]
            weights = [0.5, 0.5]
        else:
            acts = []
            weights = []
            if da_history[-1] == ACKNOWLEDGE:
                acts = [[STATEMENT_OPINION, STATEMENT_NON_OPINION], [STATEMENT_NON_OPINION, YES_NO_QUESTION]]
                weights = self._normalized_weight([13.3, 8.3])
            elif da_history[-1] == ACTION_DIRECTIVE:
                acts = [[STATEMENT_NON_OPINION, STATEMENT_OPINION], [STATEMENT_NON_OPINION, YES_NO_QUESTION]]
                weights = self._normalized_weight([12.4, 7.9])
            elif da_history[-1] == AFFIRMATIVE_NON_YES_ANSWERS:
                acts = [[STATEMENT_NON_OPINION, STATEMENT_OPINION], [STATEMENT_OPINION, YES_NO_QUESTION]]
                weights = self._normalized_weight([8.5, 6.8])
            elif da_history[-1] == AGREE_ACCEPT:
                acts = [[YES_NO_QUESTION, STATEMENT_OPINION], [STATEMENT_NON_OPINION, STATEMENT_OPINION]]
                weights = self._normalized_weight([4.7, 17.7])
            elif da_history[-1] == APOLOGY:
                acts = [[STATEMENT_NON_OPINION, STATEMENT_OPINION], [STATEMENT_OPINION, YES_NO_QUESTION]]
                weights = self._normalized_weight(([11.1, 13.6]))
            elif da_history[-1] == APPRECIATION:
                acts = [[STATEMENT_OPINION, YES_NO_QUESTION], [STATEMENT_NON_OPINION, STATEMENT_NON_OPINION]]
                weights = self._normalized_weight([5.9, 18.8])
            elif da_history[-1] == BACKCHANNEL_IN_QUESTION_FORM:
                acts = [[STATEMENT_OPINION, STATEMENT_NON_OPINION], [STATEMENT_NON_OPINION, YES_NO_QUESTION]]
                weights = self._normalized_weight([11.3, 8.7])
            elif da_history[-1] == CONVENTIONAL_CLOSING:
                acts = [[YES_NO_QUESTION, STATEMENT_NON_OPINION], [STATEMENT_NON_OPINION, STATEMENT_OPINION]]
                weights = self._normalized_weight([3.3, 11.5])
            elif da_history[-1] == CONVENTIONAL_OPENING:
                acts = [[YES_NO_QUESTION, STATEMENT_NON_OPINION], [STATEMENT_OPINION, STATEMENT_OPINION]]
                weights = self._normalized_weight([4.9, 12.2])
            elif da_history[-1] == DECLARATIVE_YES_NO_QUESTION:
                acts = [[STATEMENT_OPINION, STATEMENT_NON_OPINION], [STATEMENT_OPINION, YES_NO_QUESTION]]
                weights = self._normalized_weight([12.2, 4.7])
            elif da_history[-1] == HEDGE:
                acts = [[STATEMENT_NON_OPINION, STATEMENT_NON_OPINION], [STATEMENT_NON_OPINION, OPEN_QUESTION]]
                weights = self._normalized_weight([20.6, 1.5])
            elif da_history[-1] == NEGATIVE_NON_NO_ANSWERS:
                acts = [[STATEMENT_NON_OPINION, STATEMENT_OPINION], [STATEMENT_OPINION, YES_NO_QUESTION]]
                weights = self._normalized_weight([20.7, 8.1])
            elif da_history[-1] == NO_ANSWERS:
                acts = [[STATEMENT_OPINION, STATEMENT_NON_OPINION], [STATEMENT_OPINION, YES_NO_QUESTION]]
                weights = self._normalized_weight([13.2, 7.7])
            elif da_history[-1] == OPEN_QUESTION:
                acts = [[STATEMENT_OPINION, STATEMENT_NON_OPINION], [YES_NO_QUESTION, STATEMENT_NON_OPINION]]
                weights = self._normalized_weight([22.6, 5.6])
            elif da_history[-1] == OR_CLAUSE:
                acts = [[STATEMENT_NON_OPINION, STATEMENT_OPINION], [STATEMENT_NON_OPINION, YES_NO_QUESTION]]
                weights = self._normalized_weight([20, 10])
            elif da_history[-1] == STATEMENT_NON_OPINION:
                acts = [[STATEMENT_NON_OPINION, STATEMENT_NON_OPINION], [STATEMENT_OPINION, YES_NO_QUESTION]]
                weights = self._normalized_weight([20.2, 6.3])
            elif da_history[-1] == STATEMENT_OPINION:
                acts = [[STATEMENT_NON_OPINION, STATEMENT_NON_OPINION], [STATEMENT_NON_OPINION, YES_NO_QUESTION]]
                weights = self._normalized_weight([17.3, 5.6])
            elif da_history[-1] == THANKING:
                acts = [[STATEMENT_OPINION, CONVENTIONAL_CLOSING], [STATEMENT_OPINION, YES_NO_QUESTION]]
                weights = self._normalized_weight([1.9, 6.5])
            elif da_history[-1] == WH_QUESTION:
                acts = [[STATEMENT_OPINION, YES_NO_QUESTION], [STATEMENT_OPINION, STATEMENT_NON_OPINION]]
                weights = self._normalized_weight([3.7, 11.6])
            elif da_history[-1] == YES_ANSWERS:
                acts = [[STATEMENT_OPINION, STATEMENT_NON_OPINION], [STATEMENT_OPINION, YES_NO_QUESTION]]
                weights = self._normalized_weight([6.1, 8.3])
            elif da_history[-1] == YES_NO_QUESTION:
                acts = [[STATEMENT_OPINION, STATEMENT_NON_OPINION], [STATEMENT_OPINION, YES_NO_QUESTION]]
                weights = self._normalized_weight([14.2, 4.5])
        return acts, weights

    def _includes_knowledge(self, act):
        return self.include_knowledge.get(act, False)


if __name__ == '__main__':
    dialog_state = {
        "knowledge": "Blah",
        "da_history": [CONVENTIONAL_CLOSING],
        "knowledge_history": [""],
        "inter_turn": True
    }
    kd_policy = KnowledgeIndependentSWBDPolicy()
    print(kd_policy.get_knowledge_grounded_action(dialog_state))
