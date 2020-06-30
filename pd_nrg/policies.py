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


if __name__ == '__main__':
    dialog_state = {
        "knowledge": "Blah",
        "da_history": [STATEMENT],
        "knowledge_history": [""]
    }
    kd_policy = KnowledgeDependent()

    print(kd_policy.get_knowledge_grounded_action(dialog_state))