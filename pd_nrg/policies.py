from numpy.random import choice

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


class DialogPolicy(object):
    def _get_action_space(self, dialog_state):
        """
        Given a history of dialog acts,
        get a distribution of possible actions
        :return:
        """
        raise NotImplementedError("Get action space must be implemented")

    def get_action(self, dialog_state):
        actions, weights = self._get_action_space(dialog_state)
        action = choice(actions, p=weights)
        return action

    def _includes_knowledge(self, act):
        raise NotImplementedError("Get knowledge inclusion must be implemented")

    def _include_knowledge_in_acts(self, acts, knowledge, knowledge_history):
        return knowledge and any(self._includes_knowledge(act) for act in acts)

    def get_knowledge_grounded_action(self, dialog_state):
        action = self.get_action(dialog_state)

        if self._include_knowledge_in_acts(action, dialog_state["knowledge"], dialog_state["knowledge_history"]):
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
                       0.0343, 0.0343, 0.0343, 0.0343]
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
            acts = [[SALUTATION, STATEMENT], [SALUTATION,PROP_Q]]
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
    def get_knowledge_grounded_action(self, dialog_state):
        pass