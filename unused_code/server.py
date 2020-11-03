import argparse
import json
import logging
import pickle
from collections import defaultdict
from itertools import chain

import redis
import torch
import torch.nn.functional as F
import tornado.ioloop
import tornado.web
import tornado.httpserver
from tornado_swagger.setup import setup_swagger

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from pd_nrg.ranker import TfIdfRankerRetriever, ElasticRankerRetriever
from train_util.decode import top_filtering

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]


class CruzControlHandler(tornado.web.RequestHandler):
    """
    Handler for the Cruz Control Application Web Server.
    """
    def set_default_headers(self) -> None:
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers",
                        "Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def _select_knowledge(self, turn_history):
        if len(turn_history) == 0:
            return ""
        else:
            last_turn = self.tokenizer.decode(turn_history[-1])
            knowledge, similarity = self.ranker_retriever.get_top_n(last_turn, n=1)[0]

            if similarity > 0.2:
                return knowledge
            else:
                return ""


    def build_input_from_segments(self, history, response, fact, tokenizer, lm_labels=False):
        bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids((self.special_tokens[:-1]))

        """
        Input construction (may change):
        <bos> FACT <speaker1/2> UTT1 <speaker1/2> ... <speaker2> RESPONSE
        Considerations for design:
        1. Topical chat examples are created by adding a response every turn
        2. Last turn is always speaker2
        """
        sequence = [[bos] + fact] + history + [response]

        sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in
                                    enumerate(sequence[1:])]

        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

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

    def _reply(self, user_dialog_state, message):
        turn_history = user_dialog_state["turn_history"]
        # TODO: create separate tokenized turn history to avoid unnecessary encode/decodes
        turn_history.append(self.tokenizer.encode(message))

        knowledge = self._select_knowledge(turn_history)
        if knowledge == "":
            encoded_knowledge = self.tokenizer.encode("_nofact")
        else:
            encoded_knowledge = self.tokenizer.encode(knowledge)

        truncated_history, encoded_knowledge = self.truncate_sequences(turn_history, encoded_knowledge)
        instance = self.build_input_from_segments(truncated_history, [], encoded_knowledge, self.tokenizer)
        input_ids = instance["input_ids"]
        self.logger.info(self.tokenizer.decode(input_ids))

        token_type_ids = instance["token_type_ids"]

        current_output = []

        for j in range(self.max_length): # Add trailing tokens
            token_type_ids.append(token_type_ids[-1])
        token_type_ids = torch.tensor(token_type_ids).to(self.device)
        for j in range(args.max_length):
            prefix_input_seq = torch.tensor(input_ids + current_output).unsqueeze(0)
            truncated_tok_type_ids = token_type_ids[:prefix_input_seq.shape[-1]].unsqueeze(0)

            logits = self.model(prefix_input_seq.to(args.device), token_type_ids=truncated_tok_type_ids.to(args.device))
            if isinstance(logits, tuple) or len(logits.shape) == 4:  # for gpt2 and maybe others
                logits = logits[0]
            logits = logits[0, -1, :] / args.temperature
            logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
            probs = F.softmax(logits, dim=-1)

            prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
            if prev.item() in self.special_tokens_ids:
                while prev.item() in self.special_tokens_ids:
                    if probs.max().item() == 1:
                        # Disabled this rather noisy warning
                        # logger.warn("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=1)
            if prev.item() in self.special_tokens_ids:
                break
            current_output.append(prev.item())
        turn_history.append(current_output)
        return self.tokenizer.decode(current_output)

    def initialize(self, args, redis_client,
                   logger,
                   ranker_retriever,
                   model,
                   tokenizer,
                   special_tokens):
        self.redis_client = redis_client
        self.session_expiry = args.session_expiry
        self.max_history = args.max_history
        self.max_fact_length = args.max_fact_length
        self.max_length = args.max_length
        self.special_tokens = special_tokens

        self.ranker_retriever = ranker_retriever

        # TODO: investigate how the forward pass behaves
        #  in a multi-threaded environment (my current sense says it should be fine)
        #  since there's no mutation
        self.model, self.tokenizer = model, tokenizer

        self.special_tokens_ids = self.tokenizer.convert_tokens_to_ids(special_tokens)

        self.device = args.device
        self.logger = logger

    def post(self):
        """
        ---
        tags:
        - Reply

        summary: Send message to the server
        description: Send a message as the specified user. This creates a session if one is not present already.
        consumes:
        - application/json

        parameters:
        -   in: body
            name: message
            description: The message request for the user
            schema:
                type: object
                required:
                - userID
                - text

                properties:
                    userID:
                        type: string
                    text:
                        type: string
        produces:
        - application/json
        responses:
            '200':
                description: A response object
                content:
                    application/json:
                    schema:
                        type: object
                        properties:
                            body:
                                type: object
                                properties:
                                    utterance:
                                        type: string
                                        description: The output utterance produced by the model
        """
        body = json.loads(self.request.body.decode())
        self.logger.info(f"Received request: {body}")

        # Dialog states
        user_session_key = f'user:{body["userID"]}'
        state_data = self.redis_client.get(user_session_key)
        if state_data:
            user_dialog_state = json.loads(state_data)
        else:
            user_dialog_state = defaultdict(list)

        # user_dialog_state = self.dialog_states[body["userID"]]

        message = body["text"]

        response = self._reply(user_dialog_state, message)

        self.redis_client.set(user_session_key, json.dumps(user_dialog_state), ex=self.session_expiry)
        self.write(json.dumps({
            "body": {"utterance": response}
        }))

def _load_knowledge_index(knowledge_index_path):
    with open(knowledge_index_path, 'rb') as knowledge_index_file:
        index = pickle.load(knowledge_index_file)

    return index

def _load_model(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)

    if args.model_checkpoint in ["gpt2-medium", "gpt2-large"]:
        model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
    else:
        data = torch.load(args.model_checkpoint + "/pytorch_model.bin",map_location=args.device)
        model = data["mymodel"]
        model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint, state_dict=model.state_dict())
    model.to(args.device)
    logger.info(f"Loaded model {model}")

    return model, tokenizer

def make_app(args, logger):

    redis_client = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)

    # index = _load_knowledge_index(args.knowledge_index_path)
    # ranker_retriever = TfIdfRankerRetriever(index, new_index=True)

    ranker_retriever = ElasticRankerRetriever(args.elastic_host, args.elastic_port, args.elastic_alias)
    model, tokenizer = _load_model(args)

    routes = [
        tornado.web.url(r"/", CruzControlHandler, dict(args=args,
                                        redis_client=redis_client,
                                        logger=logger,
                                        ranker_retriever=ranker_retriever,
                                        model=model,
                                        tokenizer=tokenizer,
                                        special_tokens=SPECIAL_TOKENS))
    ]

    setup_swagger(routes)

    return tornado.web.Application(routes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CruzControl Server Script')

    parser.add_argument('--port', type=int, default=8089,
                        help='Port for the server')
    parser.add_argument('--redis_host', type=str, default="localhost",
                        help="Hostname for Redis instance")
    parser.add_argument('--redis_port', type=int, default=6379)

    parser.add_argument('--elastic_host', type=str, default="localhost",
                        help="Hostname for elastic instance")
    parser.add_argument('--elastic_port', type=int, default=9200,
                        help="Port for elastic instance")
    parser.add_argument('--elastic_alias', type=str, default="default",
                        help="Alias for elastic instance")
    parser.add_argument('--session_expiry', type=int, default=43200,
                        help="Time after which to purge a session (in seconds)")

    parser.add_argument('--model_configuration',
                        default='baseline',
                        choices=['baseline', 'kd-pd-nrg', 'kd-pd-nrg-swbd'],
                        help='Model configuration to make use of for the deployment')

    parser.add_argument('--knowledge_index_path', type=str, default="./tc_processed/tc_knowledge_index.pkl",
                        help="Path to knowledge index file")
    parser.add_argument('--model_checkpoint', type=str, default="gpt2-medium",
                        help="Path, url or short name of the model")
    parser.add_argument('--max_history', type=int, default=2,
                        help="Number of previous exchanges to condition on")
    parser.add_argument('--max_fact_length', type=int, default=200,
                        help='Number of fact tokens to include in the input')
    parser.add_argument('--experiment_name', type=str, default="topical_chats_gpt2",
                        help="The name of the experiment configuration for logging")
    # Decoding arguments
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the output utterances")

    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    logger = logging.getLogger(__file__)

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    app = make_app(args, logger)
    app.listen(args.port)
    logger.info(f"Server listening on http://localhost:{args.port}")
    tornado.ioloop.IOLoop.current().start()
