import argparse
import json
import logging
import pickle
from collections import defaultdict

import torch
import tornado.ioloop
import tornado.web

from pd_nrg.ranker import TfIdfRankerRetriever


class CruzControlHandler(tornado.web.RequestHandler):
    """
    Handler for the Cruz Control Application Web Server.
    """
    def set_default_headers(self) -> None:
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers",
                        "Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def _load_model(self, args):
        pass

    def _reply(self, history):
        return history[-1]

    def _load_knowledge_index(self, knowledge_index_path):
        with open(knowledge_index_path, 'rb') as knowledge_index_file:
            index = pickle.load(knowledge_index_file)

        return index

    def initialize(self, args, logger):
        index = self._load_knowledge_index(args.knowledge_index_path)
        self.ranker_retriever = TfIdfRankerRetriever(index, new_index=True)
        self.model = self._load_model(args)
        self.histories = defaultdict(list)
        self.logger = logger

    def post(self):
        body = json.loads(self.request.body.decode())
        self.logger.info("Received request: ", body)

        user_history = self.histories[body["userID"]]
        user_history.append(body["text"])
        response = self._reply(user_history)

        user_history.append(response)

        self.write(json.dumps({
            "body": {"utterance": response}
        }))

def make_app(args, logger):
    return tornado.web.Application([
        (r"/", CruzControlHandler, dict(args=args, logger=logger))
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CruzControl Server Script')

    parser.add_argument('--port', type=int, default=8089,
                        help='Port for the server')

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
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")

    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    logger = logging.getLogger(__file__)

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    app = make_app(args, logger)
    app.listen(args.port)
    logger.info(f"Server listening on http://localhost:{args.port}")
    tornado.ioloop.IOLoop.current().start()
