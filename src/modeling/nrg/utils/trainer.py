import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

class GPT2DoubleHeadLMTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 lm_weight=1.0,
                 mc_weight=1.0,
                 gradient_accumulation_steps=1,
                 scheduler=None,
                 writer=None,
                 log_every_n=500,
                 save_every_n=1800,
                 max_gradient_norm=1.0):

        self.model = model
        self.optimizer = optimizer

        self.scheduler = scheduler
        self.writer = writer

        self.lm_weight = lm_weight
        self.mc_weight = mc_weight

        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.log_every_n = log_every_n
        self.save_every_n = save_every_n
        self.max_gradient_norm = max_gradient_norm

    def _train(self, train_loader):
        self.model.train()

        self.optimizer.zero_grad()

        epoch_running_loss = 0.

        for i, batch in tqdm(enumerate(train_loader)):

            input_ids, mc_token_ids, labels, mc_labels, token_type_ids = batch
            lm_loss, mc_loss, *_ = self.model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
                mc_labels=mc_labels, labels=labels, return_dict=False)

            loss = (lm_loss * self.lm_weight + mc_loss * self.mc_weight) / self.gradient_accumulation_steps

            epoch_running_loss += (float(loss) - epoch_running_loss) / (i + 1)

            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)

            if (i + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if (i + 1) % self.log_every_n == 0:
                print(f"Step: {i + 1} \t Loss: {epoch_running_loss}")




    def _eval(self, val_loader):
        self.model.eval()

    def train(self, train_loader, n_epochs):

        for epoch in range(n_epochs):
            self._train(train_loader)
            # self._eval(val_loader)