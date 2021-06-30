from tqdm.auto import tqdm

class GPT2DoubleHeadLMTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 lm_weight=1.0,
                 mc_weight=1.0,
                 gradient_accumulation_steps=1,
                 scheduler=None,
                 writer=None):

        self.model = model
        self.optimizer = optimizer

        self.scheduler = scheduler
        self.writer = writer

        self.lm_weight = lm_weight
        self.mc_weight = mc_weight
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def _train(self, train_loader):
        self.model.train()

        self.optimizer.zero_grad()

        for i, batch in tqdm(enumerate(train_loader)):

            input_ids, mc_token_ids, labels, mc_labels, token_type_ids = batch

            lm_loss, mc_loss, *_ = self.model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
                mc_labels=mc_labels, labels=labels)

            loss = (lm_loss * self.lm_weight + mc_loss) / self.gradient_accumulation_steps

            loss.backward()

            if (i + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()


    def _eval(self, val_loader):
        self.model.eval()

    def train(self, train_loader, val_loader, n_epochs):

        for epoch in range(n_epochs):
            self._train(train_loader)
            self._eval(val_loader)