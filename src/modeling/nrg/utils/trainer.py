import math
import os.path

import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

class GPT2DoubleHeadLMTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 device="cpu",
                 lm_weight=1.0,
                 mc_weight=1.0,
                 gradient_accumulation_steps=1,
                 scheduler=None,
                 writer=None,
                 log_every_n=500,
                 save_every_n=1800,
                 eval_every_n=3600,
                 max_gradient_norm=1.0,
                 model_checkpoint_directory=None):

        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.model.to(device)

        self.scheduler = scheduler
        self.writer = writer

        self.lm_weight = lm_weight
        self.mc_weight = mc_weight

        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.log_every_n = log_every_n
        self.save_every_n = save_every_n
        self.eval_every_n = eval_every_n
        self.max_gradient_norm = max_gradient_norm

        self.ce_loss = CrossEntropyLoss(ignore_index=-100)

        self.model_checkpoint_directory = model_checkpoint_directory

    def _save_model_checkpoint(self, checkpoint_index):
        if self.model_checkpoint_directory:
            os.makedirs(self.model_checkpoint_directory, exist_ok=True)
            self.model.save_pretrained(os.path.join(self.model_checkpoint_directory, str(checkpoint_index)))


    def _train(self, train_loader, val_loader):
        self.model.train()

        self.optimizer.zero_grad()

        epoch_running_loss = 0.

        for i, batch in tqdm(enumerate(train_loader)):
            batch = (tensor.to(self.device) for tensor in batch)
            input_ids, mc_token_ids, labels, mc_labels, token_type_ids = batch
            lm_loss, mc_loss, *_ = self.model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
                mc_labels=mc_labels, labels=labels, return_dict=False)

            loss = (lm_loss * self.lm_weight + mc_loss * self.mc_weight) / self.gradient_accumulation_steps

            epoch_running_loss += (float(loss) - epoch_running_loss) / (i + 1)

            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)

            self.training_state["global_step_index"] += 1
            self.writer.add_scalar('Loss/train', epoch_running_loss, self.training_state["global_step_index"])

            if (i + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if (i + 1) % self.log_every_n == 0:
                print(f"Step: {i + 1} \t Loss: {epoch_running_loss}")

            if (i + 1) % self.eval_every_n == 0:
                self._eval(val_loader)

            if (i + 1) % self.save_every_n == 0:
                self._save_model_checkpoint(i + 1)

        print("\nTraining epoch completed!")
        print(f"Epoch Loss: {epoch_running_loss}")





    def _eval(self, val_loader):
        self.model.eval()

        running_nll_loss = 0.

        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_loader)):
                batch = (tensor.to(self.device) for tensor in batch)
                input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
                # if we dont send labels to model, it doesnt return losses
                lm_logits, mc_logits, *_ = self.model(
                    input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,return_dict=False
                )

                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

                loss = self.ce_loss(lm_logits_flat_shifted, lm_labels_flat_shifted)

                running_nll_loss += (float(loss) - running_nll_loss) / (i + 1)

                self.training_state["global_validation_step_index"] += 1
                self.writer.add_scalar('Loss/valid', running_nll_loss, self.training_state["global_validation_step_index"])

        print("\nValidation Completed!")
        print(f"Validation Loss: {running_nll_loss}")
        print(f"Perplexity: {math.exp(running_nll_loss)}")

        # Use validation loss for best model selection
        if running_nll_loss < self.training_state["lowest_validation_loss"]:
            self.training_state["lowest_validation_loss"] = running_nll_loss
            print("Current validation loss lower than the lowest loss. Saving model checkpoint")
            self._save_model_checkpoint("best")

    def train(self, train_loader, val_loader, n_epochs):
        self.training_state = {
            "lowest_validation_loss": float('inf'),
            "global_step_index": 0,
            "global_validation_step_index": 0
        }

        for epoch in range(n_epochs):
            print(f"\nEpoch: {epoch}")
            self._train(train_loader, val_loader)
            self._eval(val_loader)
