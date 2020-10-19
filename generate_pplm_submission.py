import argparse
import pickle
from operator import add

import numpy as np

import torch
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from tqdm.auto import tqdm, trange

from taggers.models import PPLMGPT2Classifier

SMALL_CONST = 1e-15
BIG_CONST = 1e10

CACHED_DISCRIMINATOR = {
    "athena": "taggers/checkpoints/pplm_clf_1.pt"
}

def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits)

def perturb_past(
    past,
    model,
    last,
    unpert_past,
    unpert_logits,
    accumulated_hidden,
    stepsize=0.02,
    classifier=None,
    class_label=None,
    num_iterations=3,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    kl_scale=0.01,
    device="cuda"
):
    # Delta(H_t)
    grad_accumulator = [np.zeros(p.shape).astype("float32") for p in past]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        # Discount weights from past
        decay_mask = torch.arange(0.0, 1.0 + SMALL_CONST, 1.0 / (window_length)) [1:]
    else:
        decay_mask = 1.0

    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = tuple(past[0].shape[:-2]) + tuple([window_length]) + tuple(past[0].shape[-1:])

        zeros_key_val_shape = (
                tuple(past[0].shape[:-2]) + tuple([curr_length - window_length]) + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat((ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

    # Store the loss per perturbation steps
    loss_per_iteration = []
    new_accumulated_hidden = None

    for i in range(num_iterations):
        print(f"Iteration {i + 1}")
        # Delta(H_t)
        curr_perturbation = [torch.from_numpy(p_).requires_grad_(True).to(device) for p_ in grad_accumulator]

        for p_ in curr_perturbation:
            p_.retain_grad()

        # Compute the perturbed hidden value for the past states
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape
        all_logits, _, all_hidden = model(last, past=perturbed_past)
        # Hidden state of last layer
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()

        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []

        # Compute discriminator loss
        # TODO: Implement arbitrary parameter linking
        ce_loss = torch.nn.CrossEntropyLoss()
        curr_unpert_past = unpert_past
        curr_probs = torch.unsqueeze(probs, dim=1)
        wte = model.resize_token_embeddings()

        for _ in range(horizon_length):
            inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
            _, curr_unpert_past, curr_all_hidden = model(past=curr_unpert_past, inputs_embeds=inputs_embeds)
            curr_hidden = curr_all_hidden[-1]
            new_accumulated_hidden = new_accumulated_hidden + torch.sum(curr_hidden, dim=1)

        _, prediction = classifier(new_accumulated_hidden / (curr_length + 1 + horizon_length))
        label = torch.tensor(prediction.shape[0] * [class_label], device=device, dtype=torch.long)
        discrim_loss = ce_loss(prediction, label)

        loss += discrim_loss
        loss_list.append(discrim_loss)

        # Compute KL-loss

        kl_loss = 0.
        if kl_scale > 0.:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=1)

            # This is label smoothing where we set a lower bound on probabilities. This is probably inconsistent
            unpert_probs = unpert_probs + SMALL_CONST * (unpert_probs <= SMALL_CONST).float().to(device).detach()
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(device).detach()
            corrected_probs = probs + correction.detach()

            kl_loss = kl_scale * ((corrected_probs * (corrected_probs/unpert_probs).log()).sum())
            print(f"KL loss: {kl_loss.data.cpu().numpy()}")
            loss += kl_loss

        loss_per_iteration.append(loss.data.cpu().numpy())
        print(f"PPLM loss: {(loss - kl_loss).data.cpu().numpy()}")

        # Compute gradients
        loss.backward()

        # Calculate norm of gradient of delta(H_t)
        grad_norms = [
            (torch.norm(p_.grad * window_mask) + SMALL_CONST) for index, p_ in enumerate(curr_perturbation)
        ]

        # normalize gradient for delta(H_t)
        grad = [
            -stepsize * (p_.grad * window_mask / grad_norms[index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        grad_accumulator = list(map(add, grad, grad_accumulator))

        # Zero the gradients
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # Apply the accumulated perturbations to the past
    grad_accumulator = [torch.from_numpy(p_).requires_grad_(True).to(device) for p_ in grad_accumulator]
    # H^{hat}_t = H_t + Delta(H_t)
    perturbed_past = list(map(add, past, grad_accumulator))

    return perturbed_past, new_accumulated_hidden, grad_norms, loss_per_iteration

def generate_text(
        model,
        tokenizer,
        context,
        device,
        length,
        perturb,

        classifier=None,
        class_label=None,
):
    past = None
    output_so_far = None
    grad_length = 10000
    num_iterations = 3
    current_stepsize = stepsize = 0.02
    temperature = 1.0
    repetition_penalty = 1.0
    gm_scale = 0.9
    sample = True
    top_k = 10

    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []
    for i in trange(length, ascii=True):

        # Get the past for the current output except for the last word

        # run the forward pass to get the past information
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                _, past, _ = model(output_so_far[:, :-1])

        unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
        unpert_last_hidden = unpert_all_hidden[-1]

        if i >= grad_length:
            current_stepsize = 0

        # Update the past
        if not perturb or num_iterations == 0:
            perturbed_past = past
        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                perturbed_past, _, grad_norms, loss_this_iter = perturb_past(
                    past=past,
                    model=model,
                    last=last,
                    classifier=classifier,
                    class_label=class_label,
                    accumulated_hidden=accumulated_hidden,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits
                )  # Perform past perturbation using discriminator loss

                loss_in_time.append(loss_this_iter)
            else:
                perturbed_past = past

        # Obtained the perturbed state info
        # o^{hat}_{t+1}, past, H^{hat}_{t+1} = LM(x_t, H^{hat}_{t}) as described in section 3.2
        pert_logits, past, pert_all_hidden = model(last, past=perturbed_past)
        pert_logits = pert_logits[:, -1, :] / temperature  # Reshape distribution

        # Bias against repetition
        for token_index in set(output_so_far[0].tolist()):
            if pert_logits[0, token_index] < 0:
                pert_logits[0, token_index] *= repetition_penalty
            else:
                pert_logits[0, token_index] /= repetition_penalty

        pert_probs = F.softmax(pert_logits, dim=-1)

        # Get discriminator loss

        ce_loss = torch.nn.CrossEntropyLoss()
        _, prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
        label = torch.tensor([class_label], device=device, dtype=torch.long)
        unpert_discrim_loss = ce_loss(prediction, label)
        print("Unperturbed discrim loss", unpert_discrim_loss.data.cpu().numpy())

        # Perform post-norm Geometric-Mean fusion as described in Section 3.3
        if perturb:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = (pert_probs ** gm_scale) * (unpert_probs ** (1 - gm_scale))
            pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)

            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)
        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)
            pert_probs = F.softmax(pert_logits, dim=-1)

        # Sample from top-k or perform MLE decoding
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)
        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # Update the output obtained so far
        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
        print(tokenizer.decode(output_so_far.tolist()[0]))

    return output_so_far, unpert_discrim_loss, loss_in_time

def run_full_text_generation(
    model,
    tokenizer,
    context,
    num_samples,
    classifier,
    class_label,
    device="cpu"

):
    length = 10
    unperturbed_generated_tokens, _, _ = generate_text(
        model,
        tokenizer,
        context,
        device=device,
        length=length,
        perturb=False,
        classifier=classifier,
        class_label=class_label
    )

    perturbed_generated_token_samples = []
    discriminator_losses = []
    losses_in_time = []
    for i in range(num_samples):
        perturbed_generated_tokens, discriminator_loss, loss_in_time = generate_text(
            model,
            tokenizer,
            context,
            device=device,
            length=length,
            perturb=True,
            classifier=classifier,
            class_label=class_label
        )

        perturbed_generated_token_samples.append(perturbed_generated_token_samples)
        discriminator_losses.append(discriminator_loss)

        losses_in_time.append(loss_in_time)


    return unperturbed_generated_tokens, perturbed_generated_token_samples, discriminator_losses, losses_in_time




def run_planned_pplm_examples(
        pretrained_model_checkpoint,
        cond_text,
        class_label,
        device
):
    num_samples = 30
    with open('taggers/checkpoints/pplm_config.pkl', 'rb') as pplm_training_config_file:
        training_config = pickle.load(pplm_training_config_file)

    vocab = training_config["vocab"]
    print(vocab.itos)
    discriminator = PPLMGPT2Classifier(num_labels=len(vocab), pretrained_model=pretrained_model_checkpoint, cached_mode=True,
                                       device=device)

    state_dict = torch.load(CACHED_DISCRIMINATOR["athena"])
    discriminator.load_state_dict(state_dict)

    discriminator.to(device)
    discriminator.eval()

    model = GPT2LMHeadModel.from_pretrained(pretrained_model_checkpoint,
                                            output_hidden_states=True)

    model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_checkpoint)

    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    uncond = len(cond_text) == 0
    if uncond:
        tokenized_cond_text = tokenizer.encode(tokenizer.bos_token)
    else:
        tokenized_cond_text = tokenizer.encode(tokenizer.bos_token + cond_text)

    print("== Prefix of sentence ==")
    print(tokenizer.decode(tokenized_cond_text))
    print()

    unperturbed_generated_tokens, perturbed_generated_tokens_samples, _ , _ = run_full_text_generation(
        model,
        tokenizer,
        tokenized_cond_text,
        num_samples,
        discriminator,
        class_label=class_label,
        device=device
    )

    unperturbed_generated_text = tokenizer.decode(unperturbed_generated_tokens.tolist()[0])
    print("= Unperturbed generated text =")
    print(unperturbed_generated_text)
    print()

    generated_texts = []

    for i, perturbed_generated_tokens in enumerate(perturbed_generated_tokens_samples):
        perturbed_generated_text = tokenizer.decode(perturbed_generated_tokens[0])
        print(f"= Perturbed generated text {i + 1}= ")

        generated_texts.append(perturbed_generated_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_checkpoint',
                        type=str,
                        default='gpt2-medium',
                        help='Pretrained model name')
    parser.add_argument('--cond_text',
                        type=str,
                        default='the book')
    parser.add_argument('--device',
                        type=str,
                        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--class_label', type=int, default=1,
                        help="Index of the class to predict")
    args = parser.parse_args()

    run_planned_pplm_examples(**vars(args))