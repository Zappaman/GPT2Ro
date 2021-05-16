"""
Fine-tuning a GPT2 model for the Romanian language
"""
import argparse
import shutil
from config import cfg
import logging
import math
from typing import List, Optional
from logging import Logger
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import logging
import os
from RoDataset import RoDataset

from transformers import (
    AdamW,
    DataCollatorForLanguageModeling,
    get_scheduler,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)

from utils.model_utils import freeze_layers, unfreeze_layers
from generate import test_generation

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

global_steps = 0


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str,
                        default="experiments/gpt2_ro/frozen/frozen_1.yaml")

    return parser.parse_args()


def get_pretrained_path():
    import glob
    from config import cfg
    pretrained_path = None
    try:
        ckpts = sorted(glob.glob(os.path.join(cfg.EXPERIMENT.OUTPUT_FOLDER, "model_frozen_*.pth")),
                       key=lambda x:
                       (
            int(os.path.basename(x).split("_")[2]),
            int(os.path.basename(x).split("_")[4].split(".")[0]))
        )
        pretrained_path = ckpts[-1]  # interested in most recent checkpoint
    except Exception as e:
        print(
            f"Could not load last pretrained model from {cfg.EXPERIMENT.OUTPUT_FOLDER}. Error message below:")
        print(e)

    return pretrained_path


def inner_train_loop(model: Optional[GPT2LMHeadModel] = None,
                     train_dataloader: Optional[DataLoader] = None,
                     eval_dataloader: Optional[DataLoader] = None,
                     logger: Optional[Logger] = None,
                     frozen_layers: Optional[List[str]] = None,
                     frozen_ix: Optional[int] = None,
                     max_train_steps: Optional[int] = None,
                     optimizer: Optional[torch.optim.Optimizer] = None,
                     lr_scheduler=None,
                     writer: Optional[SummaryWriter] = None,
                     local_steps: Optional[int] = None,
                     best_perplexity: Optional[int] = None,
                     last_epoch: Optional[int] = None):

    global global_steps

    use_fp16 = cfg.TRAIN.USE_FP_16
    scaler = None
    if use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    unfreeze_layers(model)  # restore to default requires_grad=true
    freeze_layers(model, frozen_layers)

    progress_bar = tqdm(range(max_train_steps),
                        initial=local_steps, disable=False)
    for epoch in range(cfg.TRAIN.NUM_TRAIN_EPOCHS):
        if epoch < last_epoch:
            logger.info(
                f"Skipping epoch {epoch}, as we're continuing from {last_epoch}")
            continue
        model.train()
        for step, batch in enumerate(train_dataloader):
            # place all batch elements on cuda
            for k, v in batch.items():
                if type(v) == type(torch.tensor(0)):
                    batch[k] = batch[k].cuda()
            if use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
            else:
                outputs = model(**batch)

            loss = outputs.loss
            loss = loss / cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if global_steps % cfg.TRAIN.LOG_EVERY == 0:
                train_perplexity = math.exp(torch.mean(loss))
                writer.add_scalar(
                    f'Perplexity/train_freeze{frozen_ix}_steps', train_perplexity, global_steps)

            if step % cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS == 0 or step == len(train_dataloader) - 1:
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            local_steps += 1
            global_steps += 1

            if local_steps % cfg.TRAIN.EVAL_STEPS == 0 or local_steps >= max_train_steps:
                logger.info("Evaluating model")
                model.eval()
                losses = []
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        for k, v in batch.items():
                            if type(v) == type(torch.tensor(0)):
                                batch[k] = batch[k].cuda()
                        if use_fp16:
                            with torch.cuda.amp.autocast():
                                outputs = model(**batch)
                        else:
                            outputs = model(**batch)

                    loss = outputs.loss
                    losses.append(loss)

                losses = torch.stack(losses)
                perplexity = math.exp(torch.mean(losses))
                if local_steps % cfg.TRAIN.LOG_EVERY == 0:
                    writer.add_scalar(
                        f'Loss/eval_freeze{frozen_ix}_steps', perplexity, global_steps)

                logger.info(f"epoch {epoch}: perplexity: {perplexity}")
                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    model_path = os.path.join(
                        cfg.EXPERIMENT.OUTPUT_FOLDER, f"best_model.pth")
                    logger.info(f"saving model to {model_path}")

                    if not os.path.exists(cfg.EXPERIMENT.OUTPUT_FOLDER):
                        os.makedirs(cfg.EXPERIMENT.OUTPUT_FOLDER)

                    save_dict = {
                        "model": model,
                        "optimizer": optimizer,
                        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                        "frozen_ix": frozen_ix,
                        "frozen_layers": frozen_layers,
                        "local_steps": local_steps,
                        "global_steps": global_steps,
                        "best_perplexity": best_perplexity,
                        "last_epoch": epoch
                    }
                    torch.save(save_dict, model_path)
                    logger.info(f"generating text to see training progress...")

                    # generate some text to see progress
                    out_list = test_generation(
                        model, eval_dataloader.dataset.tokenizer)
                    for i, o in enumerate(out_list):
                        print(f"{i}: {o}")

                model.train()

            if local_steps % cfg.TRAIN.SAVE_STEPS == 0:
                model_path = os.path.join(
                    cfg.EXPERIMENT.OUTPUT_FOLDER, f"model_frozen_{frozen_ix}_steps_{local_steps}.pth")
                logger.info(f"saving model to {model_path}")

                if not os.path.exists(cfg.EXPERIMENT.OUTPUT_FOLDER):
                    os.makedirs(cfg.EXPERIMENT.OUTPUT_FOLDER)

                save_dict = {
                    "model": model,
                    "optimizer": optimizer,
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "frozen_ix": frozen_ix,
                    "frozen_layers": frozen_layers,
                    "local_steps": local_steps,
                    "global_steps": global_steps,
                    "best_perplexity": best_perplexity,
                    "last_epoch": epoch
                }
                torch.save(save_dict, model_path)

            if local_steps >= max_train_steps:
                break

        logger.info(f"Epoch {epoch}/{cfg.TRAIN.NUM_TRAIN_EPOCHS} done")


def transfer_embeddings(
        model: Optional[GPT2LMHeadModel] = None,
        orig_tokenizer: Optional[GPT2Tokenizer] = None,
        new_tokenizer: Optional[GPT2Tokenizer] = None):
    """
    Keep the word embeddings of the common tokens between the two tokenizers.
    Remap the embeddings according to the new tokenizer encoding
    Initialize the rest of the embeddings with default values.
    """
    common_tokens = set(new_tokenizer.encoder.keys()).intersection(
        set(orig_tokenizer.encoder.keys()))
    old_wte = model.base_model.wte
    old2new = {
        k: (orig_tokenizer.encoder[k], new_tokenizer.encoder[k]) for k in common_tokens}

    new_wte = torch.nn.Embedding(
        new_tokenizer.vocab_size, old_wte.embedding_dim).cuda()
    with torch.no_grad():
        new_wte.weight[[old2new[k][1] for k in old2new.keys()]] = old_wte.weight[[
            old2new[k][0] for k in old2new.keys()]].clone()

    model.base_model.wte = new_wte


def train():
    global global_steps

    # Parse arguments from command line
    args = create_args()
    cfg.merge_from_file(args.cfg)
    print(cfg)

    # Copy source and experiment to
    output_folder = cfg.EXPERIMENT.OUTPUT_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    logger.info(f"Copying {__file__} source to {output_folder}")
    shutil.copyfile(__file__, os.path.join(
        cfg.EXPERIMENT.OUTPUT_FOLDER, os.path.basename(__file__)))

    logger.info(f"Copying {args.cfg} source to {output_folder}")
    shutil.copyfile(args.cfg, os.path.join(
        cfg.EXPERIMENT.OUTPUT_FOLDER, os.path.basename(args.cfg)))
    writer = SummaryWriter(cfg.EXPERIMENT.TENSORBOARD_FOLDER)
    # Create datasets + dataloader for validation and train
    train_dataset = RoDataset(
        cfg.DATASET.TOKENIZER_PREFIX_PATH, cfg.DATASET.TRAIN_FILE)
    eval_dataset = RoDataset(
        cfg.DATASET.TOKENIZER_PREFIX_PATH, cfg.DATASET.VALID_FILE)

    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=cfg.EXPERIMENT.NUM_DATALOADER_WORKERS,
                                  batch_size=cfg.TRAIN.BATCH_SIZE, collate_fn=DataCollatorForLanguageModeling(train_dataset.tokenizer, mlm=False))
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, num_workers=cfg.EXPERIMENT.NUM_DATALOADER_WORKERS,
                                 batch_size=cfg.TRAIN.BATCH_SIZE, collate_fn=DataCollatorForLanguageModeling(eval_dataset.tokenizer, mlm=False))

    # Load pretrained model
    model = None
    if os.path.exists(cfg.TRAIN.LAST_PRETRAINED_MODEL):
        model = torch.load(cfg.TRAIN.LAST_PRETRAINED_MODEL)['model']
    else:  # if no model specified, fallback to English pretrained
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = model.cuda()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(
        f"  Num Epochs per frozen train loop = {cfg.TRAIN.NUM_TRAIN_EPOCHS}")

    # Embedding transfer
    transfer_embeddings(model, GPT2Tokenizer.from_pretrained(
        'gpt2'), train_dataset.tokenizer)

    # Train with gradual unfreezing
    unfreeze_groups = cfg.TRAIN.UNFREEZING.UNFREEZE_GROUPS
    train_steps_list = cfg.TRAIN.UNFREEZING.TRAIN_STEPS_LIST
    learning_rate_list = cfg.TRAIN.UNFREEZING.LEARNING_RATE_LIST
    saved_state = None
    local_steps, global_steps, last_epoch = 0, 0, 0
    restored_ix = -1
    best_perplexity = 9999999
    optimizer_state_dict, lr_scheduler_state_dict = None, None

    if cfg.EXPERIMENT.RESUME_TRAINING_ON_RESTART:
        pretrained_path = get_pretrained_path()
        logger.info(
            f"Found {pretrained_path} - will resume training accordingly")
        if pretrained_path:
            saved_state = torch.load(pretrained_path)
            # cut out already trained freeze steps
            restored_ix = saved_state['frozen_ix']
            unfreeze_groups = unfreeze_groups[restored_ix:]
            train_steps_list = train_steps_list[restored_ix:]
            learning_rate_list = learning_rate_list[restored_ix:]
            logger.info(f"Cutting unfreeze groups to index {restored_ix}")
        else:
            logger.info("Could not load pretrained path!")

    for ix, (unfrozen_group, train_steps, learning_rate) in enumerate(zip(unfreeze_groups, train_steps_list, learning_rate_list)):
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        lr_scheduler = get_scheduler(
            name=cfg.TRAIN.LR_SCHEDULER_TYPE,
            num_warmup_steps=cfg.TRAIN.NUM_WARMUP_STEPS,
            optimizer=optimizer,
            num_training_steps=train_steps,
        )

        # Restore the model, optimizer, lr_scheduler, local_steps, best_perplexity, global_steps
        # from the previously saved state if one exists
        if saved_state:
            local_steps = saved_state['local_steps']
            best_perplexity = saved_state['best_perplexity']
            global_steps = saved_state['global_steps']
            lr_scheduler_state_dict = saved_state['lr_scheduler_state_dict']
            last_epoch = saved_state['last_epoch'] if 'last_epoch' in saved_state else 0
            restored_ix = saved_state['frozen_ix']
            optimizer_state_dict = saved_state['optimizer'].state_dict()

            model.load_state_dict(saved_state['model'] .state_dict())

            if optimizer_state_dict:
                optimizer.load_state_dict(optimizer_state_dict)

            if lr_scheduler_state_dict:
                lr_scheduler.load_state_dict(lr_scheduler_state_dict)

            logger.info("Restored saved state")

            saved_state = None  # reset saved state to correctly continue training

        inner_train_loop(model=model,
                         train_dataloader=train_dataloader,
                         eval_dataloader=eval_dataloader,
                         logger=logger, frozen_layers=unfrozen_group,
                         frozen_ix=ix+restored_ix,
                         max_train_steps=train_steps,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         writer=writer,
                         best_perplexity=best_perplexity,
                         local_steps=local_steps,
                         last_epoch=last_epoch)


if __name__ == "__main__":
    train()
