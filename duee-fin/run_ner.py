import copy
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW

from dataset.dataset import collate_fn, DuEEEventDataset
from metric.metric import ChunkEvaluator
from model.model import DuEEEvent_model
from utils.finetuning_argparse import get_argparse
from utils.utils import init_logger, seed_everything, logger, ProgressBar


def evaluate(args, eval_iter, model, metric):
    """evaluate"""
    metric.reset()
    batch_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(args.device)

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(eval_iter):
            for key in batch.keys():
                batch[key] = batch[key].to(args.device)
            logits = model(
                input_ids=batch['all_input_ids'],
                attention_mask=batch['all_attention_mask'],
                token_type_ids=batch['all_token_type_ids']
            )
            loss = criterion(logits.view(-1, args.num_classes),
                             batch["all_labels"].view(-1))
            batch_loss += loss.item()

            preds = torch.argmax(logits, axis=-1) #! -1是最后一个维度 logits.shape:batch, seq, class   preds.shape: batch, seq
            n_infer, n_label, n_correct = metric.compute(batch["all_seq_lens"], preds, batch['all_labels']) #! 参数1: [93,36] 参数: 句子长度 预测的trigger 真实的trigger 结果(正确的数目): 3 2 2 
            metric.update(n_infer, n_label, n_correct)

    precision, recall, f1_score = metric.accumulate()

    return precision, recall, f1_score, batch_loss/(step+1) #! 这个step+1 就相当于所有的个数

def train(args, train_iter, model):
    logger.info("***** Running train *****")
    # 优化器
    no_decay = ["bias", "LayerNorm.weight"] #! bias 和 layernorm.weiht 不需要decay
    roberta_param_optimizer = list(model.roberta.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [ #! roberta里需要有decay的 | roberta里不需要有decay的 | linear 里需要有decay的 | linear里不需要有decay的 
        {'params': [p for n, p in roberta_param_optimizer if not any(nd in n for nd in no_decay)], #! named_parameter返回 名称str和param, 如果no_decay的字符都不在当前的名称str里, 就将param加入, 代表这个param是需要decay的
         'weight_decay': args.weight_decay,
         'lr': args.learning_rate}, #! 这个learning_rate是预训练模型的learning_rate 保礼说设为-2左右就可以了
        {'params': [p for n, p in roberta_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.linear_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.linear_learning_rate},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(args.device) #! ignore_index 指定忽略某些类别的loss
    batch_loss = 0
    pbar = ProgressBar(n_total=len(train_iter), desc='Training')
    print("****" * 20)
    for step, batch in enumerate(train_iter):
        for key in batch.keys():
            batch[key] = batch[key].to(args.device)
        logits = model(
            input_ids=batch['all_input_ids'],
            attention_mask=batch['all_attention_mask'],
            token_type_ids=batch['all_token_type_ids']
        )
        logits = logits.view(-1, args.num_classes) #! batch*seq num_classes
        # 正常训练
        loss = criterion(logits, batch["all_labels"].view(-1)) #! batch*seq
        loss.backward()
        #
        batch_loss += loss.item()
        pbar(step,
             {
                 'batch_loss': batch_loss / (step + 1),
             })
        optimizer.step()
        model.zero_grad()

def main():
    args = get_argparse().parse_args()
    print(json.dumps(vars(args), sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
    # init_logger(log_file="./log/{}.log".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    seed_everything(args.seed)

    args.output_model_path = os.path.join(args.output_dir, args.dataset, args.event_type, "best_model.pkl")
    # 设置保存目录
    if not os.path.exists(os.path.dirname(args.output_model_path)):
        os.makedirs(os.path.dirname(args.output_model_path))

    # device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/home/hanqing/bixingjian/pretrained-model/chinese-roberta-wwm-ext-large")

    # dataset & dataloader
    args.train_data = "./data/{}/{}/train.tsv".format(args.dataset, args.event_type)
    args.dev_data = "./data/{}/{}/dev.tsv".format(args.dataset, args.event_type)
    args.tag_path = "./conf/{}/{}_tag.dict".format(args.dataset, args.event_type)
    train_dataset = DuEEEventDataset(args,
                                   args.train_data,
                                   args.tag_path,
                                   tokenizer)
    eval_dataset = DuEEEventDataset(args,
                                  args.dev_data,
                                  args.tag_path,
                                  tokenizer)
    logger.info("The nums of the train_dataset features is {}".format(len(train_dataset)))
    logger.info("The nums of the eval_dataset features is {}".format(len(eval_dataset)))
    train_iter = DataLoader(train_dataset, #! len=3625 (train_dataset=7250, batch_size=2)
                            shuffle=True,
                            batch_size=args.per_gpu_train_batch_size,
                            collate_fn=collate_fn, #! collate lists of samples into batches.
                            num_workers=2)
    eval_iter = DataLoader(eval_dataset,
                           shuffle=False,
                           batch_size=args.per_gpu_eval_batch_size,
                           collate_fn=collate_fn,
                           num_workers=2)

    # 用于evaluate
    args.id2label = train_dataset.label_vocab #! label: id 对
    args.num_classes = len(args.id2label)
    metric = ChunkEvaluator(label_list=args.id2label.keys(), suffix=False)

    # model
    model = DuEEEvent_model(args.model_name_or_path, num_classes=args.num_classes)
    model.to(args.device) #! 花时间

    best_f1 = 0
    early_stop = 0
    for epoch, _ in enumerate(range(int(args.num_train_epochs))):
        model.train()#!  only Dropout and BatchNorm care about that flag. By default, this flag is set to True.
        train(args, train_iter, model)
        eval_p, eval_r, eval_f1, eval_loss = evaluate(args, eval_iter, model, metric)
        logger.info(
            "The F1-score is {}".format(eval_f1)
        )
        if eval_f1 > best_f1:
            early_stop = 0
            best_f1 = eval_f1
            logger.info("the best eval f1 is {:.4f}, saving model !!".format(best_f1))
            best_model = copy.deepcopy(model.module if hasattr(model, "module") else model) #! has module是false NOTE
            torch.save(best_model.state_dict(), args.output_model_path) #! 记载的时候 model.load_state_dict
        else:
            early_stop += 1 #如果到达early_stop的容忍范围 模型的eval f1值还没有提升, 就不训练下去了
            if early_stop == args.early_stop:
                logger.info("Early stop in {} epoch!".format(epoch))
                break

if __name__ == '__main__':
    main()
