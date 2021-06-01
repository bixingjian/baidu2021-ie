# baidu2021-ie

## duee-fin

### 数据预处理

- 在`./data/DuEE-Fin/`文件夹下生成 trigger, role, enum, sentence 处理后的 tsv 文件

```bash
python3 duee_fin_data_prepare.py
```

### 训练

- 在`./output/DuEE-Fin/`文件夹下生成 trigger, role, enum, sentence 的`best_model.pkl`文件
- epoch 设置可以在`./utils/finetuning_argparse.py`文件中更改

```bash
CUDA_VISIBLE_DEVICES=3 python run_ner.py --dataset=DuEE-Fin --event_type=trigger --max_len=256 --per_gpu_train_batch_size=2 --per_gpu_eval_batch_size=2 --model_name_or_path=/home/hanqing/bixingjian/pretrained-model/chinese-roberta-wwm-ext-large --linear_learning_rate=1e-4 --early_stop=2

CUDA_VISIBLE_DEVICES=3 python run_ner.py --dataset=DuEE-Fin --event_type=role --max_len=256 --per_gpu_train_batch_size=2 --per_gpu_eval_batch_size=2 --model_name_or_path=/home/hanqing/bixingjian/pretrained-model/chinese-roberta-wwm-ext-large --linear_learning_rate=1e-4 --early_stop=2

CUDA_VISIBLE_DEVICES=3 python run_cls.py --dataset=DuEE-Fin --event_type=enum --max_len=256 --per_gpu_train_batch_size=2 --per_gpu_eval_batch_size=4 --model_name_or_path=/home/hanqing/bixingjian/pretrained-model/chinese-roberta-wwm-ext-large --linear_learning_rate=1e-4 --early_stop=2

```

### 预测

- 利用生成的`best_model.pkl`文件生成`test_result.json`

```bash
CUDA_VISIBLE_DEVICES=3 python predict_ner.py --dataset=DuEE-Fin --event_type=trigger --max_len=400 --per_gpu_eval_batch_size=32 --model_name_or_path=/home/hanqing/bixingjian/pretrained-model/chinese-roberta-wwm-ext-large --fine_tunning_model_path=./output/DuEE-Fin/trigger/best_model.pkl --test_json=./data/DuEE-Fin/sentence/test.json

CUDA_VISIBLE_DEVICES=3 python predict_ner.py --dataset=DuEE-Fin --event_type=role --max_len=400 --per_gpu_eval_batch_size=32 --model_name_or_path=/home/hanqing/bixingjian/pretrained-model/chinese-roberta-wwm-ext-large --fine_tunning_model_path=./output/DuEE-Fin/role/best_model.pkl --test_json=./data/DuEE-Fin/sentence/test.json

CUDA_VISIBLE_DEVICES=3 python predict_cls.py --dataset=DuEE-Fin --event_type=enum --max_len=400 --per_gpu_eval_batch_size=32 --model_name_or_path=/home/hanqing/bixingjian/pretrained-model/chinese-roberta-wwm-ext-large --fine_tunning_model_path=./output/DuEE-Fin/enum/best_model.pkl --test_json=./data/DuEE-Fin/sentence/test.json

```

### 整合预测结果

```bash
CUDA_VISIBLE_DEVICES=3 python duee_fin_postprocess.py --trigger_file=./output/DuEE-Fin/trigger/test_result.json --role_file=./output/DuEE-Fin/role/test_result.json --enum_file=./output/DuEE-Fin/enum/test_result.json --schema_file=./conf/DuEE-Fin/event_schema.json --save_path=./output/DuEE-Fin/duee-fin.json
```
