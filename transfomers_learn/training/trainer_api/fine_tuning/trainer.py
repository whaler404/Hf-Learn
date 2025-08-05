import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from transformers import TrainingArguments, Trainer

from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=2).to("cuda:0")
model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/SmolLM2-135M", num_labels=2).to("cuda:0")
# 配置 model 的 pad_token_id
model.config.pad_token_id = model.config.eos_token_id
# "Some weights of BertForSequenceClassification were not initialized from the model checkpoint at google-bert/bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']"
# "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."

training_args = TrainingArguments(
    output_dir="./trainer_output/SmolLM2_rotten_tomatoes_classification",
    eval_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    # push_to_hub=True,
)

# from preparing_dataset import small_train, small_eval, 
from preparing_dataset import dataset
from customizing_metric import compute_metrics

trainer = Trainer(
    model=model,
    args=training_args,
    # train_dataset=small_train,
    # eval_dataset=small_eval,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    
)
trainer.train(resume_from_checkpoint=True)