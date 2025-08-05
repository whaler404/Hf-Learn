# 0. imports
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, ModelConfig, ScriptArguments

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# 1. load a pretrained model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
tokenizer.pad_token = tokenizer.eos_token
value_model = AutoModelForSequenceClassification.from_pretrained(
    "EleutherAI/pythia-70m-deduped", num_labels=1, trust_remote_code=True
).to("cuda:0")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "EleutherAI/pythia-70m-deduped", num_labels=1, trust_remote_code=True
).to("cuda:0")
policy_model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-70m-deduped", trust_remote_code=True
).to("cuda:0")
ref_policy = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-70m-deduped", trust_remote_code=True
).to("cuda:0")

# 2. prepare a dataset
from datasets import load_dataset
dataset = load_dataset(
    "trl-internal-testing/descriptiveness-sentiment-trl-style", split="descriptiveness"
)
dataset = dataset.select(range(100))
eval_samples = 20
train_dataset = dataset.select(range(len(dataset) - eval_samples))
eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
dataset_text_field = "prompt"

def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )
train_dataset = prepare_dataset(train_dataset, tokenizer)
eval_dataset = prepare_dataset(eval_dataset, tokenizer)

# 3. initialize trainer
ppo_config = {"mini_batch_size": 1, "batch_size": 1}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(
    args=config,
    processing_class=tokenizer,
    model=policy_model,
    value_model=value_model,
    ref_model=ref_policy,
    reward_model=reward_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # peft_config=peft_config,
)

ppo_trainer.train()

# # 4. encode a query
# query_txt = "This morning I went to the "
# query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to("cuda:0")

# # 5. generate model response
# generation_kwargs = {
#     "min_length": -1,
#     "top_k": 0.0,
#     "top_p": 1.0,
#     "do_sample": True,
#     "pad_token_id": tokenizer.eos_token_id,
#     "max_new_tokens": 20,
# }
# response_tensor = ppo_trainer.generate([item for item in query_tensor], return_prompt=False, **generation_kwargs)
# response_txt = tokenizer.decode(response_tensor[0])

# # 6. define a reward for response
# # (this could be any reward such as human feedback or output from another model)
# reward = [torch.tensor(1.0).to("cuda:0")]

# # 7. train model with ppo
# train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
# ┃ query                                                                       ┃ model response                                                             ┃ score              ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
# │ The man said, "The Government of India act was passed - the longest act of  │  "You can't do a lot of the time." "You can't do a lot of the time." "You  │ 13.135725021362305 │
# │ Parliament ever - they're going to reprint it, apparently, in two sections. │ can't do a lot of the time." "You can't do a lot of the time." "You can't  │                    │
# │ Our monarch's health hasn't been two hundred percent lately - that old      │ do a lot of the                                                            │                    │
# │ injury's troubling the doctors somewhat - they can't seem to do much.       │                                                                            │                    │
# ├─────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────┤
# │ Thump.  "Who's there?"  I can't see anybody.  The sound is becoming louder  │ .  "You're the one who's here," she cries. "You can't."  "You're the one   │ 12.759991645812988 │
# │ and louder as it closes in on us.  I try to shield Marisol with my body     │ who's here."  "You're the one." "You're the one." "You're the one."        │                    │
# │ from the unseen phantom, but my muscles are frozen in place.                │ "You're the one                                                            │                    │
# ├─────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────┤
# │ After the way Lucas had ignored me this past week, I wasn't sure he had a   │ , I'm sure."                                                               │ 13.274792671203613 │
# │ right to my loyalty. Besides, it felt good, having a gorgeous guy paying    │                                                                            │                    │
# │ attention to me.                                                            │ "You're a great friend."                                                   │                    │
# │                                                                             │                                                                            │                    │
# │ Balthazar stepped a little closer. "I'm going to be glad we met.            │                                                       │                    │
# │                                                                             │                                                                            │                   
# ├─────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────┤
# │ He bounced right back, but she had already skinned out of the cage. He      │ . "You can see the cage," she said. "You can see the cage."                │ 12.785284996032715 │
# │ snagged her around the waist in mid-air as she leaped for the window, never │                                                                            │                    │
# │ mind the stained-glass panes weren't designed to open, and they were 70     │ "You can see the cage," she said. "You can see the cage."                  │                    │
# │ stories up.                                                                 │                                                                            │                    │
# │                                                                             │ "You can see the cage," she said. "You can see                             │                    │
# ├─────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────┤
# │ "Finding ways to push your red button, get you all bothered and reckless."  │ . "You know, I'm a little bit of a fan of the word "fan"," he whispers.    │ 14.447907447814941 │
# │                                                                             │ "You know, I'm a little bit of a fan of the word "fan"." "You know, I'm a  │                    │
# │ Remington turns to me, then he shoves my hair aside and tips my head back   │ little bit of a                                                            │                    │
# │ to study me, like he knows I can barely hear that man's name-much less hear │                                                                            │                    │
# │ them talk about it.                                                         │                                                                            │                    │
# └─────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────┴────────────────────┘

# {
#   "eps": 3,
#   "objective/kl": 34.519561767578125,
#   "objective/entropy": 48.06951904296875,
#   "objective/non_score_reward": -1.725978136062622,
#   "objective/rlhf_reward": 11.789320945739746,
#   "objective/scores": 13.515298843383789,
#   "policy/approxkl_avg": 6.391908168792725,
#   "policy/clipfrac_avg": 0.45224058628082275,
#   "loss/policy_avg": 0.23637911677360535,
#   "loss/value_avg": 2.3484597206115723,
#   "val/clipfrac_avg": 0.4251179099082947,
#   "policy/entropy_avg": 2.289640426635742,
#   "val/ratio": 0.5912855267524719,
#   "val/ratio_var": 0.008559092879295349,
#   "val/num_eos_tokens": 0,
#   "lr": 4.8333333333333334e-05,
#   "episode": 16,
#   "epoch": 0.2
# }