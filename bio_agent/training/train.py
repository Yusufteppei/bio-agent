from transformers import Trainer, TrainingArguments
from utils import collate_fn, tokenize

class SFTTrainer:

    def __init__(self, model, tokenizer, dataset, experiment="", base_model="", num_train_epochs=3, learning_rate=3e-4):

        self.trainer = Trainer(
            model=model,
            args=self.training_args(),
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=collate_fn,
        )
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        

    def training_args(self):

        return TrainingArguments(
            output_dir="experiments/qwen2.5-0.5b/exp002",
            learning_rate=2e-4,
            per_device_train_batch_size=8,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
            remove_unused_columns=False,
        )

    def train(self):
        self.trainer.train()

    def save(self):
        self.trainer.save_model()

    

trainer = SFTTrainer(

    model=model,

    dataset=train_dataset,

    tokenizer=tokenizer,

    data_collator=collate_fn,
)

trainer.train()

trainer.save_model(
    "experiments/qwen2.5-0.5b/exp002/adapter"
)