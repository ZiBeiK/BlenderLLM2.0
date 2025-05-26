import json
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class NameToInstructionDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                prompt = obj.get("name", "").strip()
                completion = obj.get("instruction", "").strip()
                if prompt and completion:
                    self.samples.append((prompt, completion))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, completion = self.samples[idx]
        full_text = f"{prompt}: {completion}"
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        labels = input_ids.clone()

        # Mask out the prompt part
        prompt_ids = self.tokenizer(f"{prompt}:", add_special_tokens=False)["input_ids"]
        labels[:len(prompt_ids)] = -100  # ignore prompt part in loss

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels
        }


class NameToInstructionWithCriteriaDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                name = obj.get("name", "").strip()
                instruction = obj.get("instruction", "").strip()
                criteria = obj.get("criteria", {})
                criteria_str = self.flatten_criteria(criteria)
                prompt = f"Object name: {name}\nCriteria: {criteria_str}\nInstruction:"
                self.samples.append((prompt, instruction))

    def flatten_criteria(self, criteria: dict) -> str:
        flat_parts = []
        for category, subitems in criteria.items():
            for subcat, points in subitems.items():
                for point in points:
                    flat_parts.append(f"{subcat}: {point}")
        return " | ".join(flat_parts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, instruction = self.samples[idx]
        full_text = f"{prompt} {instruction}"
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        labels = input_ids.clone()

        # Mask out the prompt part
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        labels[:len(prompt_ids)] = -100  # ignore prompt part in loss

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels
        }
