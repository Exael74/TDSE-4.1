import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader, len(dataset)

if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("--- Experiment Results ---")
    
    # Experiment 1: max_length=4, stride=4 (No overlap)
    dl1, len1 = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=4, shuffle=False)
    print(f"Ex 1: max_length=4, stride=4 -> Samples: {len1}")

    # Experiment 2: max_length=4, stride=1 (High overlap)
    dl2, len2 = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    print(f"Ex 2: max_length=4, stride=1 -> Samples: {len2}")
    
    # Experiment 3: max_length=4, stride=2 (Partial overlap)
    dl3, len3 = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=2, shuffle=False)
    print(f"Ex 3: max_length=4, stride=2 -> Samples: {len3}")
