import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

EOT = "<" + "|endoftext|" + ">"


class DatasetV1(Dataset):
    """Memory-efficient dataset that indexes into a single contiguous token tensor.

    Instead of storing N separate tensors (one per window), we store one big
    1-D tensor and compute input/target slices on the fly in __getitem__.
    This reduces memory from O(N * context_length) tensor objects to O(1).
    """

    def __init__(self, token_ids, context_length, stride):
        """
        Args:
            token_ids: 1-D torch.LongTensor of all token IDs
            context_length: number of tokens per training sample
            stride: step size between consecutive windows
        """
        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride
        self.n_samples = max(0, (len(token_ids) - context_length) // stride)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.context_length
        input_chunk = self.token_ids[start:end]
        target_chunk = self.token_ids[start + 1 : end + 1]
        return input_chunk, target_chunk


def tokenize_file_chunked(file_path, chunk_size=10 * 1024 * 1024):
    """Tokenize a large text file in chunks to avoid memory spikes.

    Reads the file in `chunk_size`-byte pieces, tokenizes each piece,
    and concatenates all token IDs into a single 1-D LongTensor.

    Args:
        file_path: path to the text file
        chunk_size: bytes to read per chunk (default 10 MB)

    Returns:
        token_ids: 1-D torch.LongTensor
        total_tokens: int
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    all_ids = []
    bytes_read = 0

    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            ids = tokenizer.encode(chunk, allowed_special={EOT})
            all_ids.extend(ids)
            bytes_read += len(chunk.encode("utf-8"))

            if len(all_ids) % 5_000_000 < len(ids):
                print(f"  Tokenized {bytes_read / 1e6:.0f} MB -> {len(all_ids):,} tokens so far...")

    token_tensor = torch.tensor(all_ids, dtype=torch.long)
    print(f"  Tokenization complete: {len(all_ids):,} tokens total")
    return token_tensor, len(all_ids)


def create_dataloader_v1(token_ids, batch_size=4, context_length=256,
                         stride=256, shuffle=True, drop_last=True,
                         num_workers=0):
    """Create a DataLoader from pre-tokenized token IDs.

    Args:
        token_ids: 1-D torch.LongTensor of token IDs
        batch_size: batch size
        context_length: sequence length per sample
        stride: stride between consecutive windows
        shuffle: whether to shuffle
        drop_last: drop last incomplete batch
        num_workers: number of data loading workers

    Returns:
        DataLoader
    """
    dataset = DatasetV1(token_ids, context_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
