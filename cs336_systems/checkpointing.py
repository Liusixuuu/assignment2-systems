import torch
from torch.utils.checkpoint import checkpoint


# Problem (gradient_checkpointing): activation checkpointing over the TransformerBlocks
# of a BasicsTransformerLM. Only the blocks are checkpointed; embedding, final norm and
# lm_head run normally.
def run_blocks(blocks, x):
    for block in blocks:
        x = block(x)
    return x


# (b) One level of checkpointing: split the N blocks into consecutive groups of
# `group_size`. Each group keeps only its input alive until backward, then recomputes
# its blocks' residuals right before its own backward.
#   peak activations ~ (N / group_size) * checkpoint_size + group_size * block_residuals
def grouped_checkpoint(blocks, x, group_size: int):
    for start in range(0, len(blocks), group_size):
        x = checkpoint(run_blocks, blocks[start:start + group_size], x, use_reentrant=False)
    return x


# (a) Nested checkpointing: checkpoint each half, and recursively each half of that.
# At any point in backward only one checkpoint per recursion level is alive plus one
# block's residuals -> O(log N) peak activation memory, but each block is recomputed
# once per level -> O(N log N) forward compute.
def recursive_checkpoint(blocks, x):
    if len(blocks) == 1:
        return blocks[0](x)
    mid = len(blocks) // 2
    x = checkpoint(recursive_checkpoint, blocks[:mid], x, use_reentrant=False)
    x = checkpoint(recursive_checkpoint, blocks[mid:], x, use_reentrant=False)
    return x


def checkpointed_forward(model, x, group_size: int = 0, recursive: bool = False):
    """Same as BasicsTransformerLM.forward, but with the blocks checkpointed."""
    x = model.token_embeddings(x)
    if recursive:
        x = recursive_checkpoint(model.layers, x)
    else:
        x = grouped_checkpoint(model.layers, x, group_size)
    x = model.ln_final(x)
    return model.lm_head(x)
