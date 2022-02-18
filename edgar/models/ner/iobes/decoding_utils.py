import torch


def expand_inputs(
        x: torch.Tensor,
        num_beams: int = 1
) -> torch.LongTensor:
    """Expand inputs number of beams times for each sample in batch.
    E.g. num_beams = 2
         Input:  [[1,3,2,10,5], [4,2,8,5,4]]
         Output: [[1,3,2,10,5], [1,3,2,10,5], [4,2,8,5,4], [4,2,8,5,4]]

    """
    batch_size = x.shape[0]

    expanded_indices = torch.arange(batch_size).view(-1, 1).repeat(1, num_beams).view(-1).to(x.device)
    # batch_size * num_beams

    # repeat input num_beam times per sample in batch
    x = x.index_select(0, expanded_indices)
    # (batch_size * num_beams) x max_seq_len
    return x
