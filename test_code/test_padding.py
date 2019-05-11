import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

seqs = ['gigantic_string','tiny_str','medium_str']

# make <pad> idx 0
vocab = ['<pad>'] + sorted(set(''.join(seqs)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# make model
embed = nn.Embedding(len(vocab), 10).to(device)
lstm = nn.LSTM(10, 5).to(device)

vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]

print(vectorized_seqs)
# get the length of each seq in your batch
seq_lengths = torch.LongTensor([len(seq) for seq in vectorized_seqs]).to(device)

# dump padding everywhere, and place seqs on the left.
# NOTE: you only need a tensor as big as your longest sequence
seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long().to(device)
for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
	seq_tensor[idx, :seqlen] = torch.LongTensor(seq)


# SORT YOUR TENSORS BY LENGTH!
seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
seq_tensor = seq_tensor[perm_idx]

# utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
# Otherwise, give (L,B,D) tensors
seq_tensor = seq_tensor.transpose(0,1) # (B,L,D) -> (L,B,D)

# embed your sequences
seq_tensor = embed(seq_tensor)

# pack them up nicely
packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy())

# throw them through your LSTM (remember to give batch_first=True here if you packed with it)
packed_output, (ht, ct) = lstm(packed_input)

# unpack your output if required
output, _ = pad_packed_sequence(packed_output)
print (output)

# Or if you just want the final hidden state?
print (ht[-1])

# REMEMBER: Your outputs are sorted. If you want the original ordering
# back (to compare to some gt labels) unsort them
_, unperm_idx = perm_idx.sort(0)
output = output[unperm_idx]
print (output)