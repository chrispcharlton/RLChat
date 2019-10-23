from _requirements import *
from seq2seq.vocab import PAD_token, EOS_token
from data.amazon.dataset import standardise_sentence

def indexesFromSentence(voc, sentence):
    sentence = standardise_sentence(sentence)
    try:
        return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
    except KeyError:
        print(sentence)

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch], device=device)
    padList = zeroPadding(indexes_batch)
    # padVar = torch.LongTensor(padList, device=device)
    padVar = torch.tensor(padList, device=device, dtype=torch.long)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    # mask = torch.BoolTensor(mask, device=device)  # ByteTensor of type uint8 raising error
    padVar = torch.tensor(padList, device=device, dtype=torch.long)
    mask = torch.tensor(mask, device=device, dtype=torch.bool)
    # padVar = torch.LongTensor(padList, device=device) # Legacy constructor doesn't support GPU
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp.t(), lengths, output, mask, max_target_len
