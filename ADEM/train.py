from _requirements import *
from seq2seq import indexesFromSentence


def prepare_batch(batch, voc):
    index_seqs = [indexesFromSentence(voc, u) + indexesFromSentence(voc, r) for u, r in
                  zip(batch.utterance, batch.response)]
    lengths = torch.LongTensor([len(s) for s in index_seqs], device=device)

    seq_tensor = torch.zeros((len(index_seqs), lengths.max()), device=device).long()
    for idx, (seq, seq_len) in enumerate(zip(index_seqs, lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq, device=device)

    lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    target = batch.rating
    target = target[perm_idx]

    return seq_tensor, target

def train_epoch(epoch, model, optimizer, criterion, data_loader, voc):
    total_loss = 0
    for i, batch in enumerate(data_loader, 1):
        seq, target = prepare_batch(batch, voc)
        output = model(seq)
        loss = criterion(output, target)
        total_loss += loss.item()
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(epoch, i * len(batch[0]), len(data_loader.dataset),
                                        100. * i * len(batch[0]) / len(data_loader.dataset), total_loss / i * len(batch)))
    return total_loss

def test_epoch(model, data_loader, voc):

    print("evaluating trained model ...")
    correct = 0
    train_data_size = len(data_loader.dataset)

    for batch in data_loader:
        seq, target = prepare_batch(batch, voc)
        pred = model.predict(seq)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, train_data_size, 100. * correct / train_data_size))
