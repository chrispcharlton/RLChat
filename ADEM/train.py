from _requirements import *
from _config import *
from seq2seq import indexesFromSentence


def prepare_batch(batch, voc):
    index_seqs = [indexesFromSentence(voc, u) + indexesFromSentence(voc, r) for u, r in
                  zip(batch.utterance, batch.response)]
    # lengths = torch.LongTensor([len(s) for s in index_seqs], device=device)
    lengths = torch.tensor([len(s) for s in index_seqs], device=device, dtype=torch.long)

    seq_tensor = torch.zeros((len(index_seqs), lengths.max()), device=device).long()

    for idx, (seq, seq_len) in enumerate(zip(index_seqs, lengths)):
        # seq_tensor[idx, :seq_len] = torch.LongTensor(seq, device=device)
        seq_tensor[idx, :seq_len] = torch.tensor(seq, device=device, dtype=torch.long)

    lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    target = batch.rating
    target = target[perm_idx].to(device)

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

    print("\nEvaluating trained model ...")
    correct = 0
    train_data_size = len(data_loader.dataset)

    for batch in data_loader:
        seq, target = prepare_batch(batch, voc)
        pred = model.predict(seq)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, train_data_size, 100. * correct / train_data_size))


def main():
    # from ADEM import *
    from data.amazon.dataset import AlexaDataset
    # from _config import *
    from ADEM.model import ADEM
    from torch.utils.data import DataLoader
    from seq2seq import loadAlexaData

    N_EPOCHS = 5
    BATCH_SIZE = 256
    output_size = 5

    ##TODO: shuffle train/test between epochs as some words are exclusive between the pre-defined sets

    voc, pairs = loadAlexaData()

    train_data = AlexaDataset('train.json')
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    test_data = AlexaDataset('test_freq.json')
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    embedding = nn.Embedding(voc.num_words, hidden_size)
    model = ADEM(hidden_size, output_size, embedding).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print('Training...')
    for epoch in range(1, N_EPOCHS + 1):
        loss = train_epoch(epoch, model, optimizer, criterion, train_loader, voc)

        torch.save({
            'iteration': epoch,
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'loss': loss,
            'voc_dict': voc.__dict__,
            'embedding': embedding.state_dict()
        }, os.path.join(BASE_DIR, SAVE_PATH_ADEM, '{}_{}.tar'.format(epoch, 'epochs')))

        test_epoch(model, test_loader, voc)
