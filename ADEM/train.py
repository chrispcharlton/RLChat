from _requirements import *
from _config import *
from seq2seq import indexesFromSentence
from ADEM import *
from data.amazon.dataset import AlexaDataset
from _config import *
from ADEM.model import ADEM
from torch.utils.data import DataLoader
from seq2seq import Voc
from constants import *


def prepare_batch(batch, voc):
    index_seqs = [indexesFromSentence(voc, u) + indexesFromSentence(voc, r) for u, r in
                  zip(batch.utterance, batch.response)]
    lengths = torch.tensor([len(s) for s in index_seqs], device=device, dtype=torch.long)

    seq_tensor = torch.zeros((len(index_seqs), lengths.max()), device=device).long()

    for idx, (seq, seq_len) in enumerate(zip(index_seqs, lengths)):
        seq_tensor[idx, :seq_len] = torch.tensor(seq, device=device, dtype=torch.long)

    lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    target = batch.rating.type(torch.long)
    target = target[perm_idx].to(device)

    return seq_tensor, target


def train_epoch(epoch, model, optimizer, criterion, data_loader, voc):
    total_loss = []
    for i, batch in enumerate(data_loader, 1):
        seq, target = prepare_batch(batch, voc)
        output = model(seq)
        loss = criterion(output, target)
        total_loss.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(epoch, i * len(batch[0]), len(data_loader.dataset),
                                        100. * i * len(batch[0]) / len(data_loader.dataset), sum(total_loss) / i * len(batch)))
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


def train(epochs=2000):
    BATCH_SIZE = 256
    output_size = 5

    ##TODO: shuffle train/test between epochs as some words are exclusive between the pre-defined sets

    voc = Voc.from_dataset(AlexaDataset(rare_word_threshold=0))

    train_data = AlexaDataset('train.json', rare_word_threshold=0)
    train_data.add_pairs_from_json('valid_freq.json')
    train_data.add_scrambled_training_data(rating='Poor')
    train_data.add_scrambled_training_data(rating='Not Good')
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    test_data = AlexaDataset('test_freq.json', rare_word_threshold=3)
    test_data.add_scrambled_training_data(rating='Poor')
    test_data.add_scrambled_training_data(rating='Not Good')
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    embedding = nn.Embedding(voc.num_words, hidden_size)
    model = ADEM(hidden_size, output_size, embedding).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()

    log = open(os.path.join(BASE_DIR, SAVE_PATH_ADEM, 'adem_training.csv'.format('epochs')), 'a')
    log.write('batch,loss\n')

    print('Training...')


    for epoch in range(1, epochs + 1):

        loss = train_epoch(epoch, model, optimizer, criterion, train_loader, voc)
        for i, l in enumerate(loss):
            log.write(','.join([str(i+((epoch-1) * len(train_loader))),str(l)]))
            log.write('\n')
        if epoch % adem_save_every == 0:
            torch.save({
                'iteration': epoch,
                'model': model.state_dict(),
                'opt': optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(BASE_DIR, SAVE_PATH_ADEM, '{}_{}.tar'.format(epoch, 'epochs')))

        test_epoch(model, test_loader, voc)

if __name__ == '__main__':
    train()