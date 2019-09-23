from ADEM import *
from _requirements import *
from data.amazon.dataset import AlexaDataset
from _config import *
from ADEM.model import ADEM
from torch.utils.data import DataLoader
from seq2seq import loadAlexaData

N_EPOCHS = 1
BATCH_SIZE = 64
output_size = 5

##TODO: shuffle train/test between epochs as some words are exclusive between the pre-defined sets

save_dir = '.\\data\\amazon\\models\\adem\\'

voc, pairs = loadAlexaData()

train_data = AlexaDataset('train.json')
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = AlexaDataset('test_freq.json')
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

embedding = nn.Embedding(voc.num_words, hidden_size)
model = ADEM(hidden_size, output_size, embedding)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, N_EPOCHS + 1):
    loss = train_epoch(epoch, model, optimizer, criterion, train_loader, voc)

    test_epoch(model, test_loader, voc)

    torch.save({
        'iteration': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'loss': loss,
        'voc_dict': voc.__dict__,
        'embedding': embedding.state_dict()
    }, os.path.join(save_dir, '{}_{}.tar'.format(epoch, 'epochs')))