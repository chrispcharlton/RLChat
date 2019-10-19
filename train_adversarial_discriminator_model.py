from Adversarial_Discriminator import *
from _requirements import *
from data.amazon.dataset import AlexaDataset
from data.movie_dialogs.dataset import CornellDataset
from _config import *
from torch.utils.data import DataLoader

from reinforcement_learning.model import RLGreedySearchDecoder
from seq2seq.models import *
from seq2seq.vocab import loadPrepareData, Voc
from Adversarial_Discriminator.train import trainAdversarialDiscriminatorOnLatestSeq2Seq, test_AdversarialDiscriminatorOnLatestSeq2Seq

N_EPOCHS = 20
output_size = 2

##TODO: shuffle train/test between epochs as some words are exclusive between the pre-defined sets

save_dir = 'data/save/Adversarial_Discriminator/'


corpus_name = "movie_dialogs"
corpus = os.path.join("../data", corpus_name)
# corpus = os.path.join("C:\\Users\\Christopher\\PycharmProjects\\RLChat","data", corpus_name)
# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

# Load/Assemble voc and pairs
#save_dir = os.path.join("data", "save")
#voc, p1 = loadPrepareData(corpus, corpus_name, datafile, save_dir)

#pairs = list(zip(p1, itertools.repeat(1, len(p1))))

#datafile2 = os.path.join(corpus, "scrambled_second_sentence_movie_lines.txt")
#voc2, p2 = loadPrepareData(corpus, corpus_name, datafile2, save_dir)

#pairs.append(list(zip(p2, itertools.repeat(0, len(p2)))))

#pairs = random.shuffle(pairs)
#batches = batch(pairs, BATCH_SIZE)

seq2seqModel = load_latest_state_dict()
voc = Voc(seq2seqModel['voc_dict']['name'])
voc.__dict__ = seq2seqModel['voc_dict']



embedding = nn.Embedding(voc.num_words, hidden_size)
model = Adversarial_Discriminator(hidden_size, output_size, embedding)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()



attn_model = 'dot'
# attn_model = 'general'
# attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1





embedding = nn.Embedding(voc.num_words, hidden_size)

encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

encoder.load_state_dict(seq2seqModel['en'])
decoder.load_state_dict(seq2seqModel['de'])

searcher = RLGreedySearchDecoder(encoder, decoder, voc)


train_data = CornellDataset( "formatted_movie_lines.txt", voc,testSetFraction=0.1)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)



for epoch in range(1, N_EPOCHS + 1):
    loss = trainAdversarialDiscriminatorOnLatestSeq2Seq(model, searcher, voc, train_loader, criterion, optimizer,embedding,save_dir,epoch)
    torch.save({
        'iteration': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'loss': loss,
        'voc_dict': voc.__dict__,
        'embedding': embedding.state_dict()
    }, os.path.join(save_dir, '{}_{}.tar'.format(epoch, 'epochs')))


test_data = CornellDataset("formatted_movie_lines.txt", voc, isTrainSet=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
test_AdversarialDiscriminatorOnLatestSeq2Seq(model, searcher, test_loader, voc)