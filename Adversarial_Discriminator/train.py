from _requirements import *
from _config import *
from constants import *

from seq2seq.trainingMethods import GreedySearchDecoder, evaluate

from Adversarial_Discriminator import *
from data.amazon.dataset import AlexaDataset
from data.movie_dialogs.dataset import CornellDataset

from ADEM.model import ADEM
from torch.utils.data import DataLoader

from reinforcement_learning.rl_methods import RLGreedySearchDecoder
from seq2seq import loadAlexaData
from seq2seq.vocab import loadPrepareData, Voc
from seq2seq.chat import *

from seq2seq.models import EncoderRNN, LuongAttnDecoderRNN
from seq2seq import indexesFromSentence, Voc, load_latest_state_dict
from seq2seq.vocab import normalizeString

from reinforcement_learning import rl_methods


def prepare_batch(batch, voc):
    # index_seqs = [indexesFromSentence(voc, normalizeString(u)) + indexesFromSentence(voc, normalizeString(r)) for u, r in
    #               zip(batch.utterance, batch.response)]
    index_seqs = [indexesFromSentence(voc, u) + indexesFromSentence(voc, r) for u, r in
                  zip(batch.utterance, batch.response)]
    lengths = torch.tensor([len(s) for s in index_seqs], device=device, dtype=torch.long) #, device=device)

    seq_tensor = torch.zeros((len(index_seqs), lengths.max()), device=device, dtype=torch.long)
    for idx, (seq, seq_len) in enumerate(zip(index_seqs, lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    target = batch.rating
    target = target[perm_idx].to(device)

    return seq_tensor, target


def trainAdversarialDiscriminatorOnLatestSeq2Seq(model, searcher, voc, data_loader, criterion, optimizer,embedding,save_dir,epoch):


    # Evaluating the searcher


    total_loss = 0

    for i, batch in enumerate(data_loader, 1):
        seq, target  = prepare_batch(batch, voc)
        target[:] = 1 #human conversation example
        output = model(seq)
        loss = criterion(output, target)
        total_loss += loss.item()
        model.zero_grad()
        loss.backward()
        optimizer.step()


        input_sentence_tokens = [indexesFromSentence(voc, normalizeString(u)) for u in batch.utterance]
        input_sentence_tokens.sort(key=len,reverse=True)
        lengths = torch.tensor([len(indexes) for indexes in input_sentence_tokens], dtype=torch.long, device=device)
        maxLength = max(lengths)

        padded_input_sentence_tokens =  [indexes + list(itertools.repeat(PAD_token,maxLength-len(indexes))) for indexes in input_sentence_tokens] #input_sentence_tokens #

        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.tensor(padded_input_sentence_tokens, device=device, dtype=torch.long)
        # Use appropriate device
        input_batch = input_batch.to(device)
        # Decode sentence with searcher

        output_sentence_tokens, scores = searcher(input_batch, MAX_LENGTH)
        compiledSequence = torch.cat([input_batch,output_sentence_tokens], dim=1)


        output =  model(compiledSequence)
        target[:] = 0 # bot conversation sample
        loss = criterion(output, target)
        total_loss += loss.item()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        print('batch {}\n'.format(i))
        if(i % 100 == 0):
            torch.save({
                'iteration': i+epoch,
                'model': model.state_dict(),
                'opt': optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(save_dir, '{}_{}.tar'.format(i+epoch, 'batches')))


    return total_loss

def test_AdversarialDiscriminatorOnLatestSeq2Seq(model, searcher, data_loader, voc):

    print("evaluating trained model ...")
    correctlyHuman = 0
    correctlyBot = 0
    test_data_size = len(data_loader.dataset)

    for i, batch in data_loader:
        seq, target = prepare_batch(batch, voc)
        target[:] = 1
        pred = model.predict(seq)

        correctlyHuman += pred.eq(target.data.view_as(pred)).cpu().sum()

        input_sentence_tokens = [indexesFromSentence(voc, normalizeString(u)) for u in batch.utterance]
        input_sentence_tokens.sort(key=len, reverse=True)
        lengths = torch.tensor([len(indexes) for indexes in input_sentence_tokens], dtype=torch.long, device=device)
        maxLength = max(lengths)
        padded_input_sentence_tokens = [indexes + list(itertools.repeat(PAD_token, maxLength - len(indexes))) for indexes in input_sentence_tokens]

        input_batch = torch.tensor(padded_input_sentence_tokens, device=device, dtype=torch.long)
        output_sentence_tokens, scores = searcher(input_batch, MAX_LENGTH)
        compiledSequence = torch.cat([input_batch, output_sentence_tokens], dim=1)
        target[:] = 0
        pred = model.predict(compiledSequence)
        correctlyBot += pred.eq(target.data.view_as(pred)).cpu().sum()
        print('test batch {}\n'.format(i))

    print('\nTest set accuracy: correctly guess human: {}/{} ({:.0f}%) ; correctly guess bot: {}/{} ({:.0f}%) \n'.format(
        correctlyHuman, test_data_size, 100. * correctlyHuman / test_data_size), correctlyBot, test_data_size, 100. * correctlyBot / test_data_size)



def train():


    N_EPOCHS = 20
    output_size = 2

    ##TODO: shuffle train/test between epochs as some words are exclusive between the pre-defined sets

    save_dir = 'data/save/Adversarial_Discriminator/'

    # corpus_name = "movie_dialogs"
    # corpus = os.path.join("../data", corpus_name)
    # # corpus = os.path.join("C:\\Users\\Christopher\\PycharmProjects\\RLChat","data", corpus_name)
    # # Define path to new file
    # datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    # Load/Assemble voc and pairs
    # save_dir = os.path.join("data", "save")
    # voc, p1 = loadPrepareData(corpus, corpus_name, datafile, save_dir)

    # pairs = list(zip(p1, itertools.repeat(1, len(p1))))

    # datafile2 = os.path.join(corpus, "scrambled_second_sentence_movie_lines.txt")
    # voc2, p2 = loadPrepareData(corpus, corpus_name, datafile2, save_dir)

    # pairs.append(list(zip(p2, itertools.repeat(0, len(p2)))))

    # pairs = random.shuffle(pairs)
    # batches = batch(pairs, BATCH_SIZE)
    attn_model = 'dot'
    # attn_model = 'general'
    # attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1

    seq2seqModel = load_latest_state_dict(savepath=SAVE_PATH_SEQ2SEQ)
    # voc = Voc(seq2seqModel['voc_dict']['name'])
    # voc.__dict__ = seq2seqModel['voc_dict']
    voc, pairs = loadAlexaData()

    embedding = nn.Embedding(voc.num_words, hidden_size)
    model = Adversarial_Discriminator(hidden_size, output_size, embedding)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # attn_model = 'dot'
    # # attn_model = 'general'
    # # attn_model = 'concat'
    # hidden_size = 500
    # encoder_n_layers = 2
    # decoder_n_layers = 2
    # dropout = 0.1

    embedding = nn.Embedding(voc.num_words, hidden_size)

    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    encoder.load_state_dict(seq2seqModel['en'])
    decoder.load_state_dict(seq2seqModel['de'])

    searcher = RLGreedySearchDecoder(encoder, decoder, voc)

    train_data = AlexaDataset('train.json')  # sorry cornell
    # train_data = CornellDataset("formatted_movie_lines.txt", voc, testSetFraction=0.1)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(1, N_EPOCHS + 1):
        loss = trainAdversarialDiscriminatorOnLatestSeq2Seq(model, searcher, voc, train_loader, criterion, optimizer,
                                                            embedding, save_dir, epoch)
        torch.save({
            'iteration': epoch,
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'loss': loss,
            'voc_dict': voc.__dict__,
            'embedding': embedding.state_dict()
        }, os.path.join(save_dir, '{}_{}.tar'.format(epoch, 'epochs')))

    # test_data = CornellDataset("formatted_movie_lines.txt", voc, isTrainSet=False)
    test_data = AlexaDataset('train.json')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    test_AdversarialDiscriminatorOnLatestSeq2Seq(model, searcher, test_loader, voc)