from _config import *

from Adversarial_Discriminator import *
from data.amazon.dataset import AlexaDataset

from torch.utils.data import DataLoader

from reinforcement_learning.model import RLGreedySearchDecoder
from seq2seq.chat import *

from seq2seq.models import EncoderRNN, LuongAttnDecoderRNN
from seq2seq import indexesFromSentence, Voc, load_latest_state_dict


def prepare_batch(batch, voc):
    # index_seqs = [indexesFromSentence(voc, normalizeString(u)) + indexesFromSentence(voc, normalizeString(r)) for u, r in
    #               zip(batch.utterance, batch.response)]
    index_seqs = [indexesFromSentence(voc, u) + indexesFromSentence(voc, r) for u, r in
                  zip(batch.utterance, batch.response)]
    lengths = torch.tensor([len(s) for s in index_seqs], device=device, dtype=torch.long) #, device=device)

    seq_tensor = torch.zeros((len(index_seqs), lengths.max()), device=device, dtype=torch.long)
    for idx, (seq, seq_len) in enumerate(zip(index_seqs, lengths)):
        # seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
        seq_tensor[idx, :seq_len] = torch.tensor(seq, device=device, dtype=torch.long)

    lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    target = batch.rating
    target = target[perm_idx].to(device)

    return seq_tensor, target


def trainAdversarialDiscriminatorOnLatestSeq2Seq(
        model,
        searcher,
        voc,
        data_loader,
        criterion,
        optimizer,
        embedding,
        save_dir,
        epoch
        ):


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

        input_sentence_tokens = [indexesFromSentence(voc, u) for u in batch.utterance]

        input_sentence_tokens.sort(key=len,reverse=True)
        lengths = torch.tensor([len(indexes) for indexes in input_sentence_tokens], dtype=torch.long, device=device)
        maxLength = max(lengths)

        padded_input_sentence_tokens = [indexes + list(itertools.repeat(PAD_token,maxLength-len(indexes))) for indexes in input_sentence_tokens] #input_sentence_tokens #

        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.tensor(padded_input_sentence_tokens, device=device, dtype=torch.long)
        # Use appropriate device
        input_batch = input_batch.to(device)
        # Decode sentence with searcher

        output_sentence_tokens, scores = searcher(input_batch, MAX_LENGTH)
        compiledSequence = torch.cat([input_batch, output_sentence_tokens], dim=1).to(device)


        output =  model(compiledSequence)
        target[:] = 0 # bot conversation sample
        loss = criterion(output, target)
        total_loss += loss.item()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        # print('batch {}\n'.format(i))

        if i % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(epoch, i * len(batch[0]), len(data_loader.dataset),
                                        100. * i * len(batch[0]) / len(data_loader.dataset), total_loss / i * len(batch)))
        # if(i % 100 == 0):
        #     torch.save({
        #         'iteration': i+epoch,
        #         'model': model.state_dict(),
        #         'opt': optimizer.state_dict(),
        #         'loss': loss,
        #         'voc_dict': voc.__dict__,
        #         'embedding': embedding.state_dict()
        #     }, os.path.join(save_dir, '{}_{}.tar'.format(i+epoch, 'batches')))


    return total_loss

def test_AdversarialDiscriminatorOnLatestSeq2Seq(model, searcher, data_loader, voc):

    print("evaluating trained model ...")
    correctlyHuman = 0
    correctlyBot = 0
    test_data_size = len(data_loader.dataset)

    for batch in data_loader:
        seq, target = prepare_batch(batch, voc)
        target[:] = 1
        pred = model.predict(seq)

        correctlyHuman += pred.eq(target.data.view_as(pred)).cpu().sum()

        # input_sentence_tokens = [indexesFromSentence(voc, normalizeString(u)) for u in batch.utterance]
        input_sentence_tokens = [indexesFromSentence(voc, u) for u in batch.utterance]

        input_sentence_tokens.sort(key=len, reverse=True)
        lengths = torch.tensor([len(indexes) for indexes in input_sentence_tokens], dtype=torch.long, device=device)
        maxLength = max(lengths)
        padded_input_sentence_tokens = [indexes + list(itertools.repeat(PAD_token, maxLength - len(indexes))) for indexes in input_sentence_tokens]

        input_batch = torch.tensor(padded_input_sentence_tokens, device=device, dtype=torch.long)
        output_sentence_tokens, scores = searcher(input_batch, MAX_LENGTH)
        compiledSequence = torch.cat([input_batch, output_sentence_tokens], dim=1).to(device)
        target[:] = 0
        pred = model.predict(compiledSequence)
        correctlyBot += pred.eq(target.data.view_as(pred)).cpu().sum()
        # print('test batch {}\n'.format(i))
        print('\nTest set accuracy: correctly guess human: {}/{} ({:.0f}%) ; correctly guess bot: {}/{} ({:.0f}%)'.format(
            correctlyHuman, test_data_size, (100. * correctlyHuman / test_data_size), correctlyBot, test_data_size, 100. * correctlyBot / test_data_size))


def train():
    N_EPOCHS = 10
    output_size = 2
    save_dir = 'data/save/Adversarial_Discriminator/'

    attn_model = 'dot'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1

    seq2seqModel = load_latest_state_dict(savepath=SAVE_PATH_SEQ2SEQ)
    voc = Voc.from_dataset(AlexaDataset(rare_word_threshold=0))

    embedding = nn.Embedding(voc.num_words, hidden_size)
    model = Adversarial_Discriminator(hidden_size, output_size, embedding)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    encoder.load_state_dict(seq2seqModel['en'])
    decoder.load_state_dict(seq2seqModel['de'])
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    searcher = RLGreedySearchDecoder(encoder, decoder, voc)

    train_data = AlexaDataset('train.json', rare_word_threshold=3)  # sorry cornell
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = AlexaDataset('train.json', rare_word_threshold=3)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    for epoch in range(1, N_EPOCHS + 1):
        loss = trainAdversarialDiscriminatorOnLatestSeq2Seq(model, searcher, voc, train_loader, criterion, optimizer,
                                                            embedding, save_dir, epoch)

        if epoch % 3 == 0:
            torch.save({
                'iteration': epoch,
                'model': model.state_dict(),
                'opt': optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(save_dir, '{}_{}.tar'.format(epoch, 'epochs')))

        test_AdversarialDiscriminatorOnLatestSeq2Seq(model, searcher, test_loader, voc)
