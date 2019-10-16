from _config import *
from seq2seq.trainingMethods import GreedySearchDecoder, evaluate

from _requirements import *
from seq2seq.models import EncoderRNN, LuongAttnDecoderRNN
from seq2seq import indexesFromSentence, Voc, load_latest_state_dict
from seq2seq.vocab import normalizeString


def prepare_batch(batch, voc):
    index_seqs = [indexesFromSentence(voc, normalizeString(u)) + indexesFromSentence(voc, normalizeString(r)) for u, r in
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

def train_epoch(epoch, model, optimizer, criterion, train_loader, voc):
    total_loss = 0
    for i, sentencePairAndRating in enumerate(train_loader, 1):
        seq, target =  prepare_batch(sentencePairAndRating, voc)
        output = model(seq)
        loss = criterion(output, target)
        total_loss += loss.item()
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(epoch, i * len(train_loader), len(),
                                        100. * i * len(train_loader) / len(), total_loss / i * len(train_loader)))
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
