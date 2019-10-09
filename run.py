import argparse

from constants import *



if __name__ == '__main__':
    """
        Main cmdline invocation.
        
        To train seq2seq model: python run.py train seq2seq
        To train adem model: python run.py train adem
        To train model with rl: python run.py train rl
        Run all cells in Colab to train all. 
        
        Replaces need for train_adem_model.py and train_rl_model.py
    """

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='subparser_name')
    train_parser = subparser.add_parser('train')
    train_parser.add_argument(
        'model',
        choices=['seq2seq', 'adem', 'rl'],
    )

    args = parser.parse_args()

    if args.subparser_name == 'train':

        if args.model == 'seq2seq':
            from seq2seq.train import train
            train()

        elif args.model == 'adem':
            from ADEM import *
            from _requirements import *
            from data.amazon.dataset import AlexaDataset
            from _config import *
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


        elif args.model == "rl":
            from reinforcement_learning import train, chat

            policy, env, total_rewards, dqn_losses = train(num_episodes=50)

            # evaluate trained model
            chat(policy, env)






