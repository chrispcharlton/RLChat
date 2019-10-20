import os
import slack

from constants import *

from _requirements import *


# NUM_EPISODES = 25  # temp manual set

# SLACK_BOT_TOKEN = os.environ.get('SLACK_BOT_TOKEN')
# SLACK_VERIFICATION_TOKEN = os.environ.get('SLACK_VERIFICATION_TOKEN')

sbt = 'xoxb-785313489105-791757561813-39MWdxaHTMvMFOkA80BSL9Lt'
SLACK_VERIFICATION_TOKEN = 'qRWydCF2TX2p0MOgS6ta9Yw1'


if __name__ == "__main__":

    import os
    import torch

    from seq2seq.models import EncoderRNN, LuongAttnDecoderRNN
    from seq2seq.vocab import Voc

    from reinforcement_learning.environment import Env
    from reinforcement_learning.model import RLGreedySearchDecoder
    from reinforcement_learning import chat

    attn_model = 'dot'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    loadFilename = os.path.join(BASE_DIR, 'data/save', 'rl_model/DQNseq2seq/100_checkpoint.tar')
    checkpoint = torch.load(loadFilename, map_location=device)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    embedding_sd = checkpoint['embedding']
    voc = Voc(checkpoint['voc_dict']['name'])  ##
    voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    encoder.eval()
    decoder.eval()

    policy = RLGreedySearchDecoder(encoder, decoder, voc)
    env = Env(voc)



    # print(f'Agent rl-training for {NUM_EPISODES} episodes...')
    # policy, env, total_rewards, dqn_losses = train(num_episodes=NUM_EPISODES)
    # print('Complete.')

    input_sentence = ''
    env.reset()
    env._state = []

    # rtmclient = slack.RTMClient(token=SLACK_BOT_TOKEN)
    rtmclient = slack.RTMClient(token=sbt)
    print("\nAgent connected and running through Slack client...")

    @slack.RTMClient.run_on(event='message')
    def converse(**payload):
        data = payload['data']
        if 'user' not in data:  # chatbot message in slack
            return

        channel_id = data['channel']
        thread_ts = data['ts']
        user = data['user']
        user_message = data['text']

        try:
            message_tensor = env.sentence2tensor(user_message)
            env.update_state(message_tensor)
            response, tensor = policy.response(env.state)
            env.update_state(tensor)
        except KeyError:
            response = "Error: Encountered unknown word."


        webclient = payload['web_client']
        webclient.chat_postMessage(
            channel=channel_id,
            text=response,
            thread_ts=thread_ts
        )


    rtmclient.start()
