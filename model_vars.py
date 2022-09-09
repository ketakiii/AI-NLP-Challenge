class Model_Vars():
    def __init__(self, latent_dim, input_token_index, target_token_index,
                 num_decoder_tokens, max_decoder_seq_length, max_encoder_seq_length, num_encoder_tokens):
        self.latent_dim = latent_dim
        self.input_token_index = input_token_index
        self.target_token_index = target_token_index
        self.num_decoder_tokens = num_decoder_tokens
        self.max_decoder_seq_length = max_decoder_seq_length
        self.max_encoder_seq_length = max_encoder_seq_length
        self.num_encoder_tokens = num_encoder_tokens
