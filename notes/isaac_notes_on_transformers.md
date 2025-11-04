https://poloclub.github.io/transformer-explainer/

tokens -> embeddings -> q, k, v embeddings -> split q, k, v across heads -> masked_self_attention[softmax(qk^T/rtdk + M)] dot V -> num_heads refined representations of each token after considering context -> MLP refines representations, expanding size -> more transformer blocks, enriching and enriching token representations -> the final token's output embedding is multiplied by learned weights, creating logits (number indicating how likely token is) -> filtered by sampling strategy, scaled through temperature, converted to probabilities through softmax. 

there are residual connections thaat add layer's input to its output: in attention heads, and mlp - to keep information and learning signals flowing through layers. in gpt-2 there is dropout before... layer normalization keeps mean and variance consistent - before self attention, before mlp, adn before final output

- queries compare with keys to measure relevance, used to weight values
- multiple heads can look for different patterns in the text, in parallel
- the mask is so that tokens can only apply to past tokens and not future tokens. softmax makes rows sum to 1
- MLP consists of linear layer to expand representation size, and GELU which lets small values pass partially large values fully
need to understand the attention mechanism a little bit better
- low temperature makes large logits larger, and vice versa
- some sampling strategies: top-k keeps k most likely, top-p keeps set where total probability is at least p (?)