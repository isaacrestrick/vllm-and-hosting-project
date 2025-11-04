https://poloclub.github.io/transformer-explainer/

--------------------------------------------------------------------------

during decoding q is not a matrix but a vector...

tokens -> embeddings -> q, k, v embeddings -> split q, k, v across heads -> masked_self_attention[softmax(qk^T/rtdk + M)] dot V -> num_heads refined representations of each token after considering context -> MLP refines representations, expanding size -> more transformer blocks, enriching and enriching token representations -> the final token's output embedding is multiplied by learned weights, creating logits (number indicating how likely token is) -> filtered by sampling strategy, scaled through temperature, converted to probabilities through softmax. 

a key fact: last token's output embedding is multiplied by learned weights in final layer -> making logits 

there are residual connections thaat add layer's input to its output: in attention heads, and mlp - to keep information and learning signals flowing through layers. in gpt-2 there is dropout before... layer normalization keeps mean and variance consistent - before self attention, before mlp, adn before final output

- queries compare with keys to measure relevance, used to weight values
- multiple heads can look for different patterns in the text, in parallel
- the mask is so that tokens can only apply to past tokens and not future tokens. softmax makes rows sum to 1
- MLP consists of linear layer to expand representation size, and GELU which lets small values pass partially large values fully
need to understand the attention mechanism a little bit better
- low temperature makes large logits larger, and vice versa
- some sampling strategies: top-k keeps k most likely, top-p keeps set where total probability is at least p (?)

----------------------------------------------------------------------------------------------------------------

Queries, are like questions, is this an adjective
Keys are like adjectives, like this is an adjective, this is not an adjective
Value is how much to adjust embedding by - residual...

the matrix multiplication "an attention pattern" is like a grid of relevance

------------------------------------------------

at the end: "what this token means, given everything before it"

------------------------------------------------

each head is like ok, based on my queries (questions), and my keys (answers), i will update the representation by scaled values (upate understanding)

maybe the FFN is doing something about a single token, as opposed to how the tokens relate to each other?

mostly 2/3 MLP?

Superposition?
