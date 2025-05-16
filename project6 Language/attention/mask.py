for i, token in enumerate(inputs.input_ids[0]):
        if token == mask_token_id:
            return i
    return None


def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    attention_score = attention_score.numpy()
    return (round(attention_score * 255), round(attention_score * 255), round(attention_score * 255))


def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.

    For each attention layer, one diagram should be generated for each
    attention head in the layer. Each diagram should include the list of
    `tokens` in the sentence. The filename for each diagram should
    include both the layer number (starting count from 1) and head number
    (starting count from 1).
    """
    for i, layer in enumerate(attentions):
        for k in range(len(layer[0])):
            layer_number = i + 1
            head_number = k + 1
            generate_diagram(
                layer_number,
                head_number,
                tokens,
                attentions[i][0][k]
            )
