## Pre-processing bounding boxes, ref: https://github.com/uakarsh/latr/blob/1e73c1a99f9a0db4d85177259226148a65556069/src/new_latr/dataset.py#L34

def normalize_box(box, width, height, size=1000):
    """
    Takes a bounding box and normalizes it to a thousand pixels. If you notice it is
    just like calculating percentage except takes 1000 instead of 100.

    Arguments:
        box: A list of bounding box coordinates
        width: The width of the image
        height: The height of the image
        size: The size to normalize to
    Returns:
        A list of normalized bounding box coordinates
    """
    return [
        int(size * (box[0] / width)),
        int(size * (box[1] / height)),
        int(size * (box[2] / width)),
        int(size * (box[3] / height)),
    ]

def get_tokens_with_boxes(bounding_boxes, list_of_words, tokenizer, pad_token_box=[0, 0, 0, 0], max_seq_len=-1, eos_token_box=[0, 0, 1000, 1000]):

    '''
    A function to get the tokens with the bounding boxes
    Arguments:
        bounding_boxes: A list of bounding boxes
        list_of_words: A list of words
        tokenizer: The tokenizer to use
        pad_token_box: The padding token box
        max_seq_len: The maximum sequence length, not padded if max_seq_len is -1
        eos_token_box: The end of sequence token box
    Returns:
        A list of input_ids, bbox_according_to_tokenizer, attention_mask
    '''

    # 2. Performing the semantic pre-processing
    encoding = tokenizer(list_of_words, is_split_into_words=True,
                         add_special_tokens=False)

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Note that, there is no need for bboxes, since the model does not use bbox as feature, so no pre-processing of that
    bbox_according_to_tokenizer = [bounding_boxes[i]
                                   for i in encoding.word_ids()]

    # Truncation of token_boxes + token_labels
    special_tokens_count = 1
    if max_seq_len != -1 and len(input_ids) > max_seq_len - special_tokens_count:
        bbox_according_to_tokenizer = bbox_according_to_tokenizer[: (
            max_seq_len - special_tokens_count)]
        input_ids = input_ids[: (max_seq_len - special_tokens_count)]
        attention_mask = attention_mask[: (max_seq_len - special_tokens_count)]

    ## Adding End of sentence token
    input_ids = input_ids + [tokenizer.eos_token_id]
    bbox_according_to_tokenizer = bbox_according_to_tokenizer + [eos_token_box]
    attention_mask = attention_mask + [1]

    # Padding
    if max_seq_len != -1 and len(input_ids) < max_seq_len:
        pad_length = max_seq_len - len(input_ids)

        input_ids = input_ids + [tokenizer.pad_token_id] * (pad_length)
        bbox_according_to_tokenizer = bbox_according_to_tokenizer + \
            [pad_token_box] * (pad_length)
        attention_mask = attention_mask + [0] * (pad_length)

    return dict(input_ids = input_ids, bboxes = bbox_according_to_tokenizer, attention_mask = attention_mask)