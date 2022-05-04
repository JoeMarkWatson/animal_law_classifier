import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
import re
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch.nn.functional as F


pad_tok_id = 1
pad_attn_value = 0


def get_train_test_split(df):
    X, y = df[['link']], df['classification_narrow']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    # fix error in test set
    index_no = X_test.loc[lambda X_test: X_test['link'] == 'https://www.bailii.org/ew/cases/EWHC/Admin/2002/908.html', :].index[0]
    y_test[index_no] = 1
    X_train = X_train.link
    X_test = X_test.link
    return X_train, X_test, y_train, y_test


def get_text_to_encode(jt):
    m = 0
    n = 0
    wt_list = []
    wt_list.append(sent_tokenize(jt))
    wt_list_vals = pd.DataFrame(wt_list).values
    wt_list_vals = wt_list_vals.flatten()
    while m < 199:
        m += len(wt_list_vals[n].split())
        n += 1
    wt_list_vals = [wt for wt in wt_list_vals[n:]]
    return " ".join(wt_list_vals)


def transform_tokenized(tokenized, max_len=4094):
    values = tokenized['input_ids']
    start_tok = values[0]
    end_tok = values[-1]
    text = values[1:-1]
    attention_value = tokenized['attention_mask'][0]

    chunks = []
    for i in range(0, len(text), max_len):
        chunk_text = [start_tok] + text[i:i + max_len] + [end_tok]
        chunks.append({'input_ids': torch.tensor(chunk_text), 'attention_mask': torch.tensor([attention_value] * len(chunk_text))})

    return [chunks[0]]


def join_batch_items(inputs, device='cpu'):
    result = {}
    for k, pad_val in [('input_ids', pad_tok_id), ('attention_mask', pad_attn_value)]:
        items = [item[k] for item in inputs]
        max_len = max(item.shape[0] for item in items)
        items = [F.pad(item, pad=(0, max_len - item.shape[0]), mode='constant', value=pad_val) for item in items]
        result[k] = torch.stack(items).to(device)

    return result


def main():
    # loading data
    df = pd.read_csv('scraped_500_cleaned_text_Apr_JuryIn.csv')
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    with open('train.csv', 'w') as f:
        f.write("\n".join([f"{a},{b}" for a, b in zip(X_train, y_train)]))
    with open('test.csv', 'w') as f:
        f.write("\n".join([f"{a},{b}" for a, b in zip(X_test, y_test)]))

    df['to_embed'] = df.judgment_text.apply(get_text_to_encode)

    model_name = 'bigbird'

    if model_name == 'longformer':
        m = 'allenai/longformer-base-4096'
    elif model_name == 'bigbird':
        m = 'google/bigbird-roberta-large'
    else:
        assert False

    # loading model
    tokenizer = AutoTokenizer.from_pretrained(m)
    model = AutoModel.from_pretrained(m).cuda()

    # apply tokenier to all
    df['tokenized'] = df.to_embed.apply(tokenizer)
    df.tokenized = df.tokenized.apply(transform_tokenized)
    df = df[['link', 'tokenized']].explode('tokenized').reset_index(drop=True)
    
    bs = 2
    df_records = df.to_records(index=False)

    link_to_arrays = defaultdict(list)
    for i in tqdm(list(range(0, len(df_records), bs)), desc='batches'):
        bailii_links, inputs = zip(*df_records[i:i+bs])
        to_input = join_batch_items(inputs, 'cuda')
        sents = tokenizer.batch_decode(to_input['input_ids'])

        output = model(**to_input)
        desired_output = output['pooler_output'].cpu().detach().numpy()

        for l, o in zip(bailii_links, desired_output):
            link_to_arrays[l].append(o)

    for k, v in link_to_arrays.items():
        link_to_arrays[k] = np.mean(v, 0)

    keys, values = zip(*list(link_to_arrays.items()))

    with open(f'{model_name}_urls.txt', 'w') as f:
        f.write("\n".join(keys))

    with open(f'{model_name}_embeds.npy', 'wb') as f:
        np.save(f, np.stack(values))


if __name__ == '__main__':
    main()

