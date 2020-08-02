import dill
filename = 'globalsave.pkl'
dill.load_session(filename)


X_train, y_train = get_data(train_data_generator)
X_test, y_test = get_data(test_data_generator)

# create a vocab to unique idx mapping and vice-versa
 
PADDING_TOKEN = "PAD"
UNKNOWN_TOKEN = "UNK"

def replace_by_counts(tokens, max_count, replace_by):
    '''
        Replaces tokens with count<=max_counts by the token 'replace_by'
    '''
    counts = dict(Counter(tokens)).items()
    vocab = [token if count > max_count else replace_by for token, count in counts]
    return list(set(vocab))

tokens = list(chain.from_iterable(X_train))
vocab = replace_by_counts(tokens, 4, UNKNOWN_TOKEN)
vocab.append(PADDING_TOKEN)
idx2word = dict(enumerate(set(vocab)))
word2idx = {value: key for key, value in idx2word.items()}

labels = list(chain.from_iterable(y_train))
labels.append(PADDING_TOKEN)
idx2label = dict(enumerate(set(labels)))
label2idx = {value: key for key, value in idx2label.items()}

# calculating weight for class weights. These weights are used in the cross-entropy loss
count_label = dict(Counter(chain.from_iterable(y_train)))
inverse_counts = {key: 1. / value for key, value in count_label.items()}
sum_inverse = np.sum([count for _, count in inverse_counts.items()])
inverse_normalized = {key: value / sum_inverse for key, value in inverse_counts.items()}
weights = np.array([0.3 + inverse_normalized[idx2label[i]] for i in range(len(idx2label))])
weights /= np.sum(weights)


# checks
def check2(X, y):
    for _x, _y in zip(X, y):
        assert len(_x) == len(_y)


check2(X_train, y_train)
check2(X_test, y_test)

# from our analysis previously, we take all the sentences with length 40 or below. For the ones less than 40, we pad. For the rest, we trim
MAX_LEN = 40


def trim(X, y, max_len=MAX_LEN):
    sequence = []
    for i in range(len(X)):
        if len(X[i]) >= max_len:
            X[i] = X[i][:max_len]
            y[i] = y[i][:max_len]
            sequence.append(max_len)
        else:
            sequence.append(len(X[i]))
    return X, y, sequence


X_train, y_train, train_seq = trim(X_train, y_train, MAX_LEN)
X_test, y_test, test_seq = trim(X_test, y_test, MAX_LEN)


# checks
def check3(X, y, sequence):
    assert len(X) == len(y)
    assert len(X) == len(sequence)
    for seq, tags in zip(X, y):
        assert len(tags) <= MAX_LEN
        assert len(seq) == len(tags)


check3(X_train, y_train, train_seq)
check3(X_test, y_test, test_seq)

inverse_normalized[PADDING_TOKEN] = 0


#


# create a vocab to unique idx mapping and vice-versa

tokens = list(chain.from_iterable(X_train))
vocab = replace_by_counts(tokens, 4, UNKNOWN_TOKEN)
vocab.append(PADDING_TOKEN)
idx2word = dict(enumerate(set(vocab)))
word2idx = {value: key for key, value in idx2word.items()}

labels = list(chain.from_iterable(y_train))
labels.append(PADDING_TOKEN)
idx2label = dict(enumerate(set(labels)))
label2idx = {value: key for key, value in idx2label.items()}


# replace tokens by their indices from the dictionary. Same for labels

def unk_map(x, token2idx, unk):
    '''
    Replace tokens by unk token idx if they are not in the vocabulary
    '''
    idx = []
    for word in x:
        if word not in token2idx.keys():
            idx.append(unk)
        else:
            idx.append(token2idx[word])
    return idx


unk_idx = word2idx["UNK"]

X_train = list(map(lambda x: unk_map(x, word2idx, unk_idx), X_train))
y_train = list(map(lambda x: [label2idx[word] for word in x], y_train))

X_test = list(map(lambda x: unk_map(x, word2idx, unk_idx), X_test))
y_test = list(map(lambda x: [label2idx[word] for word in x], y_test))


# creating pytorch dataset for iteration and generating batches
class NERDataset(Dataset):
    def __init__(self, X, y, sequence):
        self.X = X
        self.y = y
        self.seq = sequence
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.seq[idx]


def pad(X, Y, seq, pad_idx, pad_label, max_len):
    '''
    add pad_idx to tokens and pad_label to labels is correspinding PAD tokens
    '''
    X_padd = []
    y_padd = []
    for x, y in zip(X, Y):
        x_len = max_len - len(x)
        X_padd.append(x + [pad_idx] * x_len)
        y_padd.append(y + [pad_label] * x_len)
    X_padd = torch.LongTensor(X_padd)
    y_padd = torch.LongTensor(y_padd)
    seq = torch.LongTensor(seq)
    return X_padd, y_padd, seq


X_train, y_train, train_seq = pad(X_train, y_train, train_seq, word2idx["PAD"], label2idx["PAD"], MAX_LEN)

X_test, y_test, test_seq = pad(X_test, y_test, test_seq, word2idx["PAD"], label2idx["PAD"], MAX_LEN)

train_dataset = NERDataset(X_train, y_train, train_seq)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = NERDataset(X_test, y_test, test_seq)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

gc.collect()

"""Now our data is ready for deep learning

## 5.2 Model Building - LSTM
"""


class NERLstm(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, padding_idx, max_len, num_layers):
        super(NERLstm, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_dim, tagset_size)
        self.padding_idx = padding_idx
        self.max_len = max_len
        self.tagset_size = tagset_size
        self.padding_idx = padding_idx
    
    def forward(self, X, seq):
        embeddings = self.word_embeddings(X)
        packed_input = pack_padded_sequence(embeddings, seq, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.linear(output)
        return output


config = {
    'embedding_dim': 128,
    'hidden_dim': 1024,
    'vocab_size': 100,
    'tagset_size': len(idx2label),
    'padding_idx': 0,
    'max_len': 3,
    "num_layers": 2
}


# test
def test_model():
    '''
    Testing the model
    '''
    test_model = NERLstm(**config)
    test_X = torch.LongTensor([[1, 5, 3], [4, 0, 0]])
    test_y = torch.LongTensor([[1, 2, 3], [4, 5, 5]])
    test_sequence = torch.LongTensor([3, 1])
    test_output = test_model(test_X, test_sequence)
    print(test_output.shape)


test_model()


def loss(true, pred, pad_idx, target_size, max_len, weights=None, device="cpu"):
    '''
        Calculate loss without taking PAD loss into account
    '''
    batch_size = pred.shape[0]
    max_batch = pred.shape[1]
    weights = torch.Tensor(weights)
    weights = weights.to(device)
    loss_fn = nn.CrossEntropyLoss(reduction='none', weight=weights)
    pred = pred[:, :max_batch, :]
    true = true[:, :max_batch].contiguous()
    true = true.view(-1)
    pred = pred.view(-1, target_size)
    loss = loss_fn(pred, true)
    mask = true != pad_idx
    mask = mask.view(-1).type(torch.FloatTensor).to(device)
    mask /= mask.shape[0]
    return torch.dot(mask, loss) / torch.sum(mask)


def generate_predictions(model, X, seq, device="cuda"):
    X = X.to(device)
    seq = seq.to(device)
    pred = model(X, seq)
    pred_labels = torch.argmax(pred, 2)
    pred_labels = pred_labels.view(-1)
    return pred_labels


def accuracy(model, X, seq, y_true, pad_idx, device):
    y_true = y_true.view(-1)
    y_pred = generate_predictions(model, X, seq, device)
    mask = y_true != pad_idx
    mask = mask.type(torch.FloatTensor)
    matches = y_pred == y_true
    matches = matches.type(torch.FloatTensor)
    correct = torch.dot(matches, mask)
    total = len(y_pred)
    accuracy = correct.item() / total
    return accuracy


# testing masked loss
# loss(test_y, test_output, word2idx["PAD"], config["tagset_size"], config["max_len"])

"""## 5.3 Training Loop"""

EMBEDDING_DIM = 300
HIDDEN_DIM = 512
VOCAB_SIZE = len(word2idx)
TAGSET_SIZE = len(idx2label)
BATCH_SIZE = 128
LEARNING_RATE = 10e-3
EPOCHS = 22
NUM_LAYERS = 3

config = {
    'embedding_dim': EMBEDDING_DIM,
    'hidden_dim': HIDDEN_DIM,
    'vocab_size': VOCAB_SIZE,
    'tagset_size': TAGSET_SIZE,
    'padding_idx': word2idx["PAD"],
    'max_len': MAX_LEN,
    'num_layers': NUM_LAYERS
}

if torch.cuda.is_available:
    device = "cuda"
else:
    device = "cpu"

model = NERLstm(**config)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loss = []
test_loss = []
test_accuracy = []
train_accuracy = []

for i in range(EPOCHS):
    epoch_loss = 0
    model = model.train()
    LEARNING_RATE *= 0.8
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for j, (X, y, sequence) in enumerate(train_loader):
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)
        # print(X.shape, y.shape, sequence.shape)
        sequence = sequence.to(device)
        pred = model(X, sequence)
        loss_value = loss(y, pred, config['padding_idx'], config['tagset_size'], config["max_len"], weights=weights, device=device)
        loss_value.backward()
        # print(j ,": ",round(loss_value.item(),2))
        epoch_loss += loss_value.item()
        optimizer.step()
        del X, y, sequence
        torch.cuda.empty_cache()
    
    model = model.eval()
    
    train_loss.append(round(epoch_loss, 3))
    
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    test_seq = test_seq.to(device)
    
    y_pred = model(X_test, test_seq)
    test_epoch_loss = loss(y_test, y_pred, config['padding_idx'], config['tagset_size'], config["max_len"], weights, device)
    test_epoch_loss = round(test_epoch_loss.item(), 3)
    test_loss.append(test_epoch_loss)
    
    test_epoch_acc = round(accuracy(model, X_test, test_seq, y_test, config["padding_idx"], device), 3)
    test_accuracy.append(test_epoch_acc)
    
    print("-----------Epoch: {}-----------".format(i + 1))
    print("Loss:\ntrain:{0}\ntest:{1}\n".format(round(epoch_loss, 2), test_epoch_loss))
    print("Accuracy:\ntest:{0}\n".format(test_epoch_acc))

plt.plot(test_accuracy)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.plot(np.array(test_loss) / len(X_test))
# X,y,sequence = next(iter(train_loader))

plt.plot(np.array(train_loss) / len(X_train))


def F1_scores(y_true, y_pred, idx, pad_idx):
    y_pred = torch.argmax(y_pred, 2)
    y_true = y_true.to("cpu").type(torch.LongTensor).numpy().reshape(-1)
    y_pred = y_pred.to("cpu").type(torch.LongTensor).numpy().reshape(-1)
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    p = 0
    n = 0
    for i, true in enumerate(y_true):
        if true != pad_idx:
            pred = y_pred[i]
            if true == idx:
                p += 1
                if pred == idx:
                    tp += 1
                else:
                    fn += 1
            else:
                n += 1
                if pred == idx:
                    fp += 1
                else:
                    tn += 1
    
    precision = tp / (tp + fp + 0.0001)
    recall = tp / (tp + fn + 0.0001)
    f1 = 2 * precision * recall / (precision + recall + 0.0001)
    
    return round(f1, 3), round(precision, 3), round(recall, 3)


total = 0
for tag, idx in label2idx.items():
    f1, prec, rec = F1_scores(y_test, y_pred, label2idx[tag], label2idx["PAD"])
    print(tag + " stats: " + "precision: ", prec, " recall: ", rec, " F1: ", f1)
    total += f1
print("------ Average: {} ------".format(total / len(label2idx)))


# analysis
def analyze(y_pred, y_true, X_test):
    y_pred = torch.argmax(y_pred, 2)
    y_true = y_true.to("cpu").type(torch.LongTensor).numpy().reshape(-1)
    y_pred = y_pred.to("cpu").type(torch.LongTensor).numpy().reshape(-1)
    X_test = X_test.to("cpu").type(torch.LongTensor).numpy().reshape(-1)
    where_incorrect = y_true != y_pred
    incorrect_idxes = np.where(where_incorrect == 1)[0]
    incorrect_tokens = X_test[incorrect_idxes]
    return dict(Counter(incorrect_tokens))


incorrect_dict = analyze(y_pred, y_test, X_test)
incorrect_dict = sorted(incorrect_dict.items(), key=lambda x: x[1], reverse=True)
for idx, count in incorrect_dict[:20]:
    print(idx2word[idx], " ----> ", count)


def predict_tags(sentence, model=model, word2idx=word2idx,
                 idx2word=idx2word, label2idx=label2idx, idx2label=idx2label):
    tokens = sentence.lower().split()
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    length = len(tokens)
    tokens_idx = []
    for token in stemmed_tokens:
        if token not in word2idx.keys():
            tokens_idx.append(word2idx["UNK"])
        else:
            tokens_idx.append(word2idx[token])
    
    tokens_idx = torch.LongTensor(tokens_idx).unsqueeze(0)
    sequence = torch.LongTensor([length])
    predictions = generate_predictions(model, tokens_idx, sequence)
    for token, label in zip(tokens, predictions):
        print(token, " ----> ", idx2label[label.item()])


predict_tags("Swades starring shahrukh khan describes the state of rural india very well")

predict_tags("Amir khan plays mahavir phogat in the real life based film dangal")

predict_tags("sholay is said to be one of the greates film of its time")
