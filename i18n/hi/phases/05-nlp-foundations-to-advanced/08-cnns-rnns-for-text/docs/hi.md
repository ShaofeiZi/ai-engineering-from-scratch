# CNNs और RNNs पाठ के लिए

> संभ्रम n-ग्राम सीखते हैं, पुनरावृत्ति याद आती है, दोनों ध्यान से बदल जाते हैं, दोनों ही सीमित हार्डवेयर पर अभी भी मायने रखते हैं।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time:** ~75 minutes

## समस्या

TF-IDF और Word2Vec एक वर्गीकरण उन पर बनाया गया था नहीं बता सकता `dog bites man` से `man bites dog`शब्द क्रम कभी-कभी संकेत ले जाता है।

ट्रांसफार्मर आने से पहले वास्तुकला के दो परिवारों ने उस अंतर को भर दिया।

**पाठ के लिए संवर्धन जाल (TextCNN).** शब्द एम्बेडेड के अनुक्रमों पर 1D घुमाव लागू करें। चौड़ाई 3 का एक फ़िल्टर एक सीखने योग्य त्रिकोण डिटेक्टर हैः यह तीन शब्दों को फैलाता है और एक स्कोर आउटपुट करता है। बहु-पैटर्न पैटर्न का पता लगाने के लिए विभिन्न चौड़ाई (2, 3, 4, 5) को ढेर करें। एक निश्चित आकार के प्रतिनिधित्व के लिए मैक्स पूल। सपाट, समानांतर, तेजी से।

**आवर्ती जाल (RNN, LSTM, GRU).** एक समय में एक टोकन को संसाधित करें, एक छिपी हुई स्थिति बनाए रखें जो जानकारी को आगे ले जाती है। अनुक्रमिक, मेमोरी-सहन, लचीली इनपुट लंबाई। 2014 से 2017 तक क्रम मॉडलिंग पर हावी रहा, फिर ध्यान हुआ।

यह सबक दोनों को बनाता है, फिर उस असफलता का नाम देता है जिसने ध्यान आकर्षित किया।

## अवधारणा

**TextCNN** (किम, 2014) टोकन एम्बेड हो जाते हैं. एक चौड़ाई-`k` 1D घुमाव एक फिल्टर को लगातार पर स्लाइड करता है `k`-ग्राम एम्बेडमेंट, एक सुविधा नक्शा का उत्पादन। उस नक्शे पर वैश्विक अधिकतम-पूलिंग सबसे मजबूत सक्रियण चुनता है। कई फिल्टर चौड़ाई से अधिकतम-पूल आउटपुट को जोड़ें। एक वर्गीकरण सिर को फ़ीड करें।

एक फिल्टर एक सीखने योग्य n-ग्राम है। अधिकतम-पूलिंग स्थिति-विवर्तनशील है, इसलिए समीक्षा की शुरुआत या मध्य में "अच्छा" एक ही सुविधा को चलाता है। 100 फिल्टर के साथ तीन फिल्टर चौड़ाई प्रत्येक आपको 300 सीखे गए n-ग्राम डिटेक्टर देता है। प्रशिक्षण समानांतर है; कोई अनुक्रमिक निर्भरता नहीं है।

**RNN.** हर समय कदम `t`, छुपी हुई स्थिति `h_t = f(W * x_t + U * h_{t-1} + b)`साझा करें `W`, `U`, `b` समय के पार छिपे हुए राज्य `T` वर्गीकरण के लिए, पूल पार `h_1 ... h_T` (अधिकतम, औसत या अंतिम) ।

सादा RNNs गिरते हुए ग्रेडिएंट का सामना करना पड़ता है। **LSTM** जोड़े गेट जो तय करते हैं कि क्या भूलना है, क्या संग्रहीत करने के लिए, और क्या आउटपुट करने के लिए, स्थिरता gradients के माध्यम से लंबे अनुक्रमों. **GRU** सरलता LSTM दो गेट के लिए; कम पैरामीटर के साथ समान प्रदर्शन करता है।

**द्विदिश RNNs** एक चलाएँ RNN आगे और पीछे, एक साथ छिपे हुए राज्यों. प्रत्येक टोकन का प्रतिनिधित्व दोनों बाएं और दाएं संदर्भ देखता है. टैगिंग कार्यों के लिए आवश्यक है.

```figure
rnn-unroll
```

## इसे बनाओ

### चरण 1: TextCNN में PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

इन `transpose(1, 2)` पुनर्विकृति `[batch, seq_len, embed_dim]` करने के लिए `[batch, embed_dim, seq_len]` क्योंकि `nn.Conv1d` मध्य अक्ष को चैनल के रूप में माना जाता है। इनपुट लंबाई के बावजूद, pooled output फिक्स्ड-साइज है।

### चरण 2: LSTM वर्गीकरणकर्ता

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

क्रम पर अधिकतम पूल, अंतिम स्थिति पूल नहीं। वर्गीकरण के लिए, अधिकतम पूल आमतौर पर अंतिम छिपे हुए राज्य को लेने से बेहतर होता है क्योंकि लंबे अनुक्रम के अंत में जानकारी अंतिम राज्य पर हावी होती है।

### चरण 3: विलुप्त हो रहा ग्रेडिएंट डेमो (अनुभूति)

एक सादा RNN एक खिलौना कार्य पर विचार करेंः भविष्यवाणी करें कि क्या टोकन `A` किसी क्रम में कहीं भी दिखाई दिया। `A` यदि यह 1 की स्थिति में है और अनुक्रम 100 टोकन लंबा है, तो नुकसान से ग्रेडिएंट को आवर्ती वजन के 99 गुणाओं के माध्यम से वापस बहना होगा। यदि वजन 1 से कम है, तो ग्रेडिएंट गायब हो जाता है। यदि 1 से अधिक है, तो यह विस्फोट करता है।

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTMs एक के साथ इसे ठीक **सेल की स्थिति** जो केवल अतिरिक्त बातचीत के साथ नेटवर्क के माध्यम से चलता है (भुलने वाले गेट इसे गुणनशील रूप से स्केल करता है, लेकिन ग्रेडिएंट अभी भी "हाईवे" के साथ बहते हैं) । GRUs कम मापदंडों के साथ कुछ ऐसा ही करें. दोनों आपको 100+ चरण अनुक्रमों के माध्यम से स्थिर प्रशिक्षण देते हैं।

### चरण 4: यह अभी भी पर्याप्त नहीं था

तीन समस्याएं बनी रहीं LSTMs.

1. **अनुक्रमिक बोतल गला.** प्रशिक्षण RNN लंबाई 1000 के अनुक्रम पर 1000 सीरियल आगे/पीछे कदम की आवश्यकता होती है। समय के साथ समानांतर नहीं किया जा सकता है।
2. **एन्कोडर-डेकोडर सेटअप में फिक्स्ड-साइज़ संदर्भ वेक्टर।** डिकोडर केवल एन्कोडर की अंतिम छिपी हुई स्थिति को देखता है, जो पूरे इनपुट पर संपीड़ित होता है। लंबे इनपुट विवरण खो देते हैं। पाठ 09 इस पर सीधे कवर करता है।
3. **दूरी निर्भरता सटीकता सीमा।** LSTMs सादा से बेहतर प्रदर्शन RNNs लेकिन अभी भी 200 से अधिक चरणों में विशिष्ट जानकारी फैलाने के लिए संघर्ष करते हैं।

ध्यान तीनों हल किया. ट्रांसफार्मर पूरी तरह से पुनरावृत्ति गिर गया. पाठ 10 पिवोट है.

## इसका प्रयोग करें

PyTorchहै `nn.LSTM`, `nn.GRU`और `nn.Conv1d` प्रशिक्षण कोड मानक है।

गले लगाने के चेहरे जहाजों पूर्व प्रशिक्षित एम्बेड आप इनपुट परत के रूप में प्लग मेंः

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

जब-यह-फिट-द-सीमा जाँच सूची का उपयोग करें।

- **एज / ऑन डिवाइस इन्फेरेंस।** TextCNN के साथ GloVe यदि आपका डिप्लोय लक्ष्य एक फोन है, तो यह स्टैक है।
- **स्ट्रीमिंग/ऑनलाइन वर्गीकरण।** RNN एक समय में एक टोकन को संसाधित करता है; ट्रांसफार्मर को पूर्ण अनुक्रम की आवश्यकता होती है। वास्तविक समय में आने वाले पाठ के लिए, LSTMs अभी भी जीत.
- **बेसलाइन के लिए छोटे मॉडल।** एक नए कार्य पर तेजी से पुनरावृत्ति। TextCNN 5 मिनट में एक पर CPU.
- **सीमित डेटा के साथ अनुक्रम लेबलिंग।** BiLSTM-CRF (पाठ 06) अभी भी उत्पादन-स्तर है NER 1k-10k लेबल वाले वाक्य के लिए वास्तुकला।

बाकी सब कुछ एक ट्रांसफार्मर में जाता है।

## इसे भेजें

के रूप में सहेजें `outputs/prompt-text-encoder-picker.md`:

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## व्यायाम

1. **- आराम से।** ट्रेन ए TextCNN एक 3 वर्ग खिलौना डेटासेट पर (आप डेटा का आविष्कार करते हैं) सत्यापित करें कि फ़िल्टर चौड़ाई (2, 3, 4) औसत से एक ही चौड़ाई (3) से बेहतर है F1.
2. **मध्यम।** अधिकतम पूल, औसत पूल, और अंतिम राज्य पूल को लागू करने के लिए LSTM एक छोटे से डेटासेट पर तुलना करें; दस्तावेज जो pooling जीतता है और परिकल्पना क्यों।
3. **कठिन.** एक निर्माण BiLSTM-CRF NER टैगर (पाठ 06 और इस एक को संयुक्त) । CoNLL-2003. तुलना करें CRF-alone पाठ 06 से एक तक की आधार रेखा BERT प्रशिक्षण समय, स्मृति और F1.

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| TextCNN | CNN पाठ के लिए | वैश्विक अधिकतम पूल के साथ शब्द एम्बेडिंग पर 1D घुमावदार स्टैक। किम (2014). |
| RNN | आवर्ती नेट | हर समय चरण में अद्यतन छिपा हुआ राज्यः `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | गट RNN | इनपुट / भूल / आउटपुट गेट + एक सेल राज्य जोड़ता है। लंबी अनुक्रमों के माध्यम से स्थिर रूप से ट्रेन करता है। |
| GRU | सरल LSTM | तीन के बजाय दो गेट, समान सटीकता, कम मापदंडों। |
| द्विदिश | दोनों दिशाएँ | Forward + backward RNN प्रत्येक टोकन अपने संदर्भ के दोनों पक्षों को देखता है। |
| विलुप्त हो जाने वाला ग्रेडिएंट | प्रशिक्षण संकेत बंद हो जाता है | दोहराया गया गुणन by <1 सादा में वजन RNNs प्रारंभिक चरण के ग्रेडिएंट को प्रभावी रूप से शून्य बनाता है। |

## आगे पढ़ना

- [किम, वाई (2014) । वाक्य वर्गीकरण के लिए संभ्रांत तंत्रिका नेटवर्क](https://arxiv.org/abs/1408.5882)  TextCNN कागज, आठ पृष्ठ, पढ़ी जा सकती है।
- [Hochreiter, S. और Schmidhuber, J. (1997). लंबी अल्पकालिक स्मृति](https://www.bioinf.jku.at/publications/older/2604.pdf)  LSTM कागज, अप्रत्याशित रूप से स्पष्ट।
- [ओला, सी. (2015). समझ LSTM नेटवर्क](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) जो आरेख बनाए गए LSTMs सभी के लिए सुलभ।
