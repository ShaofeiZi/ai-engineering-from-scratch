# GloVe, FastText, और उपशब्द सम्मिलित

> Word2Vec एक शब्द के लिए एक प्रशिक्षित एम्बेडिंग। GloVe सह-अवसर मैट्रिक्स को कारक बनाया। FastText टुकड़ों को एम्बेड किया। BPE ट्रांसफार्मर से जोड़ दिया गया।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45 minutes

## समस्या

Word2Vec दो खुले प्रश्न छोड़ दिया।

पहले, अनुसंधान की एक समानांतर रेखा थी जो सह-घटना मैट्रिक्स को सीधे कारक बनाती थी (LSA, HAL) ऑनलाइन स्किप-ग्राम अपडेट करने की बजाय। Word2Vecपुनरावर्ती दृष्टिकोण मौलिक रूप से बेहतर था, या अंतर एक कलाकृतियों के रूप में था कैसे दो तरीकों से संभाल गिनती? **GloVe** उत्तर दिया किः एक विचारपूर्वक चुना हानि मैचों या धड़कन के साथ मैट्रिक्स कारक Word2Vec, और प्रशिक्षण के लिए कम लागत है।

दूसरा, किसी भी विधि में शब्दों के लिए एक कहानी नहीं थी जिसे उसने कभी नहीं देखा था। `Zoomer-approved`, `dogecoin`, पिछले सप्ताह का आविष्कार किया गया कोई भी उचित संज्ञा, दुर्लभ जड़ के प्रत्येक घुमावदार रूप. **FastText** यह n-ग्राम वर्णों को एम्बेड करके तय किया गयाः एक शब्द अपने हिस्सों का योग है, जिसमें मॉर्फेम शामिल हैं, इसलिए यहां तक कि शब्द संग्रह से बाहर शब्द एक समझदार वेक्टर प्राप्त करते हैं।

तीसरा, जब ट्रांसफार्मर आए, तो सवाल फिर से बदल गया। शब्द स्तर की शब्दावली लगभग एक मिलियन प्रविष्टियों को कवर करती है; वास्तविक भाषा इससे भी अधिक खुली है। **बाइट-पियर एन्कोडिंग (BPE)** और इसके रिश्तेदारों ने इसे अक्सर उपशब्द इकाइयों की एक शब्दावली को सीखकर हल किया जो सब कुछ कवर करता है। LLM एक उपशब्द टोकन है।

यह सबक तीनों को बताता है, फिर बताता है कि किसको कब तक पहुंचना है।

## अवधारणा

**GloVe (ग्लोबल वेक्टर)** शब्द-शब्द सह-प्रकृति मैट्रिक्स बनाएं `X` जहां `X[i][j]` शब्द कितनी बार `j` शब्द के संदर्भ में दिखाई देता है `i`. रेल वेक्टर ऐसे कि `v_i · v_j + b_i + b_j ≈ log(X[i][j])`वजन कम करने के लिए इतनी बार जोड़े प्रभुत्व नहीं है.

**FastText.** एक शब्द अपने वर्ण n-ग्राम के योग है और शब्द स्वयं है। `where` बन जाता है `<wh, whe, her, ere, re>, <where>`. शब्द वेक्टर उन घटक वेक्टरों का योग है. Word2Vec. लाभः अदृश्य शब्द (`whereupon`) ज्ञात n-ग्राम से बने होते हैं।

**BPE (बाइट-पियर एन्कोडिंग)** प्रत्येक बाइट (या वर्ण) की शब्दावली से शुरू करें। कॉर्पस में प्रत्येक आसन्न जोड़ी को गिनें। सबसे अधिक बार होने वाली जोड़ी को नए टोकन में मिलाएं। `k` परिणामः एक शब्दावली `k + 256` टोकन जहां आवृत्ति क्रम (`ing`, `tion`, `the`) एकल टोकन हैं और दुर्लभ शब्द परिचित टुकड़ों में टूट जाते हैं। प्रत्येक वाक्य कुछ टोकन में बदल जाता है।

```figure
n5-subword-merge
```

## इसे बनाओ

### GloVe: सह-अवसर मैट्रिक्स को फैक्टरीज़ करें

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

दो चलती टुकड़े नाम देने लायक. वजन समारोह `f(x) = (x/x_max)^alpha` बहुत अधिक बार घटते वजन जोड़े (जैसे `(the, and)`) इसलिए वे हानि पर हावी नहीं होते हैं। अंतिम सम्मिलन `W` (केंद्र) और `W_tilde` (संदर्भ) तालिकाओं. दोनों को जोड़ना एक प्रकाशित चाल है जो केवल एक का उपयोग करके बेहतर प्रदर्शन करने की प्रवृत्ति है.

### FastText: उपशब्दों के प्रति जागरूक एम्बेड

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

प्रत्येक शब्द को n-ग्राम (आमतौर पर 3 से 6 वर्ण) के सेट द्वारा दर्शाया जाता है। शब्द एम्बेडिंग उसके n-ग्राम एम्बेडिंग का योग है। स्किप-ग्राम प्रशिक्षण के लिए, इसे प्लग इन करें जहां Word2Vec एक एकल वेक्टर का उपयोग किया।

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

एक अदृश्य शब्द के लिए, आप अभी भी एक वेक्टर मिलता है जब तक कुछ इसके n-ग्राम ज्ञात हैं. `whereupon` शेयर `<wh`, `her`, `ere`और `<where` के साथ `where`, तो दोनों एक दूसरे के करीब भूमि.

### BPE: उपशब्दों की शब्दावली सीखी

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

पहले पुनरावृत्ति सबसे आम आसन्न जोड़ी को मिलाता है। पर्याप्त पुनरावृत्ति के बाद, अक्सर उपशृंखला (`low`, `est`, `tion`) एकल टोकन बन जाते हैं और दुर्लभ शब्द साफ-साफ टूट जाते हैं।

असली GPT / BERT / T5 टोकन बनाने वाले 30k-100k विलय सीखते हैं। परिणामः किसी भी पाठ को ज्ञात IDsनहीं OOV कभी नहीं.

## इसका प्रयोग करें

अभ्यास में, आप शायद ही कभी खुद इन में से किसी को प्रशिक्षित करते हैं. आप पूर्व-प्रशिक्षित चेकपोइंट लोड करते हैं।

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

के लिए BPE-style ट्रांसफार्मर युग में उपशब्द टोकनकरणः

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

इन `Ġ` उपसर्ग शब्द सीमाओं को चिह्नित करता है (a) GPT-2 प्रत्येक आधुनिक टोकन एक BPE वैरिएंट, WordPiece (BERT), या SentencePiece (T5, LLaMA).

### कौन सी चुनना है

| स्थिति | चुनें |
|-----------|------|
| सामान्य प्रयोजन के शब्द वेक्टर, नहीं OOV सहनशीलता की आवश्यकता | GloVe 300d |
| सामान्य प्रयोजन के शब्द वेक्टर, गलत वर्तनी / नवज्ञान / रूपशास्त्रीय रूप से समृद्ध भाषाओं को संभालना चाहिए | FastText |
| किसी भी चीज़ को ट्रांसफार्मर में जाने (प्रशिक्षण या निष्कर्ष) | जो भी टोकन मॉडल के साथ भेजा गया था. कभी नहीं आदान-प्रदान. |
| अपनी भाषा मॉडल को खरोंच से प्रशिक्षित करें | ट्रेन ए BPE या SentencePiece पहले अपने corpus पर टोकनराइज़र |
| रैखिक मॉडल के साथ उत्पादन पाठ वर्गीकरण | अभी भी TF-IDF. पाठ 02 |

## इसे भेजें

के रूप में सहेजें `outputs/skill-embeddings-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## व्यायाम

1. **- आराम से।** दौड़ें `char_ngrams("playing")` और `char_ngrams("played")`. दो n-ग्राम सेटों के जैकार्ड ओवरलैप की गणना करें. आपको पर्याप्त साझा टुकड़े देखने चाहिए (`pla`, `lay`, `play`), यही कारण है कि FastText यह रूपवैज्ञानिक भिन्नताओं के बीच अच्छी तरह से स्थानांतरित करता है।
2. **मध्यम।** विस्तार `learn_bpe` शब्द संग्रह की वृद्धि को ट्रैक करने के लिए। संलयन की संख्या के अनुसार प्रति कर्पस वर्ण टोकन का प्लॉट करें। आपको पहले तेजी से संपीड़न देखना चाहिए, प्रति टोकन लगभग 2-3 वर्णों को समझना चाहिए।
3. **कठिन.** एक 1k-विलय को प्रशिक्षित करें BPE शेक्सपियर के पूर्ण कार्यों पर। सामान्य शब्दों की तुलना दुर्लभ विशेषणों के साथ करें। प्रत्येक शब्द के लिए औसत टोकन का माप करें। आपको जो आश्चर्य हुआ उसे लिखें।

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| सह-प्रभावन मैट्रिक्स | शब्द-शब्द आवृत्ति तालिका | `X[i][j]` = कितनी बार शब्द `j` शब्द के चारों ओर एक खिड़की में दिखाई देता है `i`. |
| उपशब्द | एक शब्द का टुकड़ा | एक वर्ण n-ग्राम (FastText) या सीखा गया टोकन (BPE/WordPiece/SentencePiece). |
| BPE | बाइट-पियर एन्कोडिंग | जब तक शब्दावली लक्ष्य आकार तक नहीं पहुंच जाती तब तक सबसे अधिक बार आसन्न जोड़े का पुनरावर्ती विलय। |
| OOV | शब्दावली से बाहर | मॉडल ने कभी नहीं देखा है। Word2Vec/GloVe असफलता। FastText और BPE इसे संभालो. |
| बाइट स्तर BPE | BPE कच्चे बाइट्स पर | GPT-2शब्दकोश 256 बाइट्स से शुरू होता है, तो कुछ भी कभी नहीं है OOV. |

## आगे पढ़ना

- [पेनिंगटन, सोचर, मैनिंग (2014). GloVe: वर्ड प्रतिनिधित्व के लिए वैश्विक वेक्टर](https://nlp.stanford.edu/pubs/glove.pdf)  GloVe कागज, सात पृष्ठों, अभी भी हानि का सबसे अच्छा व्युत्पन्न।
- [Bojanowski et al. (2017). उपशब्द जानकारी के साथ वर्ड वेक्टर को समृद्ध करना](https://arxiv.org/abs/1607.04606) — FastText.
- [Sennrich, Haddow, Birch (2016) । उपशब्द इकाइयों के साथ दुर्लभ शब्दों का तंत्रिका मशीन अनुवाद](https://arxiv.org/abs/1508.07909) जो पेपर पेश किया गया था BPE आधुनिक NLP.
- [गले लगाना चेहरा टोकनराइज़र सारांश](https://huggingface.co/docs/transformers/tokenizer_summary) कैसे BPE, WordPieceऔर SentencePiece वास्तव में व्यवहार में भिन्न हैं।
