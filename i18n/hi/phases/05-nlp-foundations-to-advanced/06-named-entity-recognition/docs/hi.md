# नामित इकाई पहचान

> जब तक आप अस्पष्ट सीमाओं, घोंसले हुए संस्थाओं और डोमेन जार्गोन से निपट नहीं लेते तब तक नामों को बाहर निकालें।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75 minutes

## समस्या

"Apple गूगल को उसके iPhone खोज सौदा US." पांच संस्थाएं: Apple (ORG), गूगल (ORG), iPhone (PRODUCT), खोज सौदा (शायद), US (GPE) एक अच्छा NER सिस्टम सही प्रकार के साथ उन्हें सभी निकालता है. एक बुरा एक चूक जाता है iPhone, भ्रमित करता है Apple फल के साथ Apple कंपनी और लेबल "US" के रूप में PERSON.

NER यह हर संरचित निष्कर्षण पाइपलाइन के नीचे काम का घोड़ा है. पुनरीक्षण विश्लेषण, अनुपालन लॉग स्कैन, चिकित्सा रिकॉर्ड अनामिकता, खोज क्वेरी समझ, चैटबॉट प्रतिक्रियाओं के लिए ग्राउंडिंग, कानूनी अनुबंध निष्कर्षण. आप इसे कभी नहीं देख सकते हैं; आप हमेशा इस पर निर्भर करते हैं।

यह सबक शास्त्रीय मार्ग पर चलता है (नियम आधारित, HMM, CRF) आधुनिक में बदल गया (BiLSTM-CRFप्रत्येक चरण से पहले की एक विशिष्ट सीमा का समाधान होता है। पैटर्न सबक है।

## अवधारणा

**BIO टैगिंग** (या BILOU) इकाई निकासी को अनुक्रम लेबलिंग समस्या में बदल देता है। प्रत्येक टोकन को `B-TYPE` (संस्था की शुरुआत), `I-TYPE` (आंतरिक इकाई), या `O` (किसी भी इकाई के बाहर) ।

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

बहु-टोकन संस्थाओं की श्रृंखलाः `New B-GPE`, `York I-GPE`, `City I-GPE`. एक मॉडल जो समझता है BIO मनमानी स्पैन निकाल सकते हैं।

वास्तुकला प्रगति:

- **नियम आधारित।** रेजेक्स + गजटियर खोजें. ज्ञात संस्थाओं पर उच्च सटीकता, नए पर शून्य कवरेज।
- **HMM.** छिपे हुए मार्कोव मॉडल, दिए गए टोकन टैग की उत्सर्जन संभावना, टैग-टू-टैग संक्रमण संभावना, विटरबी डिकोड, लेबल किए गए डेटा पर प्रशिक्षित।
- **CRF.** सशर्त यादृच्छिक क्षेत्र. HMM लेकिन भेदभावपूर्ण, तो आप मनमाने ढंग से विशेषताएं मिश्रण कर सकते हैं (शब्द आकार, पूंजीकरण, पड़ोसी शब्दों) अभी भी 2026 में कम संसाधन तैनाती के लिए क्लासिक उत्पादन कार्यघड़ी.
- **BiLSTM-CRF.** हाथ से बने होने के बजाय तंत्रिका विशेषताएं। LSTM वाक्य दोनों दिशाओं में पढ़ता है, CRF शीर्ष पर परत लगातार टैग अनुक्रमों को लागू करता है।
- **ट्रांसफार्मर आधारित।** ठीक-ठीक BERT एक टोकन वर्गीकरण सिर के साथ. सबसे अच्छी सटीकता. सबसे गणना.

```figure
ner-bio-tagging
```

## इसे बनाओ

### चरण 1: BIO टैगिंग सहायक

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### चरण 2: हस्तनिर्मित विशेषताएं

शास्त्रीय (गैर-न्यूरल) के लिए NERखेल के लिए उपयोगी विशेषताएंः

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` रिटर्न `xXxxxx`. `word_shape("USA-2024")` रिटर्न `XXX-dddd`. पूँजीकरण पैटर्न उचित संज्ञाओं के लिए उच्च संकेत हैं.

### चरण 3: सरल नियम आधारित + शब्दकोश आधार

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

उत्पादन गजटर्स में विकिपीडिया और DBpedia. यह जानकारी अच्छी है।`Apple` यह भयानक है। यही कारण है कि सांख्यिकीय मॉडल जीत गए।

### चरण 4: CRF चरण (स्केच, पूर्ण इंप्लिकेशन नहीं)

पूर्ण CRF 50 लाइनों में शून्य से संभावना सिद्धांत के आधार के बिना नहीं है. `sklearn-crfsuite` इसके बजायः

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` और `c2` है L1 और L2 नियमितता। `all_possible_transitions=True` मॉडल अवैध अनुक्रमों को सीखने देता है (उदाहरण के लिए, `I-ORG` बाद में `O`) के लिए संभावना नहीं है, जो कि कैसे एक CRF प्रवर्तन BIO बिना आप प्रतिबंध लिखने के लिए एकीकरण।

### चरण 5: क्या एक BiLSTM-CRF जोड़ता है

इनपुटः टोकन एम्बेडमेंट (GloVe या fastText). LSTM बाएं से दाएं और दाएं से बाएं पढ़ता है. CRF आउटपुट परत। CRF अभी भी टैग-अनुक्रम सुसंगतता को लागू करता है; LSTM हाथ से बने चित्रों को शिक्षित चित्रों से बदल देता है।

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

के लिए CRF परत, उपयोग `torchcrf.CRF` (पीआईपी स्थापित pytorch-crf) हाथ से निर्मित पर लाभ CRF यह मापने योग्य है लेकिन अपेक्षा से छोटा है जब तक आपके पास दसियों हज़ार लेबल वाले वाक्य नहीं हैं।

## इसका प्रयोग करें

spaCy उत्पादन श्रेणी के जहाज NER बॉक्स से बाहर.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

नोटिस `iPhone` लेबल `ORG` बजाय `PRODUCT` — spaCyछोटे मॉडल में उत्पाद इकाई कवरेज कमजोर है।`en_core_web_lg`) बेहतर है। ट्रांसफार्मर मॉडल (`en_core_web_trf`) और भी बेहतर है।

गले लगाना चेहरा के लिए BERT-based NER:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` इसके बिना, आप टोकन स्तर लेबल मिलता है और खुद को मिलाया जाना है.

### LLM-based NER (2026 विकल्प)

शून्य शॉट और कुछ शॉट LLM NER अब कई डोमेन पर ठीक-ठीक मॉडल के साथ प्रतिस्पर्धी है, और नाटकीय रूप से बेहतर जब लेबल डेटा दुर्लभ है।

- **शून्य शॉट संकेत.** दे दी LLM इकाई प्रकारों की सूची और एक उदाहरण योजना। JSON आउटपुट. बॉक्स से बाहर काम करता है; सटीकता नए डोमेन पर मध्यम है.
- **ZeroTuneBio-style प्रेरित करना।** कार्य को उम्मीदवार निष्कर्षण → अर्थ व्याख्या → निर्णय → पुनः जांच में विघटित करें। एक बहु-चरण संकेत (एक शॉट नहीं) जैव चिकित्सा पर सटीकता को काफी बढ़ाता है NER. कानूनी, वित्तीय और वैज्ञानिक क्षेत्रों के लिए भी यही पैटर्न लागू होता है।
- **गतिशील प्रलोभन के साथ RAG.** प्रत्येक निष्कर्ष कॉल के लिए एक छोटे से टिप्पणी बीज सेट से सबसे समान लेबल किए गए उदाहरणों को पुनर्प्राप्त करें; उड़ान पर कुछ शॉट प्रॉम्प्ट बनाएं। 2026 बेंचमार्क में, यह लिफ्ट करता है GPT-4 जैव चिकित्सा NER F1 स्थैतिक उत्तेजना से 11-12% अधिक।
- **प्रति इकाई प्रकार का विघटन।** लंबे दस्तावेजों के लिए, एक एकल कॉल जो एक ही समय में सभी इकाई प्रकारों को निकालता है, लंबाई बढ़ने के साथ याद करना खो देता है। प्रति इकाई प्रकार एक निष्कर्षण पास चलाएं। उच्च निष्कर्ष लागत, काफी अधिक सटीकता। यह नैदानिक नोट्स और कानूनी अनुबंधों के लिए मानक पैटर्न है।

2026 से उत्पादन सिफारिशः एक LLM प्रशिक्षण डेटा एकत्र करने से पहले शून्य शॉट बेस लाइन। F1 पर्याप्त अच्छा है कि आप कभी भी ठीक करने की जरूरत नहीं है।

### जहां शास्त्रीय NER अभी भी जीतता है

यहां तक कि LLMs उपलब्ध, शास्त्रीय NER जीतता है जबः

- विलंबता बजट 50ms से नीचे है।
- आपके पास हजारों लेबल वाले उदाहरण हैं और आपको 98%+ की आवश्यकता है F1.
- डोमेन एक स्थिर ontology है जहां एक पूर्व प्रशिक्षित CRF या BiLSTM स्थानांतरण अच्छा है।
- नियामक प्रतिबंधों के लिए एक स्थानीय, गैर-जनकारी मॉडल की आवश्यकता होती है।

### जहां यह टूट जाता है

- **डोमेन शिफ्ट.** CoNLL-trained NER कानूनी अनुबंध पर एक राजपत्रकार से भी बदतर प्रदर्शन करता है.
- **घोंसले हुए संस्थाएं।** "बैंक ऑफ अमेरिका टॉवर" एक ही समय में एक ORG और एक FACILITY. मानक BIO आप घोंसले की जरूरत है NER (मल्टी-पास या स्पैन आधारित मॉडल) ।
- **लंबी संस्थाएं।** "संयुक्त राज्य अमेरिका के संघीय जमा बीमा निगम. " टोकन स्तर के मॉडल कभी कभी इस विभाजित. उपयोग `aggregation_strategy` या प्रक्रिया के बाद।
- **स्पायर प्रकार के लोग।** चिकित्सा NER DRUG_BRAND, ADVERSE_EVENT जैसे लेबल, DOSE. सामान्य प्रयोजन के मॉडल का कोई विचार नहीं है। BioBERT वहाँ से शुरू होने वाले बिंदु हैं।

## इसे भेजें

के रूप में सहेजें `outputs/skill-ner-picker.md`:

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## व्यायाम

1. **- आराम से।** कार्यान्वयन `bio_to_spans` (अर्थात् `spans_to_bio`) और 10 वाक्य पर वापसी-यात्रा सुसंगतता की जांच करें।
2. **मध्यम।** स्क्लेयरन-क्रफसूट को प्रशिक्षित करें CRF ऊपर पर CoNLL-2003 अंग्रेजी NER डेटा सेट प्रति इकाई रिपोर्ट F1 उपयोग `seqeval`. विशिष्ट परिणाम: ~ 84 F1.
3. **कठिन.** ठीक-ठीक `distilbert-base-cased` एक डोमेन-विशिष्ट पर NER डेटासेट (चिकित्सा, कानूनी या वित्तीय) की तुलना करें spaCy डेटा रिसाव जांच दस्तावेज और क्या आप आश्चर्यचकित किया है लिखने के लिए.

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| NER | निकालने के नाम | प्रकार के साथ लेबल टोकन सीमा (PERSON, ORG, GPE, DATE, ...). |
| BIO | टैगिंग योजना | `B-X` शुरू होता है, `I-X` जारी है, `O` बाहर. |
| BILOU | बेहतर BIO | जोड़ें `L-X` (अंतिम), `U-X` (इकाई) स्वच्छ सीमाओं के लिए। |
| CRF | संरचित वर्गीकरण | मॉडल लेबल के बीच संक्रमण, न केवल उत्सर्जन. |
| घोंसले हुए NER | ओवरलैप करने वाली संस्थाएं | एक अवधि एक इकाई है जो इसके उप-क्षेत्र से भिन्न है। BIO यह व्यक्त नहीं कर सकते। |
| इकाई स्तर F1 | उचित NER मेट्रिक | भविष्यवाणी की अवधि सही अवधि के साथ मेल खाती है. F1 सटीकता को अतिरंजित करता है। |

## आगे पढ़ना

- [Lample et al. (2016) नामित इकाई पहचान के लिए तंत्रिका वास्तुकला](https://arxiv.org/abs/1603.01360)  BiLSTM-CRF कागज, कैनोनिक।
- [डेव्लिन और अन्य (2018) BERT: गहरे द्विदिशात्मक ट्रांसफार्मरों की पूर्व-प्रशिक्षण](https://arxiv.org/abs/1810.04805) टोकन-वर्गीकरण पैटर्न को पेश करता है जो मानक बन गया।
- [spaCy भाषाई विशेषताएं  नामित संस्थाएं](https://spacy.io/usage/linguistic-features#named-entities) प्रत्येक विशेषता के लिए व्यावहारिक संदर्भ `Doc.ents` और `Span`.
- [अनुसूची](https://github.com/chakki-works/seqeval) सही मीट्रिक लाइब्रेरी। हमेशा इसका इस्तेमाल करें।
