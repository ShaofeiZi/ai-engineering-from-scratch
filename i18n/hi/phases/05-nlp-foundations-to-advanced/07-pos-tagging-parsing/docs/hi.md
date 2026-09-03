# POS टैगिंग और सिंटैक्टिक पार्सिंग

> व्याकरण कुछ समय के लिए फैशन से बाहर था। LLM पाइपलाइन संरचित निकासी को मान्य करने की जरूरत थी, और यह वापस आया।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## समस्या

पाठ 01 वादा किया कि lemmatization एक भाग-से-भाषण टैग की जरूरत है. `running` एक क्रिया है, एक lemmatizer इसे कम नहीं कर सकते `run`. बिना जाने `better` एक विशेषण है, यह कम नहीं किया जा सकता है `good`.

इस वादे से एक पूरा उपक्षेत्र छिपा है। भाषण के भाग टैगिंग व्याकरणिक श्रेणियों को सौंपता है। वाक्य की वृक्ष संरचना को पुनः प्राप्त करता है। कौन सा शब्द कौन सा संशोधित करता है, कौन सा क्रिया कौन से तर्क को नियंत्रित करता है। NLP और फिर गहन सीखने ने उन्हें एक पूर्व प्रशिक्षित ट्रांसफार्मर के ऊपर टोकन वर्गीकरण के काम में ढह दिया, और शोध समुदाय आगे बढ़ गया।

हर संरचित निकासी पाइपलाइन अभी भी उपयोग करता है POS और आश्रय पेड़ हुड के नीचे. LLM-generated JSON प्रश्न-उत्तर प्रणाली निर्भरता पार्स का उपयोग करके क्वेरी को बिखेरती है। मशीन अनुवाद गुणवत्ता मूल्यांकनकर्ता पार्स पेड़ों के संरेखण की जांच करते हैं।

यह सबक टैगसेट, बेसलाइन और उस बिंदु को पेश करता है जहां आप खरोंच से लागू करना बंद कर देते हैं और कॉल करते हैं spaCy.

## अवधारणा

**POS टैगिंग** प्रत्येक टोकन को व्याकरणिक श्रेणी के साथ लेबल करता है। **पेन ट्रीबैंक (PTB)** टैगसेट अंग्रेजी डिफ़ॉल्ट है। 36 टैग के साथ अंतर आकस्मिक पाठक को परेशान लगता हैः `NN` एकल संज्ञा, `NNS` बहुवचन संज्ञा, `NNP` विशेष संज्ञा एकल, `VBD` क्रिया अतीत समय, `VBZ` क्रिया 3rd person singular present, आदि। **सार्वभौमिक निर्भरता (UD)** टैगसेट अधिक कठोर (17 टैग) और भाषा-अज्ञानी है; यह क्रॉस-लिंग्वेज वर्क के लिए डिफ़ॉल्ट बन गया।

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**संश्लेषण विश्लेषण** दो प्रमुख शैलीः

- **निर्वाचन क्षेत्र विश्लेषण.** संज्ञा वाक्यांश, क्रिया वाक्यांश, पूर्वावचन वाक्यांश एक दूसरे के अंदर घोंसले। आउटपुट गैर-अंत श्रेणी का एक पेड़ है (NP, VP, PP) के साथ शब्दों के रूप में पत्ते।
- **निर्भरता विश्लेषण.** प्रत्येक शब्द में एक ही शीर्षक शब्द होता है जिस पर यह निर्भर करता है, जिसे व्याकरणिक संबंध के साथ लेबल किया जाता है। आउटपुट एक ऐसा पेड़ है जहां प्रत्येक किनारा एक (मुख, निर्भर, संबंध) ट्रिपल है।

निर्भरता विश्लेषण 2010 के दशक में जीता क्योंकि यह भाषाओं में स्पष्ट रूप से सामान्यीकरण करता है, विशेष रूप से मुक्त शब्द क्रम वाले।

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

```figure
pos-tagger
```

```figure
dependency-arcs
```

## इसे बनाओ

### चरण 1: सबसे अधिक बार टैग की आधार रेखा

सबसे बेवकूफ POS प्रत्येक शब्द के लिए, भविष्यवाणी टैग वह प्रशिक्षण में सबसे अधिक बार था।

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

ब्राउन कॉर्पस पर, यह मूल रेखा लगभग 85% सटीकता तक पहुंचती है। अच्छा नहीं, लेकिन तल जिसके नीचे कोई गंभीर मॉडल नहीं गिरना चाहिए।

### चरण 2: बिग्राम HMM टैगर

अनुक्रम की संयुक्त संभावना का मॉडलः

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

दो तालिकाएंः संक्रमण संभावनाएं (पूर्ववर्ती टैग दिए गए टैग) , उत्सर्जन संभावनाएं (शब्द दिए गए टैग) । लैपलेस चिकनाई के साथ गणना से दोनों का अनुमान लगाएं। विटरबी (टैग जाली पर गतिशील प्रोग्रामिंग) के साथ डिकोड करें।

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

बिग्राम HMM ब्राउन पर ~93% सटीकता हिट। 85% से 93% की छलांग ज्यादातर संक्रमण संभावनाओं है `DET NOUN` आम है और `NOUN DET` दुर्लभ है।

### चरण 3: क्यों आधुनिक टैगर्स इसे हरा

संक्रमण + उत्सर्जन संभावनाएं स्थानीय हैं। वे यह नहीं पकड़ सकते कि `saw` "मैंने एक पीतल खरीदा" में एक संज्ञा है लेकिन "मैंने फिल्म देखी" में एक क्रिया है। CRF स्वैच्छिक विशेषताओं (परसतिश, शब्द आकार, शब्द से पहले और बाद, शब्द स्वयं) के साथ ~ 97%। BiLSTM-CRF या ट्रांसफार्मर ~98%+ तक पहुंचता है।

इस कार्य पर सीमा नोटर मतभेद द्वारा निर्धारित की जाती है। मानव नोटर्स पेन ट्रीबैंक पर लगभग 97% समय पर सहमत होते हैं। 98% से अधिक मॉडल शायद परीक्षण सेट से अधिक फिट होते हैं।

### चरण 4: निर्भरता विश्लेषण स्केच

पूर्ण निर्भरता को खरोंच से विश्लेषण करने के लिए दायरे से बाहर है; कैनोनिक पाठ्यपुस्तक उपचार जुराफस्की और मार्टिन में है। दो शास्त्रीय परिवारों को जानने के लिएः

- **संक्रमण आधारित** पार्सर (आर्क-आकांक्षी, आर्क-स्टैंडर्ड) एक शिफ्ट-रिड्यूस पार्सर की तरह काम करते हैंः वे टोकन पढ़ते हैं, उन्हें स्टैक पर स्थानांतरित करते हैं, और उन कार्यों को कम करते हैं जो आर्क बनाते हैं। लालची डिकोडिंग तेजी से होती है। क्लासिक कार्यान्वयन MaltParser. आधुनिक तंत्रिका संस्करणः चेन और मैनिंग का संक्रमण आधारित पार्सर।
- **ग्राफ आधारित** पारसर्स (एस्नर के एल्गोरिथ्म, डोज़ैट-मैनिंग बीएफिन) हर संभव सिर-निर्भर किनारे को स्कोर करते हैं और अधिकतम विस्तार वाले पेड़ का चयन करते हैं। धीमी लेकिन अधिक सटीक।

अधिकांश आवेदन के लिए, कॉल करें spaCy:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

पढ़िए `dep` स्तंभ नीचे से ऊपर और वाक्य की व्याकरण संरचना गिर जाता है।

## इसका प्रयोग करें

प्रत्येक उत्पादन NLP पुस्तकालय जहाज POS और एक मानक पाइपलाइन के हिस्से के रूप में निर्भरता पारसर।

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`) तेजी से, सटीक, टोकनकरण के साथ एकीकृत + NER + lemmatization. `token.tag_` (पिन), `token.pos_` (UD), `token.dep_` (निर्भरता संबंध) ।
- **स्टैनफोर्ड NLP (stanza)**स्टैनफोर्ड के उत्तराधिकारी CoreNLP. 60 से अधिक भाषाओं पर अत्याधुनिक।
- **शांत**. ट्रांसफार्मर आधारित, अच्छा UD सटीकता।
- **NLTK**. `pos_tag`उपयोग करने योग्य, धीमी, पुरानी, शिक्षण के लिए अच्छा।

### जहां यह अभी भी 2026 में मायने रखता है

- **लम्मिटिकेशन।** पाठ 01 आवश्यकताएं POS हमेशा सही ढंग से लेमेटिज़ करने के लिए।
- **से संरचित निकासी LLM आउटपुट।** सत्यापित करें कि उत्पन्न वाक्य व्याकरणिक प्रतिबंधों का सम्मान करता है (जैसे, विषय-क्रिया अनुबंध, आवश्यक संशोधन) ।
- **पहलू आधारित भावना।** निर्भरता पार्स आपको बताता है कि कौन सा विशेषण कौन सा संज्ञा बदलता है।
- **बहुत समझदारी.** "वेस एंडरसन द्वारा निर्देशित और बिल मरे के साथ फिल्में" विश्लेषण के माध्यम से संरचित प्रतिबंधों में विघटित हो जाती हैं।
- **पार भाषा हस्तांतरण।** UD टैग और निर्भरता संबंध भाषा-अज्ञानी हैं, जिससे नई भाषाओं का शून्य-शॉट संरचनात्मक विश्लेषण संभव हो जाता है।
- **कम कम्प्यूटिंग पाइपलाइनें।** यदि आप एक ट्रांसफार्मर भेज नहीं सकते हैं, POS + निर्भरता विश्लेषण + गजट्रेटर आपको आश्चर्यजनक रूप से दूर ले जाता है।

## इसे भेजें

के रूप में सहेजें `outputs/skill-grammar-pipeline.md`:

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## व्यायाम

1. **- आराम से।** छोटे टैग किए गए कॉर्पस पर सबसे अधिक बार टैग की आधार रेखा का उपयोग करना (जैसे, NLTKब्राउन उपसमूह), पकड़ने वाले वाक्य पर सटीकता मापें। ~ 85% परिणाम की पुष्टि करें।
2. **मध्यम।** बिग्राम को प्रशिक्षित करें HMM उपरोक्त और प्रति टैग सटीकता / याद करने की रिपोर्ट। HMM सबसे भ्रमित?
3. **कठिन.** उपयोग spaCyएक 1000 वाक्य के नमूने से विषय-कार्य-वस्तु त्रिगुट निकालने के लिए निर्भरता विश्लेषण। 50 मैन्युअल रूप से लेबल किए गए त्रिगुट पर मूल्यांकन करें। दस्तावेज जहां निष्कर्षण विफल रहता है (अक्सर निष्क्रिय, समन्वय, और हटाए गए विषय) ।

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| POS टैग | शब्द का प्रकार | व्याकरण श्रेणी। PTB 36 है; UD 17 है। |
| पेन ट्रीबैंक | मानक टैगसेट | अंग्रेजी विशेष, बारीक-खोलकर क्रिया समय और संज्ञा संख्या। |
| सार्वभौमिक निर्भरताएं | बहुभाषी टैगसेट | से अधिक PTB; भाषा-तटस्थ; बहुभाषी कार्य के लिए डिफ़ॉल्ट। |
| निर्भरता विश्लेषण | वाक्य वृक्ष | प्रत्येक शब्द का एक सिर होता है, प्रत्येक किनारे का व्याकरणिक संबंध होता है। |
| विटरबी | गतिशील प्रोग्रामिंग | उत्सर्जन और संक्रमणों को देखते हुए उच्चतम संभावना टैग अनुक्रम का पता लगाता है। |

## आगे पढ़ना

- [ज्यूराफस्की और मार्टिन  भाषण और भाषा प्रसंस्करण, अध्याय 8 और 18](https://web.stanford.edu/~jurafsky/slp3/) कैनोनिक पाठ्यपुस्तक उपचार POS और विश्लेषण।
- [सार्वभौमिक निर्भरता परियोजना](https://universaldependencies.org/) प्रत्येक बहुभाषी पार्सर द्वारा उपयोग किए जाने वाले बहुभाषी टैगसेट और ट्रीबैंक संग्रह।
- [spaCy भाषाई विशेषताएं गाइड](https://spacy.io/usage/linguistic-features) प्रत्येक विशेषता के लिए व्यावहारिक संदर्भ `Token`.
- [Chen and Manning (2014) । न्यूरल नेटवर्क का उपयोग करके एक तेज़ और सटीक निर्भरता पार्सर](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) पेपर जो न्यूरल पार्सर को मुख्यधारा में लाया।
