# विषय मॉडलिंग LDA और BERTopic

> LDAदस्तावेज विषयों का मिश्रण हैं, विषय शब्दों पर वितरण हैं। BERTopic: दस्तावेजों को एम्बेडिंग स्पेस में क्लस्टर किया जाता है, क्लस्टर विषय हैं। एक ही लक्ष्य, अलग-अलग विघटन।

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Time:** ~45 minutes

## समस्या

आपके पास 10,000 ग्राहक सहायता टिकट, 50,000 समाचार लेख, या 200,000 ट्वीट हैं। आपको यह जानने की जरूरत है कि संग्रह क्या है, इसे पढ़ने के बिना। आपके पास श्रेणियां नहीं हैं। आप यह भी नहीं जानते कि कितनी श्रेणियां मौजूद हैं।

विषय मॉडलिंग के बिना इसका जवाब देता है. इसे एक corpus दें, एक छोटे से सुसंगत विषयों का एक सेट वापस प्राप्त करें और, प्रत्येक दस्तावेज़ के लिए, उन विषयों पर एक वितरण।

दो एल्गोरिथम परिवारों का वर्चस्व है। LDA (2003) प्रत्येक दस्तावेज़ को लटेंट विषयों के मिश्रण के रूप में और प्रत्येक विषय को शब्दों पर वितरण के रूप में व्यवहार करता है। इन्फेरेंस बेयिसियन है। यह अभी भी उत्पादन में जहाज करता है जहां आपको मिश्रित सदस्यता विषय असाइनमेंट और स्पष्ट शब्द-स्तर की संभावना वितरण की आवश्यकता होती है।

BERTopic (2020) दस्तावेजों को कोडित करता है BERT, आयामों को कम करता है UMAP, समूहों के साथ HDBSCAN, और वर्ग आधारित के माध्यम से विषय शब्द निकाले TF-IDF. यह लघु पाठ, सोशल मीडिया और कुछ भी जीतता है जहां अर्थिक समानता शब्द ओवरलैप से अधिक मायने रखती है। एक दस्तावेज़ एक विषय प्राप्त करता है, जो लंबे रूप की सामग्री के लिए एक सीमा है।

यह सबक दोनों के लिए अंतर्ज्ञान और नामों का निर्माण करता है कि किसी दिए गए कॉर्पस के लिए कौन सा चुनना है।

## अवधारणा

![LDA मिश्रण मॉडल बनाम BERTopic समूहबद्ध करना](../assets/topic-modeling.svg)

**LDA जनक कहानी।** प्रत्येक विषय शब्दों पर एक वितरण है। प्रत्येक दस्तावेज़ विषयों का मिश्रण है। एक दस्तावेज़ में एक शब्द उत्पन्न करने के लिए, दस्तावेज़ के मिश्रण से एक विषय का नमूना लें, फिर उस विषय के वितरण से एक शब्द का नमूना लें। इन्फेरेंस इसे उलट देता हैः दिए गए अवलोकन किए गए शब्दों को देखते हुए, प्रत्येक दस्तावेज़ पर विषय वितरण और विषय पर शब्द वितरण का अनुमान लगाएं। गिर गया गिब्स नमूना या वैरिएशनल बेयज़ गणित करता है।

कुंजी LDA आउटपुटः

- `doc_topic`: मैट्रिक्स `(n_docs, n_topics)`, प्रत्येक पंक्ति का योग 1 (दस्तावेज के विषय मिश्रण) है।
- `topic_word`: मैट्रिक्स `(n_topics, vocab_size)`, प्रत्येक पंक्ति का योग 1 (विषय के शब्द वितरण) है।

**BERTopic पाइपलाइन।**

1. प्रत्येक दस्तावेज़ को वाक्य ट्रांसफार्मर (जैसे, `all-MiniLM-L6-v2`) 384 आयामी वेक्टर।
2. आयामों को कम करने के साथ UMAP ~ 5 आयामों तक। BERT एम्बेडमेंट क्लस्टर करने के लिए बहुत अधिक अमूर्त हैं।
3. के साथ समूह HDBSCAN. घनत्व आधारित, चर आकार के क्लस्टर और एक "बाह्य" लेबल का उत्पादन करता है।
4. प्रत्येक क्लस्टर के लिए, कम्प्यूटिंग क्लास आधारित TF-IDF शीर्ष शब्दों को निकालने के लिए समूह के दस्तावेजों पर।

आउटपुट प्रति दस्तावेज़ एक विषय है (और -1 आउटलियर लेबल) वैकल्पिक रूप से, एक सॉफ्ट सदस्यता HDBSCANसंभावना वेक्टर है।

```figure
topic-drift
```

## इसे बनाओ

### चरण 1: LDA स्किट-लर्न के माध्यम से

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

नोटः स्टॉपवर्ड हटाए गए, min_df और max_df दुर्लभ और सर्वव्यापी शब्द फ़िल्टर करते हैं, CountVectorizer (नहीं TfidfVectorizer) क्योंकि LDA कच्चे गिनती की उम्मीद करता है।

### चरण 2: BERTopic (production)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

फ़िल्टर चालू `Topic != -1` बूंदें BERTopic' के बाहर की बाल्टी (दस्तावेज़) HDBSCAN समूह नहीं हो सका) । `min_topic_size` नियंत्रण HDBSCAN' न्यूनतम क्लस्टर आकार; BERTopicइस उदाहरण में पाठ के पैमाने के लिए स्पष्ट रूप से 15 पर सेट किया गया है। 10,000 से अधिक दस्तावेजों के लिए, 50 या 100 तक बढ़ाएं।

### चरण 3: मूल्यांकन

दोनों ही तरीकों से विषय के शब्द उत्पन्न होते हैं। प्रश्न यह है कि क्या ये शब्द एक दूसरे के साथ मेल खाते हैं।

- **विषय सुसंगतता (c_v)** संयोजन NPMI (सामान्य बिंदु के अनुसार पारस्परिक जानकारी) स्लाइडिंग-विंडो संदर्भों पर शीर्ष शब्द जोड़े, विषय वेक्टरों में स्कोर को एकत्रित करता है, और कॉसिन समानता के माध्यम से उन वेक्टरों की तुलना करता है। उच्च बेहतर है। उपयोग `gensim.models.CoherenceModel` के साथ `coherence="c_v"`.
- **विषय विविधता।** सभी विषयों के शीर्ष शब्दों में अद्वितीय शब्दों का अंश। उच्च बेहतर है (विषय ओवरलैप नहीं करते हैं) ।
- **गुणात्मक निरीक्षण।** क्या वे किसी वास्तविक चीज़ का नाम देते हैं? मानव न्याय अभी भी रक्षा की आखिरी रेखा है।

## कौन सी चुनना है

| स्थिति | चुनें |
|-----------|------|
| लघु पाठ (ट्वीट, समीक्षा, शीर्षक) | BERTopic |
| विषय मिश्रण के साथ लंबे दस्तावेज | LDA |
| नहीं GPU / सीमित गणना | LDA या NMF |
| दस्तावेज़ स्तर के बहु-विषयक वितरण की आवश्यकता है | LDA |
| LLM विषय लेबलिंग के लिए एकीकरण | BERTopic (प्रत्यक्ष सहायता) |
| संसाधनों के लिए सीमित किनारे तैनाती | LDA |
| अधिकतम अर्थिक सुसंगतता | BERTopic |

सबसे बड़ा व्यावहारिक विचार दस्तावेज की लंबाई है। BERT एम्बेडेड ट्रंकट; LDA किसी भी लंबाई पर काम करता है। एम्बेडिंग मॉडल के संदर्भ से अधिक लंबे दस्तावेजों के लिए, या तो टुकड़ा + संकलित या उपयोग LDA.

## इसका प्रयोग करें

2026 स्टैकः

- **BERTopic.** संक्षिप्त पाठ और अर्थशास्त्र के लिए कोई भी मायने रखता है।
- **`gensim.models.LdaModel`.** क्लासिक LDA उत्पादन के लिए, परिपक्व, युद्ध-परीक्षण।
- **`sklearn.decomposition.LatentDirichletAllocation`.** आसान LDA प्रयोगों के लिए।
- **NMF.** गैर-नकारात्मक मैट्रिक्स कारककरण। LDA, लघु पाठ पर तुलनात्मक गुणवत्ता।
- **Top2Vec.** समान डिजाइन BERTopic. छोटा समुदाय लेकिन कुछ बेंचमार्क पर अच्छा।
- **FASTopic.** नए, से तेज़ BERTopic बहुत बड़े निकायों पर।
- **LLM-based लेबलिंग।** किसी भी क्लस्टरिंग चलाएं, फिर प्रत्येक क्लस्टर का नाम देने के लिए एक मॉडल को पूछें।

## इसे भेजें

के रूप में सहेजें `outputs/skill-topic-picker.md`:

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## व्यायाम

1. **- आराम से।** फिट LDA 20 न्यूजग्रुप डेटासेट पर 5 विषयों के साथ। प्रत्येक विषय पर शीर्ष 10 शब्द प्रिंट करें। प्रत्येक विषय को हाथ से लेबल करें। क्या एल्गोरिथम ने वास्तविक श्रेणियां पाई हैं?
2. **मध्यम।** फिट BERTopic एक ही 20 न्यूजग्रुप उपसमूह पर। LDA. कौन सी वास्तविक श्रेणियों को अधिक साफ ढंग से प्रदर्शित करती है?
3. **कठिन.** दोनों के लिए सी_वी सुसंगतता की गणना करें LDA और BERTopic अपने corpus पर. 5, 10, 20, 50 विषयों के साथ प्रत्येक चलाओ. प्लॉट सुसंगतता बनाम विषय संख्या. रिपोर्ट विषय संख्याओं के बीच कौन सा विधि अधिक स्थिर है.

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| विषय | एक बात corpus के बारे में है | शब्दों पर संभावना वितरण (LDA) या इसी तरह के दस्तावेजों का एक समूह (BERTopic). |
| मिश्रित सदस्यता | डॉ कई विषयों है | LDA प्रत्येक दस्तावेज को सभी विषयों पर एक वितरण सौंपता है। |
| UMAP | आयामों में कमी | स्थानीय संरचना को बनाए रखने वाले बहुमुखी सीखने का उपयोग BERTopic. |
| HDBSCAN | घनत्व समूह | चर आकार के क्लस्टर ढूंढता है; असाधारण के लिए "गूंज" लेबल (-1) उत्पन्न करता है। |
| सी_वी सुसंगतता | विषय गुणवत्ता मेट्रिक्स | स्लाइडिंग विंडो के भीतर शीर्ष विषय शब्दों की औसत बिंदु-दृष्टि पर आपसी जानकारी। |

## आगे पढ़ना

- [Blei, Ng, Jordan (2003) । लातेंट डायरिचलेट आवंटन](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)  LDA कागज।
- [ग्रूटेंडॉर्स्ट (2022). BERTopic: कक्षा आधारित विषय के साथ तंत्रिका मॉडल TF-IDF प्रक्रिया](https://arxiv.org/abs/2203.05794)  BERTopic कागज।
- [Röder, Both, Hinneburg (2015). विषय सामंजस्य उपायों की जगह का पता लगाना](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) कागज जो सी_वी और दोस्तों को पेश किया।
- [BERTopic दस्तावेज](https://maartengr.github.io/BERTopic/) उत्पादन संदर्भ। उत्कृष्ट उदाहरण।
