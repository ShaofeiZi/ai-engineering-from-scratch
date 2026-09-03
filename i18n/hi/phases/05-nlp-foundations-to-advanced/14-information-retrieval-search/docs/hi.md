# जानकारी प्राप्त करना और खोजना

> BM25 यह एक बहुत ही सटीक है, लेकिन नाजुक है. घने एक विस्तृत जाल फेंकता है, लेकिन कुंजीशब्दों को याद करता है. हाइब्रिड 2026 डिफ़ॉल्ट है. बाकी सब कुछ ट्यूनिंग है.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 minutes

## समस्या

उपयोगकर्ता टाइप करता है "क्या होता है अगर कोई पैसा पाने के लिए झूठ बोलता है" और यह उम्मीद करता है कि वह कानून ढूंढता है जो वास्तव में कवर करता हैः "धारा 420 IPC." एक कीवर्ड खोज इसे पूरी तरह से याद करती है (कोई साझा शब्दावली नहीं) । एक अर्थपूर्ण खोज इसे याद करती है यदि एम्बेडमेंट कानूनी पाठ पर प्रशिक्षित नहीं किए गए थे। वास्तविक खोज दोनों को संभालनी है।

IR प्रत्येक के नीचे पाइपलाइन है RAG 2026 वास्तुकला जो उत्पादन में काम करता है एक एकल विधि नहीं है. यह पूरक तरीकों की एक श्रृंखला है, प्रत्येक पिछले एक की विफलताओं को पकड़ने.

यह सबक हर टुकड़े और नाम का निर्माण करता है जो हर पकड़ में विफल रहता है।

## अवधारणा

![हाइब्रिड रिट्रीवमेंटः BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

चार परतें, जो आप की जरूरत है चुनें.

1. **स्पायर रिट्रीव (BM25).** तेजी से, सटीक मैचों पर सटीक, अर्थशास्त्र पर भयानक एक उल्टा सूचकांक पर चलाएं. लाखों दस्तावेजों पर प्रति क्वेरी 10ms से नीचे. आपको विधान संदर्भ, उत्पाद कोड, त्रुटि संदेश, नामित संस्थाओं सही मिलता है।
2. **घने निकासी.** वेक्टर में क्वेरी और दस्तावेज एन्कोड करें. निकटतम पड़ोसी खोज. पैराफ्रेसेस और अर्थिक समानता को कैप्चर करता है. एक वर्ण से भिन्न सटीक कीवर्ड मैचों को याद करता है। 50-200ms प्रति क्वेरी के साथ FAISS या वेक्टर DB.
3. **संलयन।** रैंक सूची को दुर्लभ और घने से मिलाएं।RRF) आसान डिफ़ॉल्ट है क्योंकि यह कच्चे स्कोर (जो विभिन्न पैमाने में रहते हैं) को अनदेखा करता है और केवल रैंक पदों का उपयोग करता है। वजनबद्ध संलयन एक विकल्प है जब आप जानते हैं कि आपके डोमेन के लिए एक संकेत हावी है।
4. **क्रॉस-कोडर रैंक पुनर्गठन।** फ्यूजन से शीर्ष-30 ले लो। क्रॉस-एन्कोडर चलाएं (एक साथ क्वेरी + दस्तावेज़, प्रत्येक जोड़ी को स्कोर करना) । शीर्ष-5 रखें। क्रॉस-एन्कोडर प्रति जोड़ी द्वि-एन्कोडर की तुलना में धीमे हैं लेकिन बहुत अधिक सटीक हैं। आप केवल शीर्ष-30 पर उन्हें चलाकर amortize करते हैं।

तीन-तरफ़ा निकासी (BM25 + dense + learned-sparse like SPLADE) 2026 में दो-तरफा बेंचमार्क से बेहतर प्रदर्शन करता है लेकिन सीखने के लिए स्परस सूचकांक के लिए बुनियादी ढांचे की आवश्यकता होती है। अधिकांश टीमों के लिए, दो-तरफा प्लस क्रॉस-कोडर री-रैंक सबसे अच्छा स्थान है।

```figure
gx-hybrid-retrieval
```

## इसे बनाओ

### चरण 1: BM25 खरोंच से

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

दो मापदंडों को जानना लायक है। `k1=1.5` टर्म-फ्रीक्वेंसी संतृप्ति को नियंत्रित करता है; उच्च का अर्थ है टर्म पुनरावृत्ति पर अधिक वजन। `b=0.75` 0 दस्तावेज़ की लंबाई को अनदेखा करता है, 1 पूरी तरह से सामान्य करता है। डिफ़ॉल्ट रॉबर्टसन के मूल पेपर से सिफारिशें हैं और शायद ही कभी ट्यूनिंग की आवश्यकता होती है।

### चरण 2: एक द्वि-संकेतक के साथ घने पुनर्प्राप्ति

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2-normalize एम्बेडिंग तो बिंदु उत्पाद cosine के बराबर है. `all-MiniLM-L6-v2` 384dim, तेज, और अधिकांश अंग्रेजी निकालने के लिए पर्याप्त मजबूत है। बहुभाषी काम के लिए, उपयोग `paraphrase-multilingual-MiniLM-L12-v2`. उच्चतम सटीकता के लिए, `bge-large-en-v1.5` या `e5-large-v2`.

### चरण 3: पारस्परिक रैंक विलय

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

इन `k=60` निरंतर मूल से आता है RRF कागज। उच्च `k` रैंक अंतर के योगदान को समतल करता है; कम `k` 60 प्रकाशित डिफ़ॉल्ट है और शायद ही कभी ट्यूनिंग की जरूरत है।

### चरण 4: हाइब्रिड खोज + पुनः रैंक

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

तीन चरणों में रचना। BM25 शब्दकोश मिलान. घने अर्थिक मिलान मिलान. RRF क्रॉस-एन्कोडर क्वेरी-डॉकमेंट जोड़े के साथ शीर्ष-30 को फिर से स्कोर करता है, जो कि दो-एन्कोडर को याद किए गए बारीक-कट्टी प्रासंगिकता को कैप्चर करता है। शीर्ष-5 रखें।

### चरण 5: मूल्यांकन

| मेट्रिक | अर्थ |
|--------|---------|
| याद करना | सही दस्तावेज के अस्तित्व के प्रश्नों में, यह शीर्ष-के में कितनी बार होता है? |
| MRR (मध्यम पारस्परिक पद) | प्रथम प्रासंगिक दस्तावेज की 1/श्रेणी का औसत। |
| nDCG@k | केवल द्विआधारी प्रासंगिक/नहीं के लिए प्रासंगिकता ग्रेड के लिए लेखांकन। |

के लिए RAG विशेष रूप से, **याद करना** आपके पाठक जवाब नहीं दे सकते हैं अगर सही passage नहीं है प्राप्त सेट में.

डिबगिंग टिपः असफल क्वेरी के लिए, दुर्लभ और घने रैंकिंग में अंतर करें। यदि एक सही दस्तावेज़ पाता है और दूसरा नहीं करता है, तो आपके पास एक शब्दावली असंगतता (फिक्सः गायब आधा जोड़ें) या अर्थहीन अस्पष्टता (फिक्सः बेहतर एम्बेडिंग या एक रीरेंकर) है।

## इसका प्रयोग करें

2026 स्टैकः

| पैमाने | स्टैक |
|-------|-------|
| 1k-100k डॉक्स | स्मृति में BM25 + `all-MiniLM-L6-v2` embeddings + RRF. कोई अलग नहीं DB. |
| 100k-10M डॉक्स | FAISS या घने + लोचदार खोज के लिए pgvector / OpenSearch के लिए BM25. समानांतर में चलें। |
| 10M+ docs | हाइब्रिड समर्थन के साथ Qdrant / Weaviate / Vespa / Milvus। क्रॉस-एन्कोडर शीर्ष 30 पर पुनः रैंक। |
| सर्वोत्तम गुणवत्ता वाली सीमा | तीन-तरफ़ा (BM25 + dense + SPLADE) + ColBERT देर से बातचीत के लिए पुनः रैंक |

आप जो भी चुनते हैं, मूल्यांकन के लिए बजट। अंत-से-अंत बेंचमार्क से पहले बेंचमार्क निकालना RAG एक पाठक ठीक नहीं कर सकता है कि क्या रिट्रीवर चूक गया।

### 2026 उत्पादन से कठिन सीखे गए पाठ RAG

- **80% RAG विफलताएं सेवन और चकमक से होती हैं, मॉडल नहीं।** टीमों को सप्ताहों के लिए आदान-प्रदान खर्च LLMs और ट्यूनिंग संकेतों जबकि पुनर्प्राप्ति चुपचाप गलत संदर्भ हर तीसरे क्वेरी वापस करता है. पहले चकमिंग ठीक.
- **टुकड़े टुकड़े करने की रणनीति टुकड़े के आकार से अधिक मायने रखती है।** फिक्स्ड साइज स्प्लिट टेबल, कोड और घोंसले हुए हेडर तोड़ते हैं। वाक्य-जागरूक डिफ़ॉल्ट है; अर्थ या LLM-based तकनीकी दस्तावेजों और उत्पाद पुस्तिकाओं के लिए चश्मा का भुगतान किया जाता है।
- **माता-पिता-डॉक्टर पैटर्न।** सटीकता के लिए छोटे "बच्चे" टुकड़े निकालें। जब एक ही माता-पिता अनुभाग से कई बच्चे दिखाई देते हैं, तो संदर्भ को बनाए रखने के लिए माता-पिता ब्लॉक में स्विच करें। यह लगातार बिना पुनर्व्यवस्थापन के उत्तर गुणवत्ता को बढ़ाता है।
- **k_rerank=3 आमतौर पर इष्टतम होता है।** प्रत्येक अतिरिक्त टुकड़ा अतीत जो जवाब की गुणवत्ता को उठाने के बिना टोकन लागत और उत्पादन विलंबता जोड़ता है। k=8 is still better than k=3 आपके लिए, रेंकर कम प्रदर्शन कर रहा है.
- **HyDE / क्वेरी विस्तार.** प्रश्न से एक परिकल्पनात्मक उत्तर उत्पन्न करें, इसे एम्बेड करें, प्राप्त करें. छोटे प्रश्नों और लंबे दस्तावेजों के बीच वाक्यांश अंतर को पुल करता है. बिना प्रशिक्षण के मुफ्त सटीकता लिफ्ट।
- **8K टोकन के तहत संदर्भ बजट।** उस सीमा पर लगातार हिट का मतलब है कि पुनर्व्यवस्थापक की सीमा बहुत ढीली है।
- **सब कुछ संस्करण.** संकेत, टुकड़े-टुकड़े करने के नियम, एम्बेडिंग मॉडल, रीरेंकर। किसी भी बहाव चुपचाप उत्तर की गुणवत्ता को तोड़ता है। CI वफादारी, संदर्भ सटीकता, और उत्तरहीन प्रश्न दर पर गेट उपयोगकर्ताओं को देखने से पहले ब्लॉक regressions।
- **तीन-तरफ़ा निकासी (BM25 + dense + learned-sparse like SPLADE) दोतरफा प्रदर्शन से अधिक** 2026 में बेंचमार्क पर, विशेष रूप से सही संज्ञाओं को अर्थशास्त्र के साथ मिलाकर क्वेरी के लिए। SPLADE सूचकांक।

2026 में उद्योग के माप के अनुसार सही पुनर्प्राप्ति डिजाइन भ्रामकता को 70-90% कम करता है। RAG प्रदर्शन लाभ बेहतर निकासी से आते हैं, मॉडल को ठीक से समायोजित नहीं करते हैं।

## इसे भेजें

के रूप में सहेजें `outputs/skill-retrieval-picker.md`:

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## व्यायाम

1. **- आराम से।** कार्यान्वयन `hybrid_search` 500 दस्तावेजों के एक कॉर्पस पर ऊपर। परीक्षण 20 प्रश्नों. तुलना याद करने के 5 के बीच BM25-only, केवल घने, और हाइब्रिड.
2. **मध्यम।** जोड़ें MRR गणना. एक ज्ञात सही दस्तावेज़ के साथ प्रत्येक परीक्षण क्वेरी के लिए, सही दस्तावेज़ की रैंक खोजें BM25, घने और हाइब्रिड रैंकिंग। MRR प्रत्येक के लिए।
3. **कठिन.** अपने डोमेन पर एक घने एन्कोडर ठीक से समायोजित करें MultipleNegativesRankingLoss 500 क्वेरी-दस्तावेज़ जोड़े से एक प्रशिक्षण सेट बनाएं. पूर्व और पोस्ट-फाइन-ट्यून यादों की तुलना करें.

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| BM25 | कीवर्ड खोज | ओकापी BM25. शब्द आवृत्ति के अनुसार दस्तावेजों को स्कोर करता है, IDF, और लंबाई. |
| घने निकासी | वेक्टर खोज | वेक्टर में क्वेरी + डॉक एन्कोड, निकटतम पड़ोसियों को खोजने. |
| द्वि-संकेतक | एम्बेडिंग मॉडल | स्वतंत्र रूप से क्वेरी और डॉक कोड. क्वेरी समय पर तेजी से. |
| क्रॉस-एन्कोडर | रेनकर मॉडल | क्वेरी + डॉक एक साथ एन्कोड. धीमी लेकिन सटीक. |
| RRF | रैंक संलयन | योग करके दो रैंकिंग को मिलाएं `1/(k + rank)`. |
| याद करना | प्राप्ति मेट्रिक्स | प्रश्नों का अंश जहां एक प्रासंगिक दस्तावेज़ शीर्ष-के में है। |

## आगे पढ़ना

- [रॉबर्टसन और ज़ारागोसा (2009) । संभावनावादी प्रासंगिकता ढांचाः BM25 और आगे](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) अंतिम BM25 उपचार।
- [कार्पुकिन एट अल. (2020). ओपन-डोमेन के लिए घने पासज रिट्रीवल QA](https://arxiv.org/abs/2004.04906) — DPR, कैनोनिक द्वि-संकेतक.
- [औपचारिक एवं अन्य (2021). SPLADE: स्पायर लेक्सिकल और एक्सपेंशन मॉडल](https://arxiv.org/abs/2107.05720) सीखे-अवकाश रिट्रीवर जो घने के साथ अंतर को बंद करता है।
- [कॉर्मेक, क्लार्क, बुट्टचर (2009) । पारस्परिक रैंक फ्यूजन कॉन्डोर्सेट और व्यक्तिगत रैंक सीखने की विधियों से बेहतर है।](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF कागज।
- [खट्टब और ज़हरिया (2020). ColBERT: कुशल और प्रभावी मार्ग खोज](https://arxiv.org/abs/2004.12832) देर से बातचीत के बाद प्राप्ति।
