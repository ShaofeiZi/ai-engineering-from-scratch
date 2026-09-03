# मॉडल्स को एम्बेड करना  2026 गहरी गोताखोरी

> Word2Vec आधुनिक एम्बेडिंग मॉडल आपको एक वेक्टर प्रति passage, क्रॉस-भाषाई, दुर्लभ, घने और बहु-वेक्टर दृश्यों के साथ देते हैं, आपके सूचकांक के अनुरूप आकार। गलत चुनें और आपका RAG गलत चीज को वापस ले लेता है।

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (Information Retrieval)
**Time:** ~60 minutes

## समस्या

आपका RAG सिस्टम गलत मार्ग को 40% समय में प्राप्त करता है। दोषी शायद ही कभी वेक्टर डेटाबेस या प्रॉम्प्ट है। यह एम्बेडिंग मॉडल है।

2026 में एक एम्बेडिंग का चयन करने का मतलब पांच अक्षों पर चुनना हैः

1. **घने बनाम दुर्लभ बनाम बहु-वेक्टर।** एक वेक्टर प्रति passage, या एक प्रति टोकन, या शब्दों का एक दुर्लभ वजन बैग.
2. **भाषा कवरेज।** एक भाषाई अंग्रेजी मॉडल अभी भी केवल अंग्रेजी कार्यों पर जीतते हैं। बहुभाषी मॉडल तब जीतते हैं जब कॉर्पो मिश्रित होते हैं।
3. **संदर्भ लंबाई।** 512 टोकन बनाम 8,192 बनाम 32,768  और वास्तविक प्रभावी क्षमता अक्सर विज्ञापन अधिकतम का 60-70% होती है।
4. **आयाम बजट.** 3,072 पूर्णता में तैरते हैं precision = 12 KB 100M वेक्टर पर भंडारण $ 1,300 प्रति माह है. Matryoshka ट्रंकशन यह 4 गुना कटौती करता है.
5. **ओपन बनाम होस्ट किया गया।** ओपन-वेट का मतलब है कि आप स्टैक और डेटा को नियंत्रित करते हैं. होस्ट किया गया का मतलब है कि आप हमेशा नवीनतम के लिए नियंत्रण का आदान-प्रदान करते हैं।

इस पाठ में बाजी के नाम दिए गए हैं ताकि आप सबूतों पर ध्यान दे सकें, पिछले तिमाही में जो भी लोकप्रिय था, उस पर नहीं।

## अवधारणा

![घने, दुर्लभ और बहु-वेक्टर एम्बेडेड](../assets/embedding-modes.svg)

**घने एम्बेडमेंट।** एक वेक्टर प्रति मार्ग (आमतौर पर 384-3,072 आयाम) । कॉसिन समानता अर्थिक निकटता द्वारा मार्गों को रैंक करती है। OpenAI `text-embedding-3-large`, BGE-M3 घने मोड, यात्रा-3। डिफ़ॉल्ट विकल्प।

**स्पाइक एम्बेडमेंट्स.** SPLADE-style. एक ट्रांसफार्मर प्रत्येक शब्दकोश टोकन के लिए एक वजन का अनुमान लगाता है, फिर उनमें से अधिकांश को शून्य करता है। परिणाम आकार का एक दुर्लभ वेक्टर है। BM25) लेकिन सीखने वाले शब्द वजन के साथ।

**बहु-वेक्टर (अंतिम बातचीत)** ColBERTv2, Jina-ColBERT. प्रति टोकन एक वेक्टर. MaxSim: प्रत्येक क्वेरी टोकन के लिए, सबसे समान दस्तावेज़ टोकन खोजें, स्कोर को योग करें। भंडारण और स्कोर करने के लिए अधिक महंगा है, लेकिन लंबे क्वेरी और डोमेन-विशिष्ट कॉर्पो पर जीतता है।

**BGE-M3: एक साथ तीनों।** एकल मॉडल एक साथ घने, दुर्लभ और बहु-वेक्टर प्रतिनिधित्व करता है। प्रत्येक को स्वतंत्र रूप से क्वेरी किया जा सकता है; स्कोर वजन योग के माध्यम से फ्यूज। जब आप एक चेकपॉइंट से लचीलापन चाहते हैं तो 2026 डिफ़ॉल्ट।

**मैट्रियोशका प्रतिनिधित्व सीखने।** वेक्टर के पहले N आयामों को एक उपयोगी स्टैंडअलोन एम्बेडिंग बनाने के लिए प्रशिक्षित किया गया है। 1,536-dim वेक्टर को 256 dim तक काटें और 6x भंडारण बचत के लिए ~ 1% सटीकता का भुगतान करें। OpenAI पाठ-3 v4, यात्रा-4 , जिना v5, Gemini एम्बेडिंग 2, नोमिक v1.5+.

### इन MTEB रैंकिंग बोर्ड आंशिक कहानी बताता है

बड़े पैमाने पर पाठ एम्बेडिंग बेंचमार्क  लॉन्च पर 8 कार्य प्रकारों में 56 कार्य (2022), 100+ कार्यों में विस्तारित MTEB v2. 2026 की शुरुआत में, Gemini 2 शीर्ष निकासी को एम्बेड करना (67.71 MTEB-R) कोहरे एम्बेड-v4 आम लीड (65.2 MTEB). BGE-M3 रैंकिंग बोर्ड आवश्यक है लेकिन पर्याप्त नहीं है  हमेशा अपने डोमेन पर बेंचमार्क करें।

### तीन स्तरीय पैटर्न

| उपयोग के मामले | पैटर्न |
|----------|---------|
| त्वरित पहला पास | घने द्वि-संकेतक (BGE-M3, पाठ-3-छोटा) |
| याद करने की शक्ति | स्पायर (SPLADE, BGE-M3 sparse) + RRF फ्यूज |
| शीर्ष-50 पर सटीकता | बहु-वेक्टर (ColBERTv2) या क्रॉस-कोडर रीरैंकर |

अधिकांश उत्पादन ढेरों में तीनों का उपयोग किया जाता है।

```figure
gx-matryoshka
```

## इसे बनाओ

### चरण 1: बेसलाइन  स्राव के साथ घने एम्बेडेडBERT

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` हमेशा सेट करें।

### चरण 2: मैट्रियोशका काटना

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

ट्रंकिंग के बाद सामान्यीकरण। v1.5, OpenAI पाठ-3 और यात्रा-4 प्रशिक्षित हैं ताकि यह पहले कुछ स्तरों के लिए हानि मुक्त है।BERT) घटाने पर तीव्र रूप से गिरावट आती है।

### चरण 3: BGE-M3 बहुक्रियाशीलता

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

तीन सूचकांक, एक निष्कर्ष कॉल.

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

अपने डोमेन पर वजन समायोजित करें।

### चरण 4: MTEB कस्टम कार्य पर मूल्यांकन

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

अपने उम्मीदवार मॉडल पर चलाने के लिए एक *प्रतिनिधि* उपसमूह. केवल रैंक बोर्ड पर भरोसा न करें  आपका डोमेन मायने रखता है।

### चरण 5: हाथ से रोल किया गया कॉसीन खरोंच से

देखिये `code/main.py`. औसत हैशिंग ट्रिक एम्बेडमेंट (stdlib-केवल) । ट्रांसफार्मर एम्बेडमेंट के साथ प्रतिस्पर्धी नहीं है, लेकिन आकार दिखाता हैः टोकन → वेक्टर → सामान्यीकरण → डॉट उत्पाद।

## फंदे

- **पूछताछ और डॉक के लिए एक ही मॉडल।** कुछ मॉडल (भॉएज, जिना-ColBERT) असंबद्ध एन्कोडिंग का उपयोग करें  क्वेरी और दस्तावेज़ विभिन्न पथों से गुजरते हैं। हमेशा मॉडल कार्ड की जांच करें।
- **गायब उपसर्ग.** `bge-*` मॉडल की आवश्यकता `"Represent this sentence for searching relevant passages: "` 3-5 अंक याद करने के अंतर अगर आप भूल जाते हैं.
- **मैट्रियोशका को बहुत ज्यादा ट्रिमिंग करना.** 1,536 → 256 आमतौर पर सुरक्षित है. 1,536 → 64 नहीं है. अपने मूल्यांकन सेट पर मान्य करें.
- **संदर्भ काटना।** अधिकांश मॉडल अपनी अधिकतम लंबाई पर इनपुट को चुपचाप काटते हैं। लंबे डॉक्स को चकमा देने की आवश्यकता होती है (पाठ 23 देखें) ।
- **विलंबता पूंछ को अनदेखा करना.** MTEB स्कोर छिपा p99 600M मॉडल 335M मॉडल से 2 अंक से अधिक हो सकता है लेकिन प्रति क्वेरी 3 गुना अधिक लागत है।

## इसका प्रयोग करें

2026 स्टैकः

| स्थिति | चुनें |
|-----------|------|
| केवल अंग्रेजी, तेजी से, API | `text-embedding-3-large` या `voyage-3-large` |
| खुली तौल, अंग्रेजी | `BAAI/bge-large-en-v1.5` |
| खुले वजन, बहुभाषी | `BAAI/bge-m3` या `Qwen3-Embedding-8B` |
| Long context (32k+) | यात्रा-3-बड़ा, कोहरे एम्बेड-v4, Qwen3-Embedding-8B |
| CPU-only तैनाती | नामिक सम्मिलित v2 (137M पैराम, MoE) |
| भंडारण के लिए प्रतिबंधित | Matryoshka-truncated + int8 क्वांटिज़ेशन |
| कीवर्ड भारी प्रश्न | जोड़ें SPLADE दुर्लभ, RRF-fuse घने के साथ |

2026 पैटर्नः शुरू करें BGE-M3 या पाठ-3-बड़ा, अपने डोमेन पर मूल्यांकन के साथ MTEB, यदि डोमेन-विशिष्ट मॉडल 3 अंक से अधिक जीतता है।

## इसे भेजें

के रूप में सहेजें `outputs/skill-embedding-picker.md`:

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## व्यायाम

1. **- आराम से।** 100 वाक्य को एन्कोड करें `bge-small-en-v1.5` पूर्ण अंधेरा (384), फिर मात्र्योश्का 128 पर। MRR 10 प्रश्नों पर छोड़ दें।
2. **मध्यम।** तुलना करें BGE-M3 अपने डोमेन से 500 पदों पर घने, दुर्लभ, और कोलबर्ट. जो recall@10 पर जीतता है? RRF फ्यूजन सबसे अच्छा एकल मोड से बाहर है?
3. **कठिन.** दौड़ें MTEB अपने शीर्ष 2 डोमेन कार्यों में तीन उम्मीदवार मॉडल पर रिपोर्ट MTEB स्कोर, p99 100 प्रश्नों के बैच पर विलंबता, और $ 1 मिलियन प्रश्नों. Pareto-उत्तम एक चुनें.

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| घने सम्मिलित | वेक्टर | प्रति पाठ एक निश्चित आकार वेक्टर. रैंकिंग के लिए कॉसिन समानता. |
| स्पर एम्बेडिंग | सीखा BM25 | एक वजन प्रति शब्द चिन्ह; ज्यादातर शून्य; अंत-से-अंत प्रशिक्षित। |
| बहु-वेक्टर | ColBERT-style | प्रति टोकन एक वेक्टर; MaxSim स्कोरिंग; बड़ा सूचकांक, बेहतर याद। |
| मट्रोशका | रूसी गुड़िया ट्रिक | पहले N dims अपने आप में एक मान्य छोटे एम्बेडिंग हैं। |
| MTEB | बेंचमार्क | बड़े पैमाने पर पाठ एम्बेडिंग बेंचमार्क  लॉन्च पर 56 कार्य, 100+ में v2. |
| BEIR | पुनर्प्राप्ति बेंचमार्क | 18 शून्य-शॉट रिकवरी कार्य; अक्सर क्रॉस-डोमेन मजबूती के लिए उद्धृत किया जाता है। |
| असममित एन्कोडिंग | क्वेरी ≠ doc पथ | मॉडल पूछताछ और दस्तावेजों के लिए विभिन्न अनुमानों का उपयोग करता है। |

## आगे पढ़ना

- [रीमर, गुरेविच (2019) । वाक्य-BERT](https://arxiv.org/abs/1908.10084) द्वि-संकेतक कागज।
- [मुन्निगोफ एट अल (2022). MTEB: बड़े पैमाने पर पाठ एम्बेड बेंचमार्क](https://arxiv.org/abs/2210.07316) रैंकिंग बोर्ड पेपर।
- [चेन और अन्य (2024). BGE-M3: बहुभाषी, बहुक्रियाशीलता, बहु-ग्रानुलिटी](https://arxiv.org/abs/2402.03216) एकीकृत तीन मोड मॉडल।
- [कुसुपति और अन्य (2022) । मातृशोखा प्रतिनिधित्व सीखना](https://arxiv.org/abs/2205.13147) आयाम-शिखर प्रशिक्षण उद्देश्य।
- [संतानाम एवं अन्य (2022). ColBERTv2: हल्के देर से बातचीत के माध्यम से प्रभावी और कुशल पुनर्प्राप्ति](https://arxiv.org/abs/2112.01488) उत्पादन में देर से बातचीत।
- [MTEB गले लगाना चेहरा पर रैंकिंग बोर्ड](https://huggingface.co/spaces/mteb/leaderboard) लाइव रैंकिंग।
