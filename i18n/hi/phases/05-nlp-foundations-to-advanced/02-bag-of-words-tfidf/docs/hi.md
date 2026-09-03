# शब्द के बैग, TF-IDF, और पाठ प्रतिनिधित्व

> पहले गिनें, बाद में सोचें। TF-IDF अभी भी 2026 में अच्छी तरह से परिभाषित कार्यों पर एम्बेडिंग से बेहतर है।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 02 (Linear Regression from Scratch)
**Time:** ~75 minutes

## समस्या

मॉडल को संख्याओं की जरूरत है.

हर NLP पाइपलाइन एक ही सवाल का जवाब देना है. हम कैसे टोकन के एक चर लंबाई धारा एक निश्चित आकार वेक्टर में परिवर्तित कर सकते हैं कि एक वर्गीकरण का उपभोग कर सकते हैं. पहला जवाब क्षेत्र पर लैंडिंग सबसे बेवकूफ था जो काम करता है. शब्दों की गिनती. एक वेक्टर बनाएं.

उस वेक्टर ने अधिक उत्पादन किया है NLP स्पैम फ़िल्टर, विषय वर्गीकरण, लॉग विसंगतियों का पता लगाना, खोज रैंकिंग (पूर्व में) BM25), भावनात्मक विश्लेषण की पहली लहर, शैक्षणिक NLP 2026 अभ्यासकों अभी भी संकीर्ण वर्गीकरण कार्यों पर पहले इसके लिए पहुंचते हैं। यह त्वरित, व्याख्या योग्य और अक्सर उन कार्यों पर 400M पैरामीटर एम्बेडिंग मॉडल से अलग नहीं होता है जहां शब्द उपस्थिति ही मायने रखती है।

यह सबक शब्दों का बैग बनाता है, तो TF-IDFफिर तीन पंक्तियों में एक ही काम कर रहा है scikit-लर्न दिखाता है. फिर विफलता मोड नाम है कि आप एम्बेड के लिए पहुँचता है.

## अवधारणा

**शब्द का बैग (BoW)** प्रत्येक दस्तावेज़ के लिए, गणना करें कि प्रत्येक शब्दावली शब्द कितनी बार दिखाई देता है। वेक्टर लंबाई शब्दावली का आकार है। स्थिति `i` शब्द की गिनती है `i`.

**TF-IDF** पुनः वजन BoW. हर दस्तावेज़ में एक शब्द जो दिखाई देता है वह जानकारीपूर्ण नहीं है, इसलिए इसे कम करें। एक शब्द जो पूरे कॉर्पस में दुर्लभ है लेकिन एक ही दस्तावेज़ में अक्सर होता है वह संकेत है, इसलिए इसे बढ़ाएं।

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

कहाँ `TF` दस्तावेज़ में शब्द आवृत्ति है, `df` दस्तावेज़ आवृत्ति (शब्द में कितने डॉक्स हैं), `N` यह कुल दस्तावेज है। `log` हर जगह मौजूद शब्दों के लिए वजन को सीमित रखता है।

मुख्य गुणः दोनों व्याख्या योग्य अक्षों के साथ दुर्लभ वेक्टर उत्पन्न करते हैं। आप एक प्रशिक्षित वर्गीकरण के वजन को देख सकते हैं और पढ़ सकते हैं कि कौन से शब्द प्रत्येक वर्ग की ओर दस्तावेज़ को धक्का देते हैं। आप 768 आयामी के साथ ऐसा नहीं कर सकते हैं BERT सम्मिलित करना।

```figure
bow-tfidf
```

## इसे बनाओ

### चरण 1: शब्दावली का निर्माण करें

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

इनपुटः टोकन दस्तावेजों की सूची (किसी भी शब्द स्तर टोकन करने वाला करेगा; `code/main.py` इस पाठ में सरल लघु अक्षर संस्करण का उपयोग किया जाता है। `{word: index}` निर्दिष्ट करें. स्थिर सम्मिलन क्रम का अर्थ है शब्द सूचकांक 0 पहला शब्द है जो पहले दस्तावेज़ में देखा जाता है। सम्मेलन भिन्न होता है; scikit-learn क्रमशः वर्णमाला में क्रमबद्ध होता है।

### चरण 2: शब्दों का बैग

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

पंक्तियाँ दस्तावेज हैं, स्तंभ शब्दावली सूचकांक हैं। `[i][j]` है "कई बार शब्द `j` दस्तावेज़ में दिखाई देता है `i`. " डॉ 1 ने `cat` दो बार क्योंकि यह किया है. डॉ 0 है `ran` शून्य बार क्योंकि यह नहीं किया।

### चरण 3: शब्द आवृत्ति और दस्तावेज आवृत्ति

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

दो चिकनाई चाल नाम देने लायक है। `(n+1)/(d+1)` से बचता है `log(x/0)`. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . `+1` सुनिश्चित करता है कि हर दस्तावेज में एक शब्द अभी भी है IDF 1 (0 नहीं), scikit-learn के डिफ़ॉल्ट से मेल खाता है। अन्य कार्यान्वयन कच्चे का उपयोग करते हैं `log(N/df)`दोनों काम करते हैं; चिकनी संस्करण अधिक अनुकूल है।

### चरण 4: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

तीन दस्तावेज, पांच शब्दावली शब्द (`the`, `cat`, `sat`, `dog`, `ran`). `the` सभी तीन में दिखाई देता है, तो इसके IDF कम है। `dog` एक में दिखाई देता है, तो इसके IDF वेक्टर दुर्लभ हैं (ज्यादातर प्रविष्टियां छोटी हैं) और भेदभावपूर्ण शब्द पॉप अप होते हैं।

### चरण 5: L2-normalize पंक्तियाँ

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

सामान्यीकरण के बिना, एक लंबा दस्तावेज़ एक बड़ा वेक्टर प्राप्त करता है और समानता स्कोर पर हावी होता है। L2 सामान्यीकरण इकाई हाइपरस्फीयर पर हर दस्तावेज़ डालता है। पंक्तियों के बीच कॉसिन समानता अब सिर्फ एक बिंदु उत्पाद है।

## इसका प्रयोग करें

स्किट-लर्न उत्पादन संस्करण जहाजों।

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` टोकनकरण, शब्दावली और BoW एक कॉल में। `TfidfVectorizer` जोड़ता है IDF वजन और L2 सामान्यीकरण. दोनों ही दुर्लभ मैट्रिक्स लौटाते हैं. 100k दस्तावेजों के लिए, घने संस्करण स्मृति में फिट नहीं होता है; वर्गीकरण घने की मांग तक घने रहें।

बटन जो सब कुछ बदलते हैंः

| आरजी | प्रभाव |
|-----|--------|
| `ngram_range=(1, 2)` | बड़े ग्रैम शामिल करें. आमतौर पर वर्गीकरण को बढ़ाता है. |
| `min_df=2` | 2 डॉक्स से कम में शब्दों को छोड़ दें. शोर डेटा पर शब्दावली को ट्रिम करें. |
| `max_df=0.95` | डॉक में 95% से अधिक में शब्द छोड़ दें. हार्ड कोडित सूची के बिना स्टॉपवर्ड हटाने के करीब है. |
| `stop_words="english"` | कार्य-निर्भर  भावनात्मक विश्लेषण *नहीं* नकार छोड़ दें। |
| `sublinear_tf=True` | उपयोग `1 + log(tf)` कच्चे के बजाय `tf`. एक शब्द एक ही दस्तावेज़ में कई बार दोहराए जाने में मदद करता है. |

### जब TF-IDF अभी भी जीतता है (2026 से)

- स्पैम का पता लगाना, विषय लेबलिंग, लॉग विसंगति चिह्नित करना शब्द उपस्थिति ही मायने रखती है, अर्थिक बारीकियां नहीं।
- कम डेटा व्यवस्थाएं (सैकड़ों लेबल वाले उदाहरण) । TF-IDF इसके अतिरिक्त लॉजिस्टिक रेग्रेसशन में कोई पूर्व-प्रशिक्षण लागत नहीं होती है।
- जहां भी देरी मायने रखती है। TF-IDF एक ट्रांसफार्मर के माध्यम से एक दस्तावेज़ एम्बेड करने में 10-100ms लगता है।
- सिस्टम जो अपने भविष्यवाणियों की व्याख्या करना चाहिए वर्गीकरण के गुणांक की जांच करें शीर्ष सकारात्मक शब्द कारण हैं।

### जब TF-IDF विफलता

अर्थशास्त्र दृष्टिहीनता की विफलता. इन दो दस्तावेजों पर विचार करेंः

- "फिल्म बिल्कुल अच्छा नहीं था। "
- "फिल्म उत्कृष्ट था।

एक नकारात्मक समीक्षा है. एक सकारात्मक है. TF-IDF ओवरलैप ठीक है `{the, movie, was}`एक शब्द के बैग वर्गीकरण करने वाले को याद रखना होगा कि शब्द `not` निकट `good` यह पर्याप्त डेटा पर यह सीख सकता है, लेकिन कभी भी एक मॉडल के रूप में gracefully कि वाक्यविन्यास समझता है.

अन्य विफलता: निष्कर्ष पर शब्द के बाहर शब्द। BoW मॉडल पर प्रशिक्षित IMDb समीक्षाओं के साथ क्या करना है पता नहीं है `Zoomer-approved` उपशब्द एम्बेडिंग (पाठ 04) इस पर काम करते हैं। TF-IDF नहीं कर सकते।

### हाइब्रिड TF-IDF भारित एम्बेड

मध्य-डेटा वर्गीकरण के लिए 2026 व्यावहारिक डिफ़ॉल्टः उपयोग TF-IDF शब्दों के एम्बेडेड पर ध्यान के रूप में वजन।

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

आप एम्बेडमेंट से अर्थ क्षमता प्राप्त करते हैं, और दुर्लभ शब्द जोर TF-IDF. वर्गीकरणकर्ता pooled vector पर ट्रेन करता है। यह अपने आप में या तो भावना, विषय और इरादे वर्गीकरण के लिए लगभग 50k लेबल उदाहरणों से नीचे प्रदर्शन करता है।

## इसे भेजें

के रूप में सहेजें `outputs/prompt-vectorization-picker.md`:

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## व्यायाम

1. **- आराम से।** कार्यान्वयन `cosine_similarity(doc_vec_a, doc_vec_b)` पर L2-normalized TF-IDF जाँचें कि समान दस्तावेजों को 1.0 और असंगत शब्दावली दस्तावेजों को 0.0 का स्कोर मिलता है।
2. **मध्यम।** जोड़ें `n-gram` समर्थन `bag_of_words`पैरामीटर `n` उपज गणना से अधिक `n`- ग्राम. यह परीक्षण करें `n=2` पर `["the", "cat", "sat"]` के लिए बिग्राम गणना का उत्पादन करता है `["the cat", "cat sat"]`.
3. **कठिन.** निर्माण TF-IDF-weighted-embedding उपरोक्त हाइब्रिड का उपयोग करके GloVe 100d वेक्टर (एक बार डाउनलोड, कैश) । वर्गीकरण सटीकता की तुलना सादे के साथ करें TF-IDF और 20 न्यूजग्रुप डेटासेट में सादे औसत-बंद एम्बेडेड। रिपोर्ट कौन कहाँ जीतता है।

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| BoW | वर्ड आवृत्ति वेक्टर | एक दस्तावेज में शब्दावली शब्दों की गिनती। आदेश को फेंक देता है। |
| TF | अवधि आवृत्ति | दस्तावेज़ में एक शब्द की गिनती, वैकल्पिक रूप से दस्तावेज़ लंबाई द्वारा सामान्यीकृत। |
| DF | दस्तावेज़ आवृत्ति | कम से कम एक बार शब्द युक्त दस्तावेजों की गिनती। |
| IDF | दस्तावेज की उल्टा आवृत्ति | `log(N / df)` हर जगह दिखाई देने वाले शब्दों को कम वजन में रखता है। |
| स्पर वेक्टर | अधिकतर शून्य | शब्दावली आमतौर पर 10k-100k शब्द होती है; अधिकांश किसी भी दिए गए दस्तावेज़ में अनुपस्थित होते हैं। |
| कॉसिन समानता | वेक्टर कोण | डॉट उत्पाद L2-normalized वेक्टर. 1 समान है, 0 orthogonal है. |

## आगे पढ़ना

- [scikit-learn  पाठ से सुविधा निकासी](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) कैनोनिक API संदर्भ, प्लस प्रत्येक बटन पर नोट्स.
- [Salton, G., & Buckley, C. (1988). स्वचालित पाठ निकासी में शब्द-वजन दृष्टिकोण](https://www.sciencedirect.com/science/article/pii/0306457388900210) कागज जो बनाया TF-IDF एक दशक के लिए डिफ़ॉल्ट।
- ["क्यों TF-IDF अभी भी एम्बेडिंग्स से बेहतर"  अश्वक Thonikkadavan (मध्य)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) 2026 में यह निर्णय लें कि पुरानी विधि कब और क्यों जीतती है।
