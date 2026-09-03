# इकाई लिंकिंग और असंगति

> NER "पेरिस" पाया. "संबद्ध इकाई निर्णय करता हैः पेरिस, फ्रांस? पेरिस हिल्टन? पेरिस, टेक्सास? पेरिस (ट्रोजन राजकुमार)? बिना लिंक, अपने ज्ञान ग्राफ अस्पष्ट रहता है।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time:** ~60 minutes

## समस्या

एक वाक्य में लिखा हैः "जॉर्डन ने प्रेस को हराया।" NER "जॉर्डन" टैग PERSON. अच्छा है, लेकिन *जो* जॉर्डन?

- माइकल जॉर्डन (बास्केटबॉल)?
- माइकल बी. जॉर्डन (अभिनेता)?
- माइकल I. जॉर्डन (बर्केली) ML प्रोफेसर  हाँ, यह भ्रम वास्तविक है ML कागज)?
- जॉर्डन (देश)?
- जॉर्डन (इब्रानी प्रथम नाम)?

इकाई को जोड़ना (EL) ज्ञान आधार में प्रत्येक उल्लेख को एक अद्वितीय प्रविष्टि में हल करता हैः विकिडाटा, विकिपीडिया, DBpedia, या अपने डोमेन KB. दो उपकार्य:

1. **उम्मीदवार पीढ़ी।** "जॉर्डन" को देखते हुए, जो KB प्रविष्टियाँ विश्वसनीय हैं?
2. **स्पष्टीकरण।** संदर्भ को देखते हुए, कौन सा उम्मीदवार सही है?

दोनों चरण सीखने योग्य हैं। दोनों बेंचमार्क किए गए हैं। संयुक्त पाइपलाइन एक दशक से स्थिर है  जो परिवर्तन है वह है डिसाम्बिक्यूटर की गुणवत्ता।

## अवधारणा

![पाइपलाइन को जोड़ने वाली इकाईः उल्लेख → उम्मीदवार → स्पष्ट इकाई](../assets/entity-linking.svg)

**उम्मीदवार पीढ़ी।** उल्लेख सतह रूप ("जॉर्डन") को देखते हुए, एक उपनाम सूचकांक में उम्मीदवारों की तलाश करें। विकिपीडिया उपनाम शब्दकोशों में अधिकांश नामित संस्थाएं शामिल हैंः "JFK" → जॉन एफ. केनेडी, जैक्लीन केनेडी, JFK हवाई अड्डा, JFK (फिल्म) एक विशिष्ट सूचकांक प्रति उल्लेख 10-30 उम्मीदवारों को वापस देता है।

**स्पष्टीकरणः तीन दृष्टिकोण।**

1. **पूर्व + संदर्भ (मिलन और विट्टन, 2008) ।** `P(entity | mention) × context-similarity(entity, text)`. अच्छी तरह से काम करता है, तेजी से, कोई प्रशिक्षण नहीं.
2. **सम्मिलन आधारित (ESS / REL / झपकना .** एनकोड उल्लेख + संदर्भ. प्रत्येक उम्मीदवार के विवरण को एनकोड. अधिकतम कॉसिन चुनें. 2020-2024 डिफ़ॉल्ट।
3. **जनरेटिव (GENRE, 2021; LLM-based, 2023+).** इकाई के कैनोनिक नाम टोकन-दर-टोकन को डिकोड करें। वैध इकाई नामों के एक त्रिज्या तक सीमित है ताकि आउटपुट एक वैध होने की गारंटी हो KB पहचान पत्र।

**ट्यूबलाइन बनाम अंत-से-अंत।** आधुनिक मॉडल (ELQ, BLINK, ExtEnD, GENRE) चलना NER पाइपलाइन सिस्टम अभी भी उत्पादन में हावी हैं क्योंकि आप घटकों को आदान-प्रदान कर सकते हैं।

### दो माप

- **नामित वापसी (कandidate gen)** सोने का अंश जहां सही उल्लेख KB प्रवेश उम्मीदवारों की सूची में दिखाई देता है। पूरे पाइपलाइन के लिए तल।
- **स्पष्टीकरण सटीकता / F1.** सही उम्मीदवारों को देखते हुए, शीर्ष 1 कितनी बार सही होता है।

80% उम्मीदवारों को वापस लेने पर 99% अस्पष्टता वाली प्रणाली 80% पाइपलाइन है।

```figure
gx-entity-linking
```

## इसे बनाओ

### चरण 1: विकिपीडिया रीडायरेक्ट्स से एक उपनाम सूचकांक बनाएं

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

विकिपीडिया उपनाम डेटाः ~18M (असमान, इकाई) जोड़े। विकिडाटा डंप से डाउनलोड करें। उल्टा सूचकांक के रूप में स्टोर करें।

### चरण 2: संदर्भ आधारित असमंजस्य

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

जैकार्ड ओवरलैप एक खिलौना है। एम्बेडमेंट पर कोसिन समानता से प्रतिस्थापित करें (देखें `code/main.py` ट्रांस्फार्मर संस्करण के लिए चरण-2) ।

### चरण 3: एम्बेडिंग आधारित (BLINK-style)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

सूचकांक समय पर, प्रत्येक एम्बेड KB प्रश्न समय में, उल्लेख + संदर्भ एक बार, उम्मीदवार पूल के खिलाफ डॉट-उत्पाद एम्बेड, अधिकतम चुनें.

### चरण 4: जनरेटिव इकाई लिंकिंग (कन्सेप्ट)

GENRE इकाई के विकिपीडिया शीर्षक वर्ण-दर-वर्ण को डिकोड करता है। प्रतिबंधित डिकोडिंग (पाठ 20 देखें) सुनिश्चित करता है कि केवल वैध शीर्षक आउटपुट किए जा सकते हैं। KB-backed आधुनिक वंशज है REL-GEN और LLM-prompted EL संरचित आउटपुट के साथ।

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

एक श्वेतसूची के साथ संयुक्त (आवरण) `choice`), यह सबसे सरल है EL पाइपलाइन 2026 में जहाज करने के लिए।

### चरण 5: मूल्यांकन करें AIDA-CoNLL

AIDA-CoNLL मानक है EL बेंचमार्क: 1,393 रॉयटर्स लेख, 34k उल्लेख, विकिपीडिया संस्थाएं।KB सटीकता (`P@1`) और बाहर-KB NIL-detection दर।

## फंदे

- **NIL संभाल।** कुछ उल्लेखों में नहीं हैं KB (उभरती हुई संस्थाओं, अस्पष्ट लोगों) सिस्टम को भविष्यवाणी करनी चाहिए NIL गलत इकाई की अनुमान लगाने के बजाय अलग से मापा गया।
- **सीमा त्रुटियों का उल्लेख करें।** ऊपर की धारा NER आंशिक अवधि को याद करता है ("बैंक ऑफ अमेरिका" केवल "बैंक" के रूप में टैग किया गया है) । EL याद करने के लिए ड्रॉप।
- **लोकप्रियता पूर्वाग्रह।** प्रशिक्षित प्रणालियों अक्सर संस्थाओं के बारे में अधिक भविष्यवाणी. ML कागज अक्सर बास्केटबॉल जॉर्डन से लिंक करता है।
- **बहुभाषी EL.** अंग्रेजी विकिपीडिया संस्थाओं के लिए चीनी पाठ में मैपिंग का उल्लेख करना। एक बहुभाषी एन्कोडर या अनुवाद चरण की आवश्यकता होती है।
- **KB स्थिरता।** नई कंपनियां, घटनाएं, लोग पिछले साल के विकिपीडिया डंप में नहीं हैं। उत्पादन पाइपलाइनों को एक ताज़ा लूप की आवश्यकता है।

## इसका प्रयोग करें

2026 स्टैकः

| स्थिति | चुनें |
|-----------|------|
| General-purpose English + Wikipedia | BLINK या REL |
| पार-भाषाई, KB = Wikipedia | mGENRE |
| LLM-friendly, कुछ उल्लेख/दिन | शीघ्र Claude/GPT-4 उम्मीदवार सूची + सीमित JSON |
| डोमेन विशिष्ट KB (medical, legal) | कस्टम BERT के साथ KB-aware retrieval + fine-tune on domain AIDA-style सेट |
| अत्यंत कम विलंबता | केवल सटीक मैच पूर्व (मिलन-विटन बेसलाइन) |
| अनुसंधान SOTA | GENRE / ExtEnD / जनरेटिव LLM-EL |

2026 में जहाजों का उत्पादन पैटर्नः NER → कोरफ → EL प्रत्येक उल्लेख पर → एक समूह के लिए एक कैनोनिक इकाई में गिरने क्लस्टर। आउटपुटः एक KB दस्तावेज़ में प्रति इकाई आईडी, प्रति उल्लेख नहीं।

## इसे भेजें

के रूप में सहेजें `outputs/skill-entity-linker.md`:

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## व्यायाम

1. **- आराम से।** पूर्व + संदर्भ भेदभाव को लागू करें `code/main.py` 10 अस्पष्ट उल्लेखों पर (पेरिस, जॉर्डन, Apple) सही इकाई को हाथ से लेबल करें। सटीकता मापें।
2. **मध्यम।** एक वाक्य ट्रांसफार्मर के साथ 50 अस्पष्ट उल्लेखों को एन्कोड करें। प्रत्येक उम्मीदवार के विवरण को एम्बेड करें। एम्बेड-आधारित असंबद्धता की तुलना जैकार्ड संदर्भ ओवरलैप से करें।
3. **कठिन.** 1k इकाई डोमेन बनाएं KB (उदाहरण के लिए, आपकी कंपनी में कर्मचारी + उत्पाद) NER + EL अंत से अंत तक. 100 लंबे समय तक किए गए वाक्य पर सटीकता मापें और याद रखें।

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| इकाई को जोड़ना (EL) | विकिपीडिया का लिंक | एक अद्वितीय के लिए एक उल्लेख नक्शा KB प्रवेश। |
| उम्मीदवार पीढ़ी | यह कौन हो सकता है? | संभव की एक शॉर्टलिस्ट लौटाएं KB उल्लेख के लिए प्रविष्टियाँ। |
| स्पष्टीकरण | सही चुनें | संदर्भ का उपयोग करके उम्मीदवारों को स्कोर करें, विजेता चुनें। |
| उपनाम सूचकांक | खोज तालिका | सतह से नक्शा → उम्मीदवार संस्थाओं के रूप में। |
| NIL | नहीं में KB | स्पष्ट भविष्यवाणी कि नहीं KB प्रवेश मैच। |
| KB | ज्ञान आधार | विकिडेटा, विकिपीडिया, DBpedia, या अपने डोमेन KB. |
| AIDA-CoNLL | बेंचमार्क | 1,393 रॉयटर्स लेखों के साथ सोने के इकाई लिंक. |

## आगे पढ़ना

- [मिलन, विटन (2008). विकिपीडिया से लिंक करना सीखना](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) मौलिक पूर्व + संदर्भ दृष्टिकोण।
- [Wu et al. (2020). शून्य शॉट इकाई लिंकिंग से घने इकाई रिट्रीवल (BLINK)](https://arxiv.org/abs/1911.03814) एम्बेडिंग आधारित कार्यघोड़ा।
- [डी काओ एट अल. (2021).GENRE)](https://arxiv.org/abs/2010.00904) जनरेटिव EL सीमित डिकोडिंग के साथ।
- [Hoffart et al. (2011). पाठ में नामित संस्थाओं की मजबूत असंगति (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) बेंचमार्क पेपर।
- [REL: एक इकाई लिंकर जो दिग्गजों के कंधों पर खड़ा है (2020)](https://arxiv.org/abs/2006.01969) खुले उत्पादन स्टैक।
