# संबंध निष्कर्षण और ज्ञान ग्राफ निर्माण

> NER एक ज्ञान ग्राफ नोड्स, किनारों और उनकी उत्पत्ति का योग है। एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान ग्राफ के रूप में, एक ज्ञान के रूप में, एक ज्ञान के रूप में, एक ज्ञान के रूप में, एक ज्ञान के रूप में, एक ज्ञान के रूप में, एक ज्ञान के रूप में, एक ज्ञान के रूप में, एक ज्ञान में, एक ज्ञान के रूप में, एक ज्ञान में, एक ज्ञान के रूप में, एक ज्ञान में, एक ज्ञान में, एक ज्ञान के रूप में, एक रूप में, एक रूप में, एक रूप में, एक रूप में, एक रूप में, एक रूप में, एक रूप में, एक रूप में, एक रूप में, एक रूप में, एक रूप में, एक रूप में, एक रूप में, में, एक रूप में, एक रूप में, में, में, एक रूप में, में, एक रूप में, में, में, एक रूप में, में, एक रूप में, में, एक रूप में, में, में, में, में, एक रूप में, में, में, में, में, एक रूप में, में, में, में, में, में, में, में, में, एक रूप में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में, में

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**Time:** ~60 minutes

## समस्या

एक विश्लेषक ने कहाः "टीम कुक CEO के Apple 2011 में" चार तथ्यः

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

संबंध निकासी (RE) मुक्त पाठ को संरचित तिगुना में बदल देता है `(subject, relation, object)`एक corpus के माध्यम से संकलित करें और आप एक ज्ञान ग्राफ है. संकलित और क्वेरी और आप एक तर्क सब्सट्रेट है RAG, विश्लेषण, या अनुपालन लेखा परीक्षा।

2026 की समस्याः LLMs एक बार जब आप एक व्यक्ति के साथ एक रिश्ते को जोड़ते हैं, तो आप एक व्यक्ति के साथ एक रिश्ते को जोड़ते हैं। AEVS-style एंकर-और-सत्यापन पाइपलाइनों।

## अवधारणा

![पाठ → त्रिगुट → ज्ञान ग्राफ](../assets/relation-extraction.svg)

**तीन प्रकार का।** `(subject_entity, relation_type, object_entity)`. संबंध एक बंद ओंटोलॉजी (विकिडेटा गुण, FIBO, UMLS) या एक खुला सेट (OpenIE-style, कुछ भी चला जाता है).

**तीन निकासी दृष्टिकोण.**

1. **नियम / पैटर्न आधारित।** हर्स्ट पैटर्नः "X जैसे Y" → `(Y, isA, X)`और हाथ से बना रेजेक्स, भंगुर, सटीक, स्पष्ट।
2. **पर्यवेक्षित वर्गीकरण।** एक वाक्य में दो इकाई उल्लेखों को देखते हुए, एक निश्चित सेट से संबंध की भविष्यवाणी करें। TACRED, ACE, KBP. मानक 20152022.
3. **जनरेटिव LLM.** मॉडल को तीनों को उत्सर्जित करने के लिए प्रेरित करें। यह बॉक्स से बाहर काम करता है। मूल की आवश्यकता है, या भ्रमों को सापेक्ष दिखने वाले कचरे।

**AEVS (अंकर-एक्सट्रैक्शन-वेरिफिकेशन-संपूरक, 2026) ।** वर्तमान भ्रम-शमन-शमन ढांचाः

- **एंकर.** प्रत्येक इकाई अवधि और संबंध-वक्तों अवधि को सटीक स्थानों के साथ पहचानें।
- **निकालें।** एंकर स्पैन से जुड़े ट्रिपल उत्पन्न करें।
- **जाँच करें.** प्रत्येक त्रिकोणीय तत्व को स्रोत पाठ के साथ मेल खाता है; असमर्थित किसी भी चीज़ को खारिज कर दें।
- **पूरक।** एक कवरेज पास यह सुनिश्चित करता है कि कोई लंगर span गिर नहीं जाता है।

भ्रामकता में तेजी से गिरावट आती है, अधिक गणना की आवश्यकता होती है, लेकिन यह लेखा परीक्षा योग्य है।

**खुले बनाम बंद व्यापार।**

- **बंद ऑन्टोलॉजी।** फिक्स्ड प्रॉपर्टी लिस्ट (जैसे, विकिडाटा के 11,000+ प्रॉपर्टीज) । पूर्वानुमान योग्य। खोज योग्य। आविष्कार करना मुश्किल है।
- **खुला IE.** किसी भी मौखिक वाक्यांश एक रिश्ते बन जाता है उच्च याददाश्त कम सटीकता, गड़बड़ पूछने के लिए.

उत्पादन KGs आम तौर पर मिलाएंः खुला IE खोज के लिए, फिर मुख्य ग्राफ में विलय से पहले एक बंद ontology पर संबंधों को कैनोनिकलाइज़ करें।

```figure
relation-triples
```

## इसे बनाओ

### चरण 1: पैटर्न आधारित निकासी

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

देखिये `code/main.py` हर्स्ट पैटर्न अभी भी डोमेन विशिष्ट पाइपलाइन में जहाज क्योंकि वे डिबग करने योग्य हैं।

### चरण 2: पर्यवेक्षित संबंध वर्गीकरण

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL एक है seq2seq संबंध निकालनेः पाठ में, बाहर, पहले से ही विकिडाटा संपत्ति आईडी में. दूरस्थ निगरानी डेटा पर ठीक से समायोजित. मानक खुला वजन आधार।

### चरण 3: LLM-prompted जंजीर के साथ निकासी

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

स्रोत के खिलाफ हर लौटा span सत्यापित करें. `text[start:end] != triple_entity`. . यह है AEVS "जाँच" कदम अपने न्यूनतम रूप में.

### चरण 4: बंद ओंटोलॉजी पर कैनोनिकलाइज़ करें

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

कैनोनिकेशन अक्सर इंजीनियरिंग काम का 60-80% है। इसके लिए बजट।

### चरण 5: एक छोटे से ग्राफ और क्वेरी बनाएं

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

यह प्रत्येक के परमाणु है RAG-over-KG प्रणाली. इसे स्केल RDF तीन दुकानें (ब्लेज़ेग्राफ, वर्चुओसो), संपत्ति ग्राफ (Neo4j), या वेक्टर-वृद्धि ग्राफ स्टोर।

## फंदे

- **पूर्व में कोरफ़ेरेंस RE.** "उसने स्थापना की Apple" — RE पहले कोरफ (पाठ 24) को चलाएं।
- **इकाई के कैनोनिकेशन.** "Apple इंक" और "Apple" को उसी नोड पर हल करना चाहिए। पहले जोड़ने वाली इकाई (पाठ 25) ।
- **तीन बार पगला।** LLMs तीन बार emit करें पाठ समर्थन नहीं करता है। स्पैन सत्यापन लागू करें।
- **संबंध कैनोनिकेशन बहाव।** खुला IE संबंध असंगत हैं ("जन्म में," "आता है," "आता है, के मूल निवासी है") । कैनोनिकल आईडी या ग्राफ में गिरावट अपरिहार्य है।
- **समय की त्रुटियां।** "टीम कुक है CEO के Apple"  वर्तमान में सही, 2005 में गलत। कई संबंध समय सीमाबद्ध हैं। योग्यताओं का उपयोग करें (`P580` प्रारंभ समय, `P582` विकिडाटा में अंत समय) ।
- **डोमेन असंगतता.** REBEL कानूनी, चिकित्सा और वैज्ञानिक पाठ को अक्सर डोमेन-अच्छी तरह से ट्यून करने की आवश्यकता होती है RE मॉडल।

## इसका प्रयोग करें

2026 स्टैकः

| स्थिति | चुनें |
|-----------|------|
| तेजी से उत्पादन, सामान्य डोमेन | REBEL या LlamaPred विकिडाटा के साथ कैनोनिकेशन |
| डोमेन-विशिष्ट (बायोमेडिक, कानूनी) | SciREX-style domain fine-tune + custom ontology |
| LLM-prompted, लेखापरीक्षण उत्पादन | AEVS पाइपलाइनः एंकर → एक्सट्रैक्ट → सत्यापित → पूरक |
| बड़ी मात्रा में समाचार IE | Pattern-based + supervised hybrid |
| एक निर्माण KG खरोंच से | खुला IE + manual canonicalization pass |
| समय KG | क्वालीफायर के साथ निकासी (शुरु/अंत समय, समय में बिंदु) |

एकीकरण पैटर्नः NER → कोरफ → इकाई जोड़ने → संबंध निष्कर्षण → ओंटोलॉजी मैपिंग → ग्राफ लोड। प्रत्येक चरण एक संभावित गुणवत्ता गेट है।

## इसे भेजें

के रूप में सहेजें `outputs/skill-re-designer.md`:

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## व्यायाम

1. **- आराम से।** पैटर्न एक्सट्रैक्टर को चालू करें `code/main.py` 5 समाचार लेख वाक्य पर. हाथ से जांच सटीकता.
2. **मध्यम।** उपयोग REBEL (या एक छोटी LLMतीनों की तुलना करें. कौन सा निकालनेवाला अधिक सटीक है?
3. **कठिन.** निर्माण AEVS पाइपलाइनः निकासी के साथ LLM + स्रोत के खिलाफ स्पैन्स सत्यापित करें. 50 विकिपीडिया शैली के वाक्य पर सत्यापन चरण से पहले बनाम बाद में पग्लुसीनेशन दर मापें।

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| तीन | विषय-संबंध-वस्तु | `(s, r, o)` tuple जो एक परमाणु इकाई है KG. |
| खुला IE | कुछ भी निकालें | खुले शब्दावली संबंध वाक्यांश; उच्च याद, कम सटीकता। |
| बंद ओंटोलॉजी | फिक्स्ड स्कीम | संबंध प्रकारों का सीमित सेट (विकिडेटा, UMLS, FIBO). |
| कैनोनिककरण | सब कुछ सामान्य करें | मैप सतह के नाम / कैनोनिकल आईडी के संबंध। |
| AEVS | जमीनी निकासी | एंकर-एक्सट्रैक्शन-वेरिफिकेशन-सप्लाई पाइपलाइन (2026). |
| उत्पत्ति | सत्य के स्रोत से लिंक | प्रत्येक त्रिगुट अपने स्रोत के लिए एक डॉक्यूमेंट आईडी + चार-स्पेन ले जाता है। |
| दूरस्थ निगरानी | सस्ते लेबल | मौजूदा पाठ के साथ पाठ को संरेखित करें KG प्रशिक्षण डेटा बनाने के लिए। |

## आगे पढ़ना

- [मिंट्ज़ और अन्य (2009) । बिना लेबल किए गए डेटा के संबंध निकासी के लिए दूरस्थ निगरानी](https://www.aclweb.org/anthology/P09-1113.pdf) दूरस्थ निगरानी के लिए कागज।
- [हुगेट कैबोट, नेविगली (2021). REBEL: अंत-से-अंत भाषा पीढ़ी द्वारा संबंध निकासी](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE काम का घोड़ा।
- [वाडेन एट अल. (2019). संदर्भागत स्पैन प्रतिनिधित्वों के साथ इकाई, संबंध और घटना निष्कर्षण (DyGIE++)](https://arxiv.org/abs/1909.03546) संयुक्त IE.
- [AEVS एंकर-अवहरण-सत्यापन-पूरीकरण ढांचा](https://www.mdpi.com/2073-431X/15/3/178) 2026 में पगडंडी-मटाईकरण डिजाइन।
- [विकिडेटा SPARQL ट्यूटोरियल](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) कैनोनिक ग्राफ क्वेरी।
