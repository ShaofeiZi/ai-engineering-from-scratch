# प्राकृतिक भाषा का तर्क  पाठ संबंधी सम्मिलन

> "t h को शामिल करता है" का अर्थ है कि मानव पढ़ना t निष्कर्ष निकालता है h सही है। NLI उपरीक्षा पर बोरिंग, उत्पादन में लोड-बहन।

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**Time:** ~60 minutes

## समस्या

आपने एक सारांशक बनाया, जिसने सारांश दिया. आप कैसे जानते हैं कि सारांश में कोई पगड़ी नहीं है?

आपने एक चैटबॉट बनाया. उसने "हाँ" कहा. आप कैसे जानते हैं कि उत्तर को प्राप्त हुए passage द्वारा समर्थित है?

आपको 10,000 समाचार लेखों को विषय के अनुसार वर्गीकृत करने की आवश्यकता है. आपके पास प्रशिक्षण लेबल नहीं हैं. क्या आप एक मॉडल का पुनः उपयोग कर सकते हैं?

ये तीनों समस्याएं प्राकृतिक भाषा के तर्क में ही आती हैं। NLI प्रश्न करता हैः एक आधार दिया गया है `t` और एक परिकल्पना `h`, है `h` द्वारा उत्पन्न `t`, विरोधाभासी या तटस्थ (गैर-संबंधित)?

- **प्यास की जाँचः** `t` = स्रोत दस्तावेज, `h` = संक्षेप में दावा। नहीं entailment = hallucination.
- **जमीनी QA:** `t` = प्राप्त मार्ग, `h` = उत्पन्न उत्तर। नहीं entailment = fabrication.
- **शून्य शॉट वर्गीकरणः** `t` = दस्तावेज, `h` = शब्दबद्ध लेबल ("यह खेल के बारे में है") । Entailment = predicted लेबल।

एक कार्य, तीन उत्पादन उपयोग। यही कारण है कि हर RAG मूल्यांकन ढांचे के जहाजों और NLI हुड के नीचे मॉडल.

## अवधारणा

![NLI: तीनतरफा वर्गीकरण, प्रमेय बनाम परिकल्पना](../assets/nli.svg)

**तीनों लेबल.**

- **सम्मिलितता।** `t` → `h`"मक्खी गद्दे पर है" का अर्थ है "एक बिल्ली है।
- **विरोधाभास।** `t` → ¬`h`. "मक्खी मैट पर है" "कोई बिल्ली नहीं है" के विपरीत है.
- **तटस्थ।** "मक्खी मैट पर है" "मक्खी भूख लगी है" से तटस्थ है।

**कोई तार्किक निष्कर्ष नहीं।** NLI है *प्राकृतिक* भाषा का निष्कर्ष  एक विशिष्ट मानव पाठक क्या निष्कर्ष होगा, सख्त तर्क नहीं. "जॉन अपने कुत्ते के साथ चला" के साथ "जॉन एक कुत्ते है" NLI, लेकिन सख्त प्रथम श्रेणी तर्क केवल यह स्वीकार करेगा यदि आप axiomatize स्वामित्व.

**डेटासेट।**

- **SNLI** (2015). 570k मानव-विवरण जोड़े, छवि कैप्शन के रूप में स्थान। संकीर्ण डोमेन।
- **MultiNLI** 2026 में मानक प्रशिक्षण पाठ्यक्रम।
- **ANLI** (2019). प्रतिकूल NLI. मनुष्य ने उदाहरण लिखे हैं जो विशेष रूप से मौजूदा मॉडल को तोड़ने के लिए डिज़ाइन किए गए हैं।
- **DocNLI, ConTRoL** (202021). दस्तावेज-लंबाई के स्थान। बहु-हॉप और लंबी दूरी के निष्कर्ष का परीक्षण।

**वास्तुकला।** एक ट्रांसफार्मर एन्कोडर (BERT, RoBERTa, DeBERTa) पढ़ता है `[CLS] premise [SEP] hypothesis [SEP]`. . . `[CLS]` प्रतिनिधित्व एक 3-तरफा softmax खिलाता है। MNLI, बनाए गए बेंचमार्क पर मूल्यांकन करें, वितरण जोड़े पर 90% से अधिक सटीकता प्राप्त करें।

**शून्य शॉट के माध्यम से NLI.** एक दस्तावेज़ और उम्मीदवार लेबल दिए जाने पर, प्रत्येक लेबल को एक परिकल्पना में बदल दें ("यह पाठ खेल के बारे में है") । प्रत्येक के लिए सम्मिलित होने की संभावना की गणना करें। अधिकतम चुनें। यह Hugging Face के पीछे तंत्र है `zero-shot-classification` पाइपलाइन।

```figure
nli-router
```

## इसे बनाओ

### चरण 1: पूर्व-प्रशिक्षित चलाएं NLI मॉडल

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

उत्पादन के लिए NLI, `facebook/bart-large-mnli` और `microsoft/deberta-v3-large-mnli` खुले डिफ़ॉल्ट हैं। DeBERTa-v3 शीर्ष स्थानों पर।

### चरण 2: शून्य शॉट वर्गीकरण

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

टेम्पलेट "यह उदाहरण के बारे में है {लेबल}." डिफ़ॉल्ट रूप से है। के साथ अनुकूलित करें `hypothesis_template`कोई प्रशिक्षण डेटा की आवश्यकता नहीं है, कोई सूक्ष्म समायोजन नहीं है.

### चरण 3: वफादारी की जांच करें RAG

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

यह मूल है RAGAS निष्ठा. उत्पन्न उत्तर को परमाणु दावों में विभाजित करें. प्रत्येक दावों को प्राप्त संदर्भ के साथ जांचें. जो अंश शामिल है, रिपोर्ट करें।

### चरण 4: हाथ से रोल किया गया NLI वर्गीकरण (संदर्भात्मक)

देखिये `code/main.py` केवल स्ट्डलिब के लिए खिलौना के लिएः प्रमेय और परिकल्पना की तुलना लेक्सिकल ओवरलैप + नकारण का पता लगाने के माध्यम से की जाती है। ट्रांसफार्मर मॉडल के साथ प्रतिस्पर्धी नहीं है  लेकिन यह कार्य का आकार दिखाता हैः दो पाठों में, 3-तरफा लेबल बाहर, loss = cross- एंट्रॉपी खत्म `{entail, contradict, neutral}`.

## फंदे

- **केवल परिकल्पना शॉर्टकट।** मॉडल केवल परिकल्पना से लेबल की भविष्यवाणी कर सकते हैं ~ 60% पर SNLI क्योंकि "नहीं", "कोई नहीं", "कभी नहीं" विरोधाभास के साथ संबद्ध हैं। लेबल लीक का पता लगाने के लिए मजबूत आधार।
- **लक्ज़िकल ओवरलैप हेउरिस्टिक।** उपक्रम हेरिस्टिक ("प्रत्येक उपक्रम शामिल है") पास SNLI लेकिन असफलता HANS/ANLI. प्रतिकूल बेंचमार्क का प्रयोग करें।
- **दस्तावेज लंबाई में गिरावट।** एकल वाक्य NLI models drop 20+ F1 दस्तावेज लंबाई के परिसरों पर। DocNLI-trained दीर्घ संदर्भ के लिए मॉडल।
- **शून्य शॉट टेम्पलेट संवेदनशीलता।** "यह उदाहरण के बारे में है {लेबल}" बनाम "{लेबल}" बनाम "विषय है {लेबल}" सटीकता 10+ अंक स्विंग कर सकते हैं। टेम्पलेट ट्यून करें।
- **डोमेन असंगतता.** MNLI सामान्य अंग्रेजी पर ट्रेनें। कानूनी, चिकित्सा और वैज्ञानिक पाठ को डोमेन-विशिष्ट आवश्यकता होती है NLI मॉडल (जैसे, SciNLI, MedNLI).

## इसका प्रयोग करें

2026 स्टैकः

| उपयोग के मामले | मॉडल |
|---------|-------|
| सामान्य प्रयोजन NLI | `microsoft/deberta-v3-large-mnli` |
| तेज / किनारा | `cross-encoder/nli-deberta-v3-base` |
| शून्य शॉट वर्गीकरण (हल्का वजन) | `facebook/bart-large-mnli` |
| दस्तावेज स्तर NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| बहुभाषी | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| हलूसिनेशन का पता लगाना RAG | NLI अंदर की परत RAGAS / DeepEval |

2026 मेटा-पैटर्नः NLI जब भी आपको "क्या A B का समर्थन करता है?" या "क्या A B के विपरीत है?" की आवश्यकता होती है NLI और दूसरे के लिए हाथ उठाए LLM फोन करें।

## इसे भेजें

के रूप में सहेजें `outputs/skill-nli-picker.md`:

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## व्यायाम

1. **- आराम से।** दौड़ें `facebook/bart-large-mnli` 20 हस्तनिर्मित (प्रिमिसेस, परिकल्पना, लेबल) ट्रिपल पर तीनों वर्गों को कवर करें। सटीकता मापें। प्रतिकूल "उपक्रम हेरिस्टिक" जाल ("मैंने केक नहीं खाया" बनाम "मैंने केक खाया") जोड़ें और देखें कि क्या यह टूटता है।
2. **मध्यम।** शून्य शॉट टेम्पलेट की तुलना करें `"This text is about {label}"` विरोध `"The topic is {label}"` और `"{label}"` 100 पर AG समाचार के शीर्षक, सटीकता की रिपोर्ट।
3. **कठिन.** एक निर्माण RAG निष्ठा परीक्षकः परमाणु-ध claim विघटन + NLI प्रति दावा 50 पर मूल्यांकन RAG-generated गलत-सकारात्मक और गलत-नकारात्मक दरों को हाथ लेबल के साथ मापें।

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| NLI | प्राकृतिक भाषा का तर्क | प्रमेय-अनुमान संबंधी संबंध का तीन-तरफा वर्गीकरण। |
| RTE | पाठ की संबद्धता को पहचानना | पुराने नाम के लिए NLI; एक ही कार्य। |
| सम्मिलित | "t" का अर्थ है "h" | एक साधारण पाठक यह निष्कर्ष निकालता है कि h सही है क्योंकि t है। |
| विरोधाभास | "t बाहर निकालने h" | एक साधारण पाठक यह निष्कर्ष निकालता है कि h गलत है क्योंकि t है। |
| तटस्थ | "अनिर्णय" | किसी भी तरह से t से h तक कोई निष्कर्ष नहीं है। |
| शून्य शॉट वर्गीकरण | NLI वर्गीकरणकर्ता के रूप में | लेबल को परिकल्पना के रूप में शब्दबद्ध करें, अधिकतम संबद्धता चुनें। |
| वफादार | क्या इसका उत्तर समर्थित है? | NLI over (बढ़वाया गया संदर्भ, उत्पन्न उत्तर) । |

## आगे पढ़ना

- [Bowman et al. (2015) प्राकृतिक भाषा inference सीखने के लिए एक बड़ा नोटित corpus](https://arxiv.org/abs/1508.05326) — SNLI.
- [विल्यम्स, नंगिया, बोमैन (2017) । इन्फेरेंस के माध्यम से वाक्य समझने के लिए एक व्यापक कवरेज चुनौती कॉर्पस](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). प्रतिकूल NLI](https://arxiv.org/abs/1910.14599)  ANLI बेंचमार्क।
- [Yin, Hay, Roth (2019). बेंचमार्किंग शून्य-शॉट टेक्स्ट वर्गीकरण](https://arxiv.org/abs/1909.00161) — NLI-as-classifier.
- [वह और अन्य (2021). DeBERTa: डिकोडिंग-सुधार BERT विघटित ध्यान के साथ](https://arxiv.org/abs/2006.03654) 2026 NLI काम का घोड़ा।
