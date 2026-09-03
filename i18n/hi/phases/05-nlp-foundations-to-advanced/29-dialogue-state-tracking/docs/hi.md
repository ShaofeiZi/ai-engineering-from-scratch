# संवाद राज्य ट्रैकिंग

> "मैं उत्तर में एक सस्ता रेस्तरां चाहता हूँ ... वास्तव में इसे मध्यम ... और इतालवी जोड़ने के लिए. " तीन मोड़, तीन राज्य अद्यतन. DST स्लॉट मूल्य के निर्देश को समक्रमण में रखता है ताकि बुकिंग काम करे।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**Time:** ~75 minutes

## समस्या

कार्य उन्मुख संवाद प्रणाली में, उपयोगकर्ता का लक्ष्य स्लॉट-मूल्य जोड़े के सेट के रूप में एन्कोड किया जाता हैः `{cuisine: italian, area: north, price: moderate}`. प्रत्येक उपयोगकर्ता बारी एक स्लॉट जोड़ सकते हैं, बदल सकते हैं, या हटा सकते हैं. सिस्टम को पूरी बातचीत पढ़नी चाहिए और वर्तमान स्थिति को सही ढंग से आउटपुट करना चाहिए.

एक भी स्लॉट गलत हो तो सिस्टम गलत रेस्तरां बुक करता है, गलत उड़ान का समय निर्धारित करता है, या गलत कार्ड चार्ज करता है। DST यह जो उपयोगकर्ता ने कहा और जो बैकएंड निष्पादित करता है के बीच की पिंजरे है।

क्यों यह अभी भी 2026 में मायने रखता है LLMs:

- अनुपालन-संवेदनशील डोमेन (बैंकिंग, स्वास्थ्य सेवा, एयरलाइन बुकिंग) के लिए निर्धारक स्लॉट मानों की आवश्यकता होती है, मुक्त रूप के उत्पादन की नहीं।
- उपकरण उपयोग एजेंटों अभी भी कॉल करने से पहले स्लॉट संकल्प की जरूरत है APIs.
- मल्टी-टर्न सुधार करना जितना लगता है उससे कठिन हैः "वास्तव में नहीं, इसे गुरुवार बनाओ।"

आधुनिक पाइपलाइनः शास्त्रीय DST concepts + LLM extractors + structured-output guardrails.

## अवधारणा

![DST: संवाद इतिहास → स्लॉट-मूल्य स्थिति](../assets/dst.svg)

**कार्य संरचना।** एक योजना डोमेन (रेस्टोरेंट, होटल, टैक्सी) और उनके स्लॉट (खाद्य, क्षेत्र, मूल्य, लोग) को परिभाषित करती है। प्रत्येक स्लॉट खाली हो सकता है, एक बंद सेट से एक मूल्य (मूल्यः {सस्ते, मध्यम, महंगे}) के साथ भरा जा सकता है, या एक मुक्त-रूप मूल्य (नामः "द कॉपर केटल") ।

**दो DST सूत्रों।**

- **वर्गीकरण।** प्रत्येक (स्लॉट, उम्मीदवार_मूल्य) जोड़ी के लिए, हाँ / नहीं की भविष्यवाणी करें। बंद-वोकैबल स्लॉट के लिए काम करता है। 2020 से पहले मानक।
- **पीढ़ी।** संवाद को देखते हुए, मुक्त पाठ के रूप में स्लॉट मान उत्पन्न करें। खुले-शब्द स्लॉट के लिए काम करता है। आधुनिक डिफ़ॉल्ट।

**मीट्रिक.** संयुक्त लक्ष्य सटीकता (JGA)  मोड़ का अंश जहां *हर* स्लॉट सही है. MultiWOZ 2024 में 2.4 प्रतिशत की रैंकिंग 83 प्रतिशत के आसपास है।

**वास्तुकला।**

1. **Rule-based (slot regex + keyword).** संकीर्ण डोमेन के लिए मजबूत आधार। डिबग करने योग्य.
2. **TripPy / BERT-DST.** कॉपी आधारित पीढ़ी BERT एन्कोडिंग।LLM मानक।
3. **LDST (LLaMA + LoRA).** निर्देशों के साथ ट्यून LLM डोमेन स्लॉट के साथ संकेत। ChatGPT-level गुणवत्ता पर MultiWOZ 2.4.
4. **ऑन्टोलॉजी मुक्त (202426).** स्कीमा को छोड़ें; सीधे स्लॉट नाम और मान उत्पन्न करें. खुले डोमेन को संभालता है.
5. **त्वरित + संरचित आउटपुट (202426).** LLM Pydantic योजना + प्रतिबंधित डिकोडिंग के साथ कोड की 5 पंक्तियों, उत्पादन के लिए तैयार.

### क्लासिक विफलता मोड

- **मोड़ों के पार सह-संदर्भ।** "चलो पहले विकल्प के साथ रहने दें. " किस विकल्प को हल करने की जरूरत है.
- **ओवर-लिखे बनाम जोड़ें।** उपयोगकर्ता कहता है "इतालवी जोड़ें।" क्या आप रसोई या जोड़ने की जगह लेते हैं?
- **स्पष्ट पुष्टि।** "OK अच्छा"  क्या यह प्रस्तावित बुकिंग को स्वीकार करता है?
- **सुधार।** "वास्तव में यह 7 बजे बनाओ. " अन्य स्लॉट को साफ किए बिना समय को अपडेट करना चाहिए.
- **पूर्व प्रणाली कथन का संदर्भ।** "हाँ, वह एक। " कौन "वह"?

```figure
n5-slot-tracker
```

## इसे बनाओ

### चरण 1: नियम आधारित स्लॉट एक्सट्रैक्टर

देखिये `code/main.py`. रेजेक्स + पर्यायवाची शब्दकोश संकीर्ण डोमेन में 70% कैनोनिक बयानों को कवर करते हैंः

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

कैनोनिक शब्दावली के बाहर भंगुर. निर्धारक स्लॉट पुष्टि के लिए काम करता है.

### चरण 2: राज्य अद्यतन लूप

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

तीन अपरिवर्तनीयः

- कभी भी उस स्लॉट को रीसेट न करें जिसे उपयोगकर्ता ने छुआ न हो।
- स्पष्ट इनकार ("खाद्य को मत छोड़ो") को स्पष्ट करना चाहिए।
- उपयोगकर्ता सुधार ("वास्तव में ...") को जोड़ने के बजाय ओवरराइट करना चाहिए।

### चरण 3: LLM-driven DST संरचित आउटपुट के साथ

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

प्रशिक्षक + Pydantic एक मान्य राज्य वस्तु की गारंटी देता है. कोई regex, कोई योजना असंगतता, कोई पगड़ी स्लॉट नहीं.

### चरण 4: JGA मूल्यांकन

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

माप: सिस्टम को कितने मोड़ प्राप्त होते हैं ALL स्लॉट सही? MultiWOZ 2.4, शीर्ष 2026 प्रणालियों: 80-83%. आपके डोमेन में सिस्टम को आपके संकीर्ण शब्दावली या LLM मूल रेखा आपको हराती है।

### चरण 5: संभाल सुधार

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

एक पता लगाया सुधार पर, जोड़ने के बजाय अंतिम अद्यतन स्लॉट को ओवरराइट करें. बिना सही पाने के लिए मुश्किल है LLM आधुनिक पैटर्नः हमेशा LLM यह स्वाभाविक रूप से सुधारों को संभालता है।

## फंदे

- **पूर्ण इतिहास पुनर्जनन लागत।** अनुमति देने के लिए LLM प्रति बारी लागत O ((n2) कुल टोकन. कैप इतिहास या पुराने बारी का सारांश.
- **योजना बहाव।** पोस्ट हॉक के बाद नए स्लॉट जोड़ने पुराने प्रशिक्षण डेटा तोड़ता है.
- **मामले की संवेदनशीलता।** "इतालवी" बनाम "इतालवी" बनाम "इतालवी"ITALIAN"  हर जगह सामान्य हो।
- **अप्रत्यक्ष विरासत।** यदि उपयोगकर्ता ने पहले "4 लोगों के लिए" निर्दिष्ट किया है, तो एक अलग समय के लिए एक नया अनुरोध लोगों को साफ नहीं करना चाहिए। हमेशा पूरा इतिहास पास करें।
- **मुक्त रूप बनाम बंद सेट।** नाम, समय और पते को मुक्त रूप से स्लॉट की आवश्यकता होती है; रसोई और क्षेत्र बंद हैं। दोनों को योजना में मिलाएं।

## इसका प्रयोग करें

2026 स्टैकः

| स्थिति | दृष्टिकोण |
|-----------|----------|
| संकीर्ण डोमेन (एक या दो इरादे) | Rule-based + regex |
| व्यापक डोमेन, लेबल किए गए डेटा उपलब्ध | LDST (LLaMA + LoRA पर MultiWOZ-style डेटा) |
| व्यापक डोमेन, कोई लेबल नहीं, प्रो-तैयार | LLM + Instructor + Pydantic schema |
| बोलना / आवाज | ASR + normalizer + LLM-DST |
| बहु-डोमेन बुकिंग प्रवाह | योजना-निर्देशित LLM प्रति डोमेन पाइडान्टिक मॉडल के साथ |
| अनुपालन-संवेदनशील | नियम आधारित प्राथमिक, LLM पुष्टि प्रवाह के साथ वापसी |

## इसे भेजें

के रूप में सहेजें `outputs/skill-dst-designer.md`:

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## व्यायाम

1. **- आराम से।** नियम आधारित राज्य ट्रैकर में निर्माण `code/main.py` 3 स्लॉट (खाद्य, क्षेत्र, कीमत) के लिए परीक्षण 10 हस्तनिर्मित संवादों पर। माप JGA.
2. **मध्यम।** इंस्ट्रक्टर + पायदान्टिक + एक छोटे से साथ एक ही डेटासेट LLM. तुलना करें JGA. सबसे कठिन मोड़ों की जांच करें।
3. **कठिन.** दोनों को लागू करें और मार्गः नियम आधारित प्राथमिक, LLM नियम आधारित होने पर वापसी emits <2 आत्मविश्वास के साथ स्लॉट। JGA और प्रति बारी अनुमान लागत।

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| DST | संवाद स्थिति ट्रैकिंग | संवाद मोड़ों के दौरान स्लॉट-मूल्य का निर्देशांक बनाए रखें। |
| स्लॉट | उपयोगकर्ता इरादे की इकाई | बैक-एंड की जरूरतों (किचन, तिथि) के नामित पैरामीटर। |
| डोमेन | कार्य क्षेत्र | रेस्तरां, होटल, टैक्सी  स्लॉट सेट। |
| JGA | संयुक्त लक्ष्य की सटीकता | मोड़ का अंश जहां हर स्लॉट सही है। |
| MultiWOZ | बेंचमार्क | बहु-क्षेत्र WOZ डेटा सेट; मानक DST मूल्यांकन। |
| ऑन्टोलॉजी मुक्त DST | कोई योजना नहीं | सीधे स्लॉट नाम और मान उत्पन्न करें, कोई निश्चित सूची नहीं। |
| सुधार | "वास्तव में... " | एक पहले से भरा हुआ स्लॉट को ओवरराइट करता है। |

## आगे पढ़ना

- [बुज़ियानोवस्की et al. (2018). MultiWOZ एक बड़े पैमाने पर मल्टी-डोमेन जादूगर-ऑफ-ओज़](https://arxiv.org/abs/1810.00278) कैनोनिक बेंचमार्क।
- [फेंग एट एल्स (2023) LLM-driven संवाद राज्य अनुगमन (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA निर्देशों के लिए ट्यूनिंग DST.
- [हेक और अन्य (2020). TripPy मूल्य स्वतंत्र तंत्रिका संवाद राज्य ट्रैकिंग के लिए एक तीन प्रतिलिपि रणनीति](https://arxiv.org/abs/2005.02877) कॉपी-आधारित DST काम का घोड़ा।
- [किंग, फ्लेनगन (2024). अंत-से-अंत कार्य-उन्मुख संवाद LLMs](https://arxiv.org/abs/2404.10753) — EM-based अनियंत्रित TOD.
- [MultiWOZ रैंकिंग बोर्ड](https://github.com/budzianowski/multiwoz) कैनोनिक DST परिणाम।
