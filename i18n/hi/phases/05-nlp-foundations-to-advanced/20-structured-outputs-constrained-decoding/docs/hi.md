# संरचित आउटपुट और प्रतिबंधित डिकोडिंग

> एक पूछें LLM के लिए JSON. जाओ JSON उत्पादन में, "अधिकतर" समस्या है। सीमित डिकोडिंग "अधिकतर" को "हमेशा" में बदल देता है नमूना लेने से पहले लॉजिट को संपादित करके।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## समस्या

एक वर्गीकरण एक LLM: "एक {सकारात्मक, नकारात्मक, तटस्थ} के एक लौटाएं।" मॉडल लौटाता है "भावना सकारात्मक है  यह समीक्षा जबरदस्त रूप से अनुकूल है क्योंकि ग्राहक स्पष्ट रूप से कहते हैं कि वे ...। " आपका पार्सर दुर्घटनाग्रस्त हो जाता है। F1 0.0 है।

मुक्त रूप उत्पादन एक अनुबंध नहीं है, यह एक सुझाव है। एक उत्पादन प्रणाली को अनुबंध की आवश्यकता है।

2026 में तीन परतें मौजूद हैं।

1. **जल्दी से.** "केवल वापस करने के लिए JSON वस्तु. " सीमा मॉडल पर ~ 80% काम करता है, छोटे मॉडल पर कम।
2. **मूल संरचनात्मक उत्पादन APIs.** OpenAI `response_format`, Anthropic उपकरण का उपयोग, Gemini JSON समर्थित योजनाओं पर विश्वसनीय. विक्रेता द्वारा लॉक।
3. **प्रतिबंधित डिकोडिंग।** प्रत्येक पीढ़ी के कदम पर लॉजिट को संशोधित करें ताकि मॉडल *नहीं कर सकते* 100% निर्माण द्वारा मान्य है। किसी भी स्थानीय मॉडल पर काम करता है।

यह सबक तीनों के लिए अंतर्ज्ञान का निर्माण करता है और नाम देता है कि किसके लिए पहुंचना है।

## अवधारणा

![प्रत्येक चरण में अमान्य टोकन को छिपाने वाले प्रतिबंधित डिकोडिंग](../assets/constrained-decoding.svg)

**कैसे सीमित डिकोडिंग काम करता है।** प्रत्येक पीढ़ी के कदम पर, LLM पूर्ण शब्दावली (~ 100k टोकन) पर एक लॉजिट वेक्टर उत्पन्न करता है। *लॉजिट प्रोसेसर* मॉडल और नमूना लेने वाले के बीच बैठता है। यह लक्ष्य व्याकरण में वर्तमान स्थिति को देखते हुए कौन से टोकन मान्य हैं JSON स्कीमा, रेजेक्स, संदर्भ मुक्त व्याकरण  और सभी अमान्य टोकन के लॉजिट को नकारात्मक अनंत पर सेट करता है। शेष लॉजिट पर सॉफ्टमैक्स केवल वैध निरंतरताओं पर संभावना द्रव्यमान रखता है।

2026 में कार्यान्वयनः

- **रेखाचित्र।** संकलन JSON एक अंत-राज्य मशीन में योजना या regex. प्रत्येक टोकन एक O  1) मान्य-अगले टोकन खोज प्राप्त करता है. FSM-based, तो पुनरावर्ती योजनाओं को सपाट करने की जरूरत है।
- **XGrammar / मार्गदर्शन.** संदर्भ मुक्त व्याकरण इंजन। संभाल पुनरावर्ती JSON योजना, शून्य के करीब ओवरहेड डिकोडिंग। OpenAI 2025 में उनके संरचित उत्पादन कार्यान्वयन में मार्गदर्शन का श्रेय दिया गया।
- **vLLM निर्देशित डिकोडिंग।** अंतर्निहित `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` रेखांकन के माध्यम से, XGrammar, या आईएम-फॉर्मेट-इंफोर्सर बैकेंड्स.
- **प्रशिक्षक.** किसी भी पर पाइडान्टिक आधारित रैपर LLM. सत्यापन विफलता पर पुनः प्रयास करें। क्रॉस-प्रोवाइडर, लेकिन लॉग्स को संशोधित नहीं करता है  यह पुनः प्रयासों + संरचित-आउटपुट-जागरूक संकेतों पर निर्भर करता है।

### विपरीत परिणाम

प्रतिबंधित डिकोडिंग अक्सर *तेजी से* एक और कारण है कि यह एक और विकल्प है जो एक और विकल्प है, जो कि एक और विकल्प है। `{"name": "` प्रत्येक बाइट निर्धारित किया गया है) ।

### उस जाल में जो आपको खर्च करता है

क्षेत्र आदेश मायने रखता है. `answer` पहले `reasoning`, और मॉडल सोचने से पहले एक जवाब देने के लिए प्रतिबद्ध है। JSON कोई भी सत्यापन इसे पकड़ता है।

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

स्कीमा क्षेत्र क्रम तर्क है, स्वरूपण नहीं।

```figure
constrained-decoder
```

## इसे बनाओ

### चरण 1: रेजेक्स-सीमित पीढ़ी खरोंच से

देखिये `code/main.py` एक स्वतंत्र व्यक्ति के लिए FSM 30 पंक्तियों में मूल विचारः

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

इन FSM यह पता चलता है कि हमने अभी तक व्याकरण के किन हिस्सों को संतुष्ट किया है। `valid_tokens(state, tokenizer)` गणना करता है कि कौन से शब्दावली टोकन आगे बढ़ सकते हैं FSM और न किसी मार्ग से निकलते

### चरण 2: योजनाओं के लिए रूपरेखा JSON योजना

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

शून्य सत्यापन त्रुटियों. कभी नहीं. FSM अमान्य आउटपुट को अछूता बनाता है।

### चरण 3: प्रदाता-अज्ञानी Pydantic के लिए प्रशिक्षक

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

अलग तंत्र. प्रशिक्षक लॉजिट को छूता नहीं है। यह स्कीमा को प्रॉम्प्ट में प्रारूपित करता है, आउटपुट को पार्स करता है, और सत्यापन विफलता पर पुनः प्रयास करता है (पूर्वनिर्धारित 3 बार) । किसी भी प्रदाता के साथ काम करता है। पुनः प्रयासों में देरी और लागत बढ़ जाती है। क्रॉस-प्रोवाइडर पोर्टेबिलिटी बिक्री बिंदु है।

### चरण 4: मूल विक्रेता APIs

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

सर्वर-साइड प्रतिबंधित डिकोडिंग समर्थित योजनाओं के लिए रूपरेखा के साथ विश्वसनीयता समानता कोई स्थानीय मॉडल प्रबंधन नहीं है। आप आपूर्तिकर्ता के लिए लॉक करता है।

## फंदे

- **पुनरावर्ती योजनाएं।** रेखाचित्र एक निश्चित गहराई तक पुनरावृत्ति को सपाट करता है। पेड़-संरचित आउटपुट (निस्ट टिप्पणी, AST) आवश्यकता XGrammar या मार्गदर्शन (CFG-based).
- **विशाल enums.** 10,000-विकल्प एनयूएम धीमी गति से या समय से संकलित करता है। एक रिट्रीवर पर स्विच करेंः पहले शीर्ष-के उम्मीदवारों की भविष्यवाणी करें, उन पर प्रतिबंध लगाएं।
- **व्याकरण बहुत सख्त है।** बल `date: "YYYY-MM-DD"` रेजेक्स और मॉडल आउटपुट नहीं कर सकते `"unknown"` एक तारीख का आविष्कार करके मॉडल मुआवजा देता है। `null` या एक प्रहरी.
- **समय से पहले प्रतिबद्धता।** ऊपर दिए गए फील्ड ऑर्डर फेल को देखें. हमेशा तर्क को पहले रखें.
- **विक्रेता JSON बिना योजना के मोड।** शुद्ध JSON केवल मोड गारंटी मान्य JSON, वैध नहीं *आपके उपयोग के मामले के लिए*हमेशा एक पूर्ण योजना प्रदान करें।

## इसका प्रयोग करें

2026 स्टैकः

| स्थिति | चुनें |
|-----------|------|
| OpenAI/Anthropic/Google मॉडल, सरल योजना | मूल विक्रेता संरचित आउटपुट |
| कोई भी प्रदाता, Pydantic वर्कफ़्लो, पुनः प्रयासों को सहन कर सकता है | प्रशिक्षक |
| स्थानीय मॉडल, 100% वैधता की आवश्यकता, फ्लैट योजना | परिदृश्य (FSM) |
| स्थानीय मॉडल, पुनरावर्ती योजना | XGrammar या मार्गदर्शन |
| स्व-होस्ट किए गए अनुमान सर्वर | vLLM निर्देशित डिकोडिंग |
| पुनः प्रयासों के साथ बैच प्रसंस्करण स्वीकार्य | Instructor + cheapest model |

## इसे भेजें

के रूप में सहेजें `outputs/skill-structured-output-picker.md`:

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## व्यायाम

1. **- आराम से।** छोटे खुले-वजन मॉडल को प्रमोट करें (जैसे, Llama-3.2-3B) बिना बाध्यकारी डिकोडिंग के `Review(sentiment, confidence, evidence_span)`. उस अंश को मापें जो वैध के रूप में विश्लेषण करता है JSON 100 समीक्षाओं पर।
2. **मध्यम।** सार के साथ एक ही कॉर्पस JSON अनुपालन दर, विलंबता और अर्थपूर्ण सटीकता की तुलना करें।
3. **कठिन.** फोन नंबरों के लिए एक regex-बंद डिकोडर को खरोंच से लागू करें (`\d{3}-\d{3}-\d{4}`) 1000 नमूनों पर 0 अमान्य आउटपुट की जांच करें।

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| प्रतिबंधित डिकोडिंग | बल वैध आउटपुट | प्रत्येक पीढ़ी के चरण में अमान्य टोकन लॉगिंग का मुखौटा। |
| लॉजिट प्रोसेसर | जो कुछ भी प्रतिबंधित करता है | कार्यः `(logits, state) -> masked_logits`. |
| FSM | परिमित अवस्था की मशीन | संकलित व्याकरण प्रतिनिधित्व; O(1) मान्य-अगले टोकन खोज। |
| CFG | संदर्भ मुक्त व्याकरण | व्याकरण जो पुनरावृत्ति को संभालता है; धीमा लेकिन अधिक अभिव्यक्तिपूर्ण FSM. |
| योजना क्षेत्र क्रम | क्या इससे कोई फर्क पड़ता है? | हाँ  पहला क्षेत्र प्रतिबद्ध करता है; उत्तर से पहले तर्क को हमेशा रखें। |
| निर्देशित डिकोडिंग | vLLMइसका नाम | वही अवधारणा, inference सर्वर में एकीकृत. |
| JSON मोड | OpenAIप्रारंभिक संस्करण | गारंटी JSON संश्लेषण; करता है NOT गारंटी योजना मेल। |

## आगे पढ़ना

- [विलाड, लूफ (2023) LLMs](https://arxiv.org/abs/2307.09702) आउटलाइन पेपर।
- [XGrammar कागज (2024)](https://arxiv.org/abs/2411.15100) तेजी से CFG-based सीमित डिकोडिंग।
- [vLLM संरचित आउटपुट](https://docs.vllm.ai/en/latest/features/structured_outputs.html) इन्फेरेंस सर्वर इंटीग्रेशन।
- [OpenAI संरचित आउटपुट गाइड](https://platform.openai.com/docs/guides/structured-outputs) — API reference + gotchas.
- [प्रशिक्षक पुस्तकालय](https://python.useinstructor.com/) Pydantic + प्रदाताओं के बीच पुनः प्रयास करता है।
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) बेंचमार्किंग 6 प्रतिबंधित डिकोडिंग फ्रेमवर्क।
