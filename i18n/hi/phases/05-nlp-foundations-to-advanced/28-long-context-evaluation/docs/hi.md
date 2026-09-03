# दीर्घ संदर्भ मूल्यांकन NIAH, RULER, LongBench, MRCR

> Gemini 3 प्रो संदर्भ के 10M टोकन का विज्ञापन करता है। 1M टोकन पर, 8-नाल MRCR विज्ञापन ≠ उपयोग करने योग्य। दीर्घ संदर्भ मूल्यांकन आपको बताता है कि आप जिस मॉडल पर शिपिंग कर रहे हैं उसकी वास्तविक क्षमता क्या है।

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 23 (Chunking Strategies)
**Time:** ~60 minutes

## समस्या

आपके पास 200 पृष्ठों का अनुबंध है। मॉडल 1M टोकन संदर्भ का दावा करता है। आप अनुबंध को पेस्ट करते हैं और पूछते हैंः "समाप्ति खंड क्या है?" मॉडल जवाब देता है  लेकिन कवर पेज से जवाब देता है क्योंकि समापन खंड 120k टोकन की गहराई पर बैठता है, जहां मॉडल वास्तव में भाग लेता है।

यह 2026 संदर्भ क्षमता अंतर है। स्पेसिफिकेशन शीट में 1M या 10M कहते हैं। वास्तविकता कहती है कि 60-70% उपयोग योग्य है, और "उपयोग योग्य" कार्य पर निर्भर करता है।

- **निकासी (फन ढेर में एकल सुई):** सीमा मॉडल पर विज्ञापनित अधिकतम तक लगभग सही।
- **बहु-हॉप / संश्लेषणः** अधिकांश मॉडल पर ~ 128k से अधिक तेजी से गिरावट आती है।
- **फैला हुआ तथ्य पर तर्क देनाः** असफल होने वाला पहला कार्य।

दीर्घ संदर्भ मूल्यांकन इन अक्षों को मापता है। इस पाठ में बेंचमार्क का नाम दिया गया है, प्रत्येक वास्तव में क्या मापता है, और कैसे अपने डोमेन के लिए एक कस्टम सुई परीक्षण का निर्माण करें।

## अवधारणा

![NIAH मूल रेखा, RULER बहु-कार्य, LongBench समग्र](../assets/long-context-eval.svg)

**सुई-इन-ए-हैस्टैक (NIAH, 2023).** एक तथ्य ("जादू शब्द अनानास है") को एक लंबी संदर्भ में नियंत्रित गहराई पर रखें। मॉडल से इसे प्राप्त करने के लिए कहें। गहराई × लंबाई को झाड़ें। मूल लंबे संदर्भ बेंचमार्क। सीमा मॉडल अब इसे संतृप्त करते हैं; यह एक आवश्यक है लेकिन पर्याप्त आधार रेखा नहीं है।

**RULER (एनवीडिया, 2024) ।** 4 श्रेणियों में 13 कार्य प्रकारः पुनर्प्राप्त (एक / बहु-कुंजी / बहु-मूल्य), बहु-हॉप ट्रैकिंग (भयावह ट्रैकिंग), संश्लेषण (सामान्य शब्द आवृत्ति), QA. कॉन्फ़िगरेबल संदर्भ लंबाई (4k से 128k+) । संतोषजनक मॉडल प्रकट करता है NIAH 2024 रिलीज में, केवल आधे 17 मॉडल 32k+ संदर्भ का दावा करते हैं 32k गुणवत्ता बनाए रखा।

**LongBench v2 (2024).** 503 बहुविकल्पीय प्रश्न, 8k-2M शब्द संदर्भ, छह कार्य श्रेणियाँः एकल-डॉक QA, बहु-doc QA, लंबे समय तक संदर्भ में सीखने, लंबे संवाद, कोड रेपो, लंबे समय तक संरचित डेटा। वास्तविक दुनिया में लंबे समय तक संदर्भ व्यवहार के लिए उत्पादन बेंचमार्क।

**MRCR (बहु-राउंड कोरफेरेंस संकल्प) ।** स्केल में बहु-टर्न कोरफेरेंस. 8 सुई, 24 सुई, 100 सुई संस्करण. एक मॉडल ध्यान गिरावट से पहले कितने तथ्यों को प्रकट कर सकते हैं.

**NoLiMa.** "गैर-लक्सिकल सुई।" सुई और क्वेरी में शाब्दिक ओवरलैप नहीं है; पुनर्प्राप्त करने के लिए एक अर्थवादी तर्क की आवश्यकता होती है। NIAH.

**HELMET.** कई दस्तावेजों को जोड़ता है, किसी से भी सवाल पूछता है, चुनिंदा ध्यान का परीक्षण करता है।

**BABILong.** सम्मिलित bAbI तर्कहीन सवेरा ढेरों के अंदर तर्क श्रृंखलाओं. परीक्षण तर्क-इन-एक सवेरा ढेर, न केवल निकासी.

### क्या रिपोर्ट करना है

- **विज्ञापन संदर्भ विंडो।** स्पेसिफिकेशन शीट नंबर।
- **प्रभावी निकासी की लंबाई।** NIAH कुछ सीमा पर पारित करें (उदाहरण के लिए, 90%) ।
- **प्रभावी तर्क लंबाई।** उस सीमा पर मल्टी-हॉप या एग्रीगेशन पास।
- **गिरावट वक्र.** परिप्रेक्ष्य लंबाई के साथ सटीकता, प्रत्येक कार्य प्रकार के अनुसार चित्रित।

आपके विनिर्देश पत्र के लिए दो संख्याएंः पुनः प्राप्ति-प्रभावी और तर्क-प्रभावी। आमतौर पर तर्क-प्रभावी विज्ञापन विंडो का 25-50% है।

```figure
gx-niah-decay
```

## इसे बनाओ

### चरण 1: एक रीति-रिवाज NIAH आपके डोमेन के लिए

देखिये `code/main.py`. . कंकाल:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

पोंछना `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}. हीटमैप का सारण। NIAH अपने लक्ष्य मॉडल के लिए कार्ड.

### चरण 2: बहु-नाल संस्करण

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

"तीन जादू के शब्द क्या हैं?" जैसे प्रश्नों के लिए तीनों को निकालना आवश्यक है। एक ही सुई की सफलता कई सुई की सफलता की भविष्यवाणी नहीं करती है।

### चरण 3: बहु-हॉप चर का पता लगाना (RULER-style)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

उत्तर के लिए तीन कार्यों को जोड़ना आवश्यक है। 128k पर फ्रंटियर मॉडल अक्सर 50-70% सटीकता तक गिर जाते हैं।

### चरण 4: LongBench v2 अपने ढेर पर

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

प्रति श्रेणी सटीकता रिपोर्ट करें. समग्र स्कोर कार्य स्तर में बड़े अंतर छिपाता है।

## फंदे

- **NIAH-only मूल्यांकन।** गुजरना NIAH 1M टोकन पर मल्टी-हॉप के बारे में कुछ नहीं कहता है। हमेशा चलाने RULER या एक कस्टम मल्टी-हॉप परीक्षण।
- **एक समान गहराई का नमूनाकरण।** कई कार्यान्वयन केवल परीक्षण depth=0.5. परीक्षण depth=0, 0.25, 0.5, 0.75, 1.0  "मध्य में खो" प्रभाव वास्तविक है।
- **भरने के साथ लक्ज़िकल ओवरलैप।** यदि सुई ने फिलर के साथ कीवर्ड साझा किए हैं, तो निकालना तुच्छ हो जाता है। NoLiMa-style गैर-परचमते सुइयों।
- **विलंबता को अनदेखा करना.** 1M टोकन संकेतों को पूर्व भरने में 30-120 सेकंड लगते हैं। सटीकता के साथ समय से पहले टोकन को मापें।
- **विक्रेता द्वारा स्वयं रिपोर्ट किए गए संख्या।** OpenAI, गूगल, Anthropic आप अपने उपयोग के मामले पर हमेशा स्वतंत्र रूप से फिर से चलाने.

## इसका प्रयोग करें

2026 स्टैकः

| स्थिति | बेंचमार्क |
|-----------|-----------|
| त्वरित मानसिक जांच | कस्टम NIAH 3 गहराई × 3 लंबाई पर |
| उत्पादन के लिए मॉडल चयन | RULER (13 कार्य) आपके लक्ष्य लंबाई पर |
| वास्तविक दुनिया QA गुणवत्ता | LongBench v2 एकल-doc-QA उपसमूह |
| बहु-हॉप तर्क | BABILong या कस्टम चर-ट्रैकिंग |
| वार्ता / संवाद | MRCR अपने लक्ष्य लंबाई पर 8-नाल |
| मॉडल उन्नयन प्रतिगमन | घर में फिक्स्ड NIAH + RULER हर नए मॉडल पर चलाने के लिए |

उत्पादन के लिए अंगूठे का नियमः जब तक आप एक संदर्भ विंडो पर भरोसा नहीं करते NIAH + 1 reasoning task at your intended length.

## इसे भेजें

के रूप में सहेजें `outputs/skill-long-context-eval.md`:

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## व्यायाम

1. **- आराम से।** एक निर्माण NIAH 3 गहराई (0.25, 0.5, 0.75) × 3 लंबाई (1k, 4k, 16k) के साथ। किसी भी मॉडल पर चलाएं। 3 × 3 हीटमैप के रूप में प्लॉट पास दर।
2. **मध्यम।** एक 3 सुई संस्करण जोड़ें. प्रत्येक लंबाई पर सभी 3 का माप लें. समान लंबाई पर एकल सुई पास दर की तुलना करें.
3. **कठिन.** चर-ट्रेसिंग कार्य का निर्माण करें (X1 → X2 → X33 सीमा मॉडल में सटीकता मापें प्रति मॉडल प्रभावी तर्क लंबाई रिपोर्ट

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| NIAH | पिन स्टैक में सुई | एक तथ्य भरने में लगाएं, मॉडल से पूछें कि वह इसे प्राप्त करें। |
| RULER | NIAH स्टेरॉयड पर | 13 कार्य प्रकारों के पार निकालने / बहु-हॉप / संश्लेषण / QA. |
| प्रभावी संदर्भ | वास्तविक क्षमता | लंबाई जिसमें सटीकता अभी भी सीमा से ऊपर है। |
| बीच में खो गया | गहराई पूर्वाग्रह | लंबे इनपुट के बीच में सामग्री को मॉडल कम ध्यान देते हैं। |
| बहु-इगल | एक ही समय में कई तथ्य | कई पौधे; ध्यान जुगलिंग का परीक्षण, केवल पुनर्प्राप्ति नहीं। |
| MRCR | बहु-राउंड कोरफ | 8, 24, या 100 सुई कोरफेरेंस; ध्यान संतृप्ति का खुलासा करता है। |
| NoLiMa | गैर-लक्सिकल सुई | सुई और प्रश्न का कोई शाब्दिक संकेत नहीं है; तर्क की आवश्यकता है। |

## आगे पढ़ना

- [काम्राट (2023) हेस्टैक विश्लेषण में सुई](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) मूल NIAH रेपो।
- [Hsieh et al. (2024). RULER: आपके लंबे संदर्भ का वास्तविक संदर्भ आकार क्या है LMs?](https://arxiv.org/abs/2404.06654) बहु-कार्य बेंचमार्क।
- [बाई एट अल. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) वास्तविक दुनिया में दीर्घ संदर्भ मूल्यांकन।
- [Modarressi et al. (2024). NoLiMa: गैर-लक्ज़िकल सुइयों](https://arxiv.org/abs/2404.06666) कठिन सुइयों।
- [कुराटोव और अन्य (2024). BABILong](https://arxiv.org/abs/2406.10149) तर्क-नार-नार में।
- [लियू और अन्य (2024) । मिडिल में खोयाः कैसे भाषा मॉडल लंबे संदर्भ का उपयोग करते हैं](https://arxiv.org/abs/2307.03172) गहराई पूर्वाग्रह कागज।
