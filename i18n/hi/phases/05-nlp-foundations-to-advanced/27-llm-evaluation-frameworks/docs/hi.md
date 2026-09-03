# LLM मूल्यांकन RAGAS, DeepEval, G-Eval

> सटीक-मिलान और F1 मानव समीक्षा पैमाने नहीं है। LLM-as-judge संख्या पर भरोसा करने के लिए पर्याप्त माप के साथ उत्पादन उत्तर है।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 minutes

## समस्या

आपका RAG प्रणाली का उत्तर है, "29 जून, 2007।"
सोने का संदर्भ हैः "29 जून, 2007"
सटीक मैच 0 स्कोर करता है। F1 एक इंसान 100% स्कोर करता है।

अब 10,000 परीक्षण मामलों से गुणा करें। रिट्रीवर, चश्मा, प्रॉम्प्ट या मॉडल में हर बदलाव से फिर से गुणा करें। आपको एक मूल्यांकनकर्ता की आवश्यकता है जो अर्थ को समझता है, स्केल पर सस्ते में चलता है, प्रतिगमन के बारे में झूठ नहीं बोलता है, और सही विफलता मोड को उजागर करता है।

2026 में तीन ढांचे हैं जो इस समस्या का मालिक हैं।

- **RAGAS.** पुनर्प्राप्ति-बढ़ती पीढ़ी ASsessment. चार RAG मानकों (निष्ठा, उत्तर-संदिग्धता, संदर्भ-सटीकता, संदर्भ-याद) के साथ NLI + LLM-judge अनुसंधान समर्थित, हल्के वजन।
- **DeepEval.** के लिए Pytest LLMs. जी-ईवल, कार्य पूरा, भ्रम, पूर्वाग्रह मेट्रिक्स। CI/CD-native.
- **G-Eval.** एक विधि (और एक DeepEval मेट्रिक): LLM-as-judge सोच श्रृंखला, कस्टम मानदंडों के साथ, 0-1 स्कोर.

तीनों ही पर निहित LLM-as-judge. यह पाठ विधि और उसके आसपास के विश्वास परत के लिए अंतर्ज्ञान का निर्माण करता है।

## अवधारणा

![चार मूल्यांकन आयाम, LLM-as-judge वास्तुकला](../assets/llm-evaluation.svg)

**LLM-as-judge.** एक स्थैतिक मीट्रिक को एक से बदलें LLM जो एक rubric के अनुसार आउटपुट स्कोर करता है। `(query, context, answer)`, एक न्यायाधीश को बुला LLM: "निष्ठा पर 0-1 स्कोर।" स्कोर लौटाएं।

यह काम क्यों करता हैः LLMs लागत के एक छोटे से अंश पर अनुमानित मानव न्याय। GPT-4o-mini ~ $0.003 प्रति स्कोर मामले 1000 नमूना प्रतिगमन मूल्यांकन $ 5 से कम के लिए चलाता है सक्षम बनाता है।

यह चुपचाप क्यों विफल रहता हैः

1. **न्याय पूर्वाग्रह।** न्यायाधीशों को लंबे समय तक उत्तर पसंद हैं, अपने स्वयं के मॉडल परिवार से उत्तर, उत्तर जो शीघ्र शैली से मेल खाते हैं।
2. **JSON विश्लेषण विफलताओं।** बुरा JSON → NaN स्कोर → चुपचाप समग्र से बाहर रखा गया। RAGAS प्रयास / सिवाय + स्पष्ट विफलता मोड के साथ गेट.
3. **मॉडल संस्करणों पर बहना।** न्यायाधीश को अपग्रेड करने से हर मीट्रिक बदल जाता है।

**इन RAG चार।**

| मेट्रिक | प्रश्न | बैकेंड |
|--------|----------|---------|
| वफादार | क्या उत्तर में प्रत्येक कथन प्राप्त संदर्भ से आता है? | NLI-based समावेशी |
| उत्तर प्रासंगिकता | क्या इसका जवाब इस सवाल का जवाब देता है? | उत्तर से परिकल्पनात्मक प्रश्न उत्पन्न करें; वास्तविक प्रश्न से तुलना करें |
| संदर्भ सटीकता | प्राप्त टुकड़ों में से कौन सा अंश प्रासंगिक था? | LLM-judge |
| संदर्भ याद | क्या निकासी ने आवश्यक सब कुछ वापस कर दिया? | LLM-judge स्वर्ण उत्तर के विपरीत |

**G-Eval.** एक कस्टम मानदंड परिभाषित करेंः "क्या उत्तर सही स्रोत का हवाला देता है?" फ्रेमवर्क स्वचालित रूप से विचार श्रृंखला मूल्यांकन चरणों में विस्तारित होता है, फिर 0-1 स्कोर करता है। डोमेन-विशिष्ट गुणवत्ता आयामों के लिए अच्छा है RAGAS कवर नहीं करता है।

**मापने.** जब तक आप मानव लेबल के साथ एक सहसंबंध नहीं है कच्चे न्यायाधीश स्कोर पर भरोसा कभी नहीं. 100 हाथ लेबल उदाहरण चलाएं. प्लॉट न्यायाधीश बनाम मानव. गणना स्पीयरमैन rho. अगर rho < 0.7, अपने न्यायाधीश rubric काम करने की जरूरत है.

```figure
n5-judge-gauge
```

## इसे बनाओ

### चरण 1: निष्ठा NLI (RAGAS-style)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

उत्तर को परमाणु दावों में विघटित करें। NLI-check प्राप्त संदर्भ के विरुद्ध प्रत्येक दावा। Faithfulness = fraction समर्थित।

### चरण 2: उत्तर प्रासंगिकता

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

यदि उत्तर में पूछे गए प्रश्न से भिन्न प्रश्न हैं, तो प्रासंगिकता कम होती है।

### चरण 3: G-Eval कस्टम मीट्रिक

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

मूल्यांकन चरणों को rubric हैं. स्पष्ट चरणों को स्पष्ट "स्कोर 0-1" संकेतों की तुलना में अधिक स्थिर हैं।

### चरण 4: CI द्वार

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

एक पिटेस्ट फ़ाइल के रूप में जहाज. हर पर चल PR. ब्लॉक विघटन पर विलय करता है।

### चरण 5: खरोंच से खिलौना मूल्यांकन

देखिये `code/main.py`. केवल निष्ठा (संदर्भ के साथ उत्तर दावों के ओवरलैप) और प्रासंगिकता (संदर्भ टोकन के साथ उत्तर टोकन के ओवरलैप) के अनुमान। उत्पादन नहीं। आकार दिखाता है।

## फंदे

- **कोई माप नहीं।** मानव लेबल के साथ 0.3 संबद्धता के साथ एक न्यायाधीश शोर है. शिपिंग से पहले एक माप चलाने की आवश्यकता है.
- **आत्म-मूल्यांकन।** उसी का उपयोग LLM एक अलग मॉडल परिवार का उपयोग करें, एक अलग मॉडल परिवार का उपयोग करें।
- **जोड़ी निर्णय में स्थिति पूर्वाग्रह।** न्यायाधीश पहले विकल्प को पसंद करते हैं। हमेशा क्रम क्रमबद्ध करें और दोनों को चलाएं।
- **कच्चे पदार्थ विफलताओं को छिपाते हैं।** औसत स्कोर 0.85 अक्सर 5% आपदा विफलताओं को छिपाता है। हमेशा नीचे क्वांटिल की जांच करें।
- **सोने के डेटासेट सड़ने.** अनवर्स किए गए मूल्यांकन सेट जो समय के साथ बहते हैं, longitudinal comparison को तोड़ते हैं। प्रत्येक परिवर्तन के साथ डेटासेट को टैग करें।
- **LLM लागत।** पैमाने पर, न्यायाधीशों को कॉल लागत पर हावी है. सबसे सस्ता मॉडल का उपयोग करें जो मापने की सीमा को पूरा करता है. GPT-4o-mini, Claude हैकू, Mistral-छोटे.

## इसका प्रयोग करें

2026 स्टैकः

| उपयोग के मामले | ढांचा |
|---------|-----------|
| RAG गुणवत्ता निगरानी | RAGAS (4 मेट्रिक) |
| CI/CD प्रतिगमन द्वार | DeepEval + pytest |
| कस्टम डोमेन मानदंड | G-Eval अंदर DeepEval |
| ऑनलाइन लाइव ट्रैफ़िक निगरानी | RAGAS संदर्भ मुक्त मोड के साथ |
| मानव-सक्रिय स्पॉट जांच | LangSmith या नोट के साथ फीनिक्स UI |
| रेड-टीमिंग / सुरक्षा मूल्यांकन | Promptfoo + DeepEval |

विशिष्ट स्टैकः RAGAS निगरानी के लिए, DeepEval के लिए CIतीनों को चलाएं; वे उपयोगी रूप से असहमत हैं।

## इसे भेजें

के रूप में सहेजें `outputs/skill-eval-architect.md`:

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## व्यायाम

1. **- आराम से।** उपयोग RAGAS 10 पर RAG ज्ञात भ्रमों के साथ उदाहरणों. सत्यापित करने के लिए वफादारी मीट्रिक प्रत्येक पकड़ता है.
2. **मध्यम।** हाथ लेबल 50 QA सही के लिए 0-1 के जवाब. G-Eval के साथ स्कोर. न्यायाधीश और मानव के बीच Spearman rho मापने.
3. **कठिन.** एक पिटेस्ट बनाएं CI के साथ गेट DeepEval. जानबूझकर रिट्रीवर को वापस करें, गेट विफलता की पुष्टि करें, सबसे कम 10% पर सीमा की जांच के माध्यम से नीचे क्वांटिल अलर्ट जोड़ें।

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| LLM-as-judge | एक के साथ स्कोर LLM | एक रूब्रिक दिए गए आउटपुट 0-1 को स्कोर करने के लिए एक न्यायाधीश मॉडल को प्रेरित करें। |
| RAGAS | इन RAG मेट्रिक लाइब्रेरी | 4 संदर्भ मुक्त के साथ ओपन सोर्स मूल्यांकन ढांचा RAG मेट्रिक्स। |
| वफादार | क्या इसका जवाब सही है? | उत्तर के संदर्भ से उत्पन्न दावे का अंश। |
| संदर्भ सटीकता | क्या निकाले गए टुकड़े प्रासंगिक थे? | शीर्ष-के टुकड़ों का एक अंश जो वास्तव में मायने रखता था। |
| संदर्भ याद | क्या निकासी सब कुछ मिला? | प्राप्त टुकड़ों द्वारा समर्थित स्वर्ण-जवाब के दावों का अंश। |
| G-Eval | कस्टम LLM न्यायाधीश | रूब्रिक + विचार श्रृंखला मूल्यांकन चरण + 0-1 स्कोर। |
| माप | भरोसा करें लेकिन सत्यापित करें | न्यायाधीश स्कोर और मानव स्कोर के बीच स्पीयरमैन संबंध। |

## आगे पढ़ना

- [एट अल. (2023). RAGAS: पुनर्प्राप्ति का स्वचालित मूल्यांकन बढ़ी हुई पीढ़ी](https://arxiv.org/abs/2309.15217)  RAGAS कागज।
- [लीउ एट अल. (2023) NLG मूल्यांकन GPT-4 मानव के बेहतर संरेखण के साथ](https://arxiv.org/abs/2303.16634) G-Eval पेपर।
- [DeepEval डॉक्स](https://deepeval.com/docs/metrics-introduction) खुले उत्पादन स्टैक।
- [झेंग एट एल्स (2023) । न्याय LLM-as-a-Judge के साथ MT-Bench और चैटबॉट एरेना](https://arxiv.org/abs/2306.05685) पूर्वाग्रह, माप, सीमाएं।
- [MLflow GenAI स्कोरर](https://mlflow.org/blog/third-party-scorers) एकीकृत करने वाली एकजुटता RAGAS, DeepEval, फीनिक्स.
