# उपशब्द टोकनाइज़ेशन BPE, WordPiece, यूनिग्राम, SentencePiece

> शब्द टोकन बनाने वाले अदृश्य शब्दों पर थूक जाते हैं। वर्ण टोकन करने वाले अनुक्रम की लंबाई को बढ़ा देते हैं। उपशब्द टोकन करने वाले अंतर को विभाजित करते हैं। हर आधुनिक LLM जहाजों पर एक.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## समस्या

आपके शब्दावली में 50,000 शब्द हैं। एक उपयोगकर्ता टाइप करता है "अनटोकनेज"। आपका टोकनेज़र वापस आता है `[UNK]`. मॉडल में अब शब्द के बारे में कोई संकेत नहीं है. और इससे भी बदतर: आपके कॉर्पस में 90वें प्रतिशत दस्तावेज़ में 40 दुर्लभ शब्द हैं, जिसका अर्थ है कि प्रति दस्तावेज़ 40 बिट्स की जानकारी गिर गई है.

उपशब्द टोकनकरण इस समस्या का समाधान करता है. सामान्य शब्द एकल टोकन बने रहते हैं. दुर्लभ शब्द सार्थक टुकड़ों में विघटित होते हैंः `untokenizable` → `un`, `token`, `izable`प्रशिक्षण डेटा सब कुछ शामिल है क्योंकि किसी भी स्ट्रिंग अंततः बाइट्स का एक अनुक्रम है।

हर सीमा LLM 2026 में तीन एल्गोरिदम में से एक पर जहाज (BPE, यूनिग्राम, WordPiece), तीन पुस्तकालयों में से एक में लपेटा हुआ (टिकटोकन, SentencePiece, HF आप एक भाषा मॉडल को चुनने के बिना नहीं भेज सकते।

## अवधारणा

![BPE बनाम यूनिग्राम बनाम WordPiece, चरित्र-दर-चरित्र](../assets/subword-tokenization.svg)

**BPE (बाइट-पियर एन्कोडिंग)** वर्ण स्तर की शब्दावली से शुरू करें. प्रत्येक आसन्न जोड़ी की गणना करें. सबसे अधिक बार होने वाली जोड़ी को नए टोकन में मिलाएं. जब तक आप लक्ष्य शब्दावली आकार को नहीं मिलते तब तक दोहराएं। GPT-2/3/4, Llama, जेम्मा, Qwen2, Mistral.

**बाइट स्तर BPE.** वही एल्गोरिथ्म लेकिन यूनिकोड वर्णों के बजाय कच्चे बाइट्स (256 बेस टोकन) पर। शून्य गारंटी `[UNK]` टोकन  किसी भी बाइट अनुक्रम कोड। GPT-2 uses 50,257 tokens (256 bytes + 50,000 merges + 1 special).

**यूनिग्राम.** एक विशाल शब्दावली से शुरू करें. प्रत्येक टोकन को एक एकलसूची की संभावना असाइन करें. प्रतिवारात्मक रूप से टोकन काटें जिनकी हटाने से कॉर्पस लॉग-संभावना कम से कम बढ़ जाती है। निष्कर्ष पर संभावनाः नमूना टोकनकरण (उपशब्द नियमितकरण के माध्यम से डेटा बढ़ाने के लिए उपयोगी) । T5, mBART, ALBERT, XLNet, जेम्मा.

**WordPiece.** मिश्रण जोड़े जो कच्चे आवृत्ति के बजाय प्रशिक्षण कॉर्पस की संभावना को अधिकतम करते हैं। BERT, DistilBERT, ELECTRA.

**SentencePiece बनाम टिक टॉक.** SentencePiece क्या पुस्तकालय है कि *ट्रेनें* शब्दावली (BPE या Unigram) सीधे कच्चे यूनिकोड पाठ पर, कोडिंग व्हाइटस्पेस के रूप में `▁`. टिक टॉक है OpenAIयह तेजी से है *एन्कोडर* पूर्व निर्मित शब्दावली के खिलाफ; यह प्रशिक्षण नहीं देता है।

अंगूठे का नियमः

- **एक नई शब्दावली का प्रशिक्षणः** SentencePiece (बहुभाषी, कोई पूर्व-टोकनाइज़ेशन नहीं) या HF टोकन बनाने वाले।
- **तीव्र निष्कर्ष GPT शब्दशः** tiktoken (cl100k_base, o200k_base) ।
- **दोनोंः** HF टोकन बनाने वाले  एक पुस्तकालय, प्रशिक्षण + सेवा।

```figure
bpe-merge
```

## इसे बनाओ

### चरण 1: BPE खरोंच से

देखिये `code/main.py`. . लूपः

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

एल्गोरिथ्म तीन तथ्यों को एन्कोड करता है। `</w>` शब्द अंत के निशान इसलिए "कम" (सफल) और "कम" (पूर्व) अलग रहते हैं। आवृत्ति भार उच्च आवृत्ति जोड़े जल्दी जीतता है। विलय सूची क्रमबद्ध है  निष्कर्ष प्रशिक्षण क्रम में विलय लागू होता है।

### चरण 2: सीखे गए विलय के साथ कोड करें

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

उत्पादन कार्यान्वयन (उत्पादन कार्यान्वयन) HF टोकन) प्राथमिकता कतारों के साथ विलय-रैंक खोज का उपयोग करते हैं और लगभग रैखिक समय में चलते हैं।

### चरण 3: SentencePiece व्यवहार में

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

नोटः कोई पूर्व-टोकेनाइज़ेशन की आवश्यकता नहीं, स्थान को कोडित किया गया है `▁`, `character_coverage` नियंत्रण करता है कि आक्रामक रूप से दुर्लभ वर्णों को संरक्षित किया जाता है बनाम मैप किया जाता है `<unk>`.

### चरण 4: टिकटोक के लिए OpenAI-compatible शब्द

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

केवल एन्कोडिंग।Rust बैकेंड) के साथ सटीक मेल खाता है GPT-4/5 बाइट-कंटेंट, लागत अनुमान, संदर्भ-विंडो बजटिंग के लिए टोकनकरण।

## 2026 में भी फंसे हुए जाल

- **टोकनलाइज़र बहाव.** वाक्यांश ए पर प्रशिक्षण, वाक्यांश बी के खिलाफ तैनात करना। टोकन IDs भिन्नता; मॉडल आउटपुट कचरा। `tokenizer.json` हश में CI.
- **सफेद अंतरिक्ष अस्पष्टता।** BPE "hello" बनाम "hello" अलग टोकन उत्पन्न करते हैं। हमेशा निर्दिष्ट करें `add_special_tokens` और `add_prefix_space` स्पष्ट रूप से।
- **बहुभाषी प्रशिक्षण।** अंग्रेजी-भारी कॉर्पोरेस शब्दावली का उत्पादन करते हैं जो गैर-लैटिन लिपि को 5-10 गुना अधिक टोकन में विभाजित करते हैं। GPT-3.5. o200k_base आंशिक रूप से इस तय किया.
- **इमोजी विभाजित.** एक एकल इमोजी 5 टोकन ले सकता है। संदर्भ बजट करते समय चेकपॉइंट इमोजी संभाल।

## इसका प्रयोग करें

2026 स्टैकः

| स्थिति | चुनें |
|-----------|------|
| एक भाषाई मॉडल को खरोंच से प्रशिक्षित करना | HF टोकन (BPE) |
| बहुभाषी मॉडल का प्रशिक्षण | SentencePiece (एकसूत्र, `character_coverage=0.9995`) |
| सेवा OpenAI-compatible API | टिक टॉक (`o200k_base` के लिए GPT-4+) |
| डोमेन-विशिष्ट शब्दावली (कोड, गणित, प्रोटीन) | ट्रेन कस्टम BPE डोमेन कॉर्पस पर, मूल शब्दावली के साथ विलय करें |
| किनारे का अनुमान, छोटा मॉडल | यूनोग्राम (छोटे शब्दावली बेहतर काम करती है) |

शब्दकोश का आकार एक स्केलिंग निर्णय है, एक स्थिर नहीं। for <1B params, 50-100k 1-10B के लिए, बहुभाषी/सीमा के लिए 200k+।

## इसे भेजें

के रूप में सहेजें `outputs/skill-bpe-vs-wordpiece.md`:

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## व्यायाम

1. **- आराम से।** 500-मिलन को प्रशिक्षित करें BPE पर `code/main.py`तीन लंबे समय तक चलने वाले शब्दों को एन्कोड करें। कितने ने एक टोकन को उत्पन्न किया vs >1 टोकन?
2. **मध्यम।** 100 अंग्रेजी विकिपीडिया वाक्यों पर टोकन गिनती की तुलना करें `cl100k_base`, `o200k_base`, और एक SentencePiece BPE आप 32k के साथ अभ्यास करते हैं। प्रत्येक के संपीड़न अनुपात की रिपोर्ट करें।
3. **कठिन.** के साथ एक ही corpus को प्रशिक्षित BPE, यूनिग्राम, और WordPiece. एक छोटे से भावना वर्गीकरण पर प्रत्येक का उपयोग करते समय डाउनस्ट्रीम सटीकता मापें। क्या विकल्प ने सुई को 1 बिंदु से अधिक स्थानांतरित किया है F1?

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| BPE | बाइट-पियर एन्कोडिंग | लक्ष्य शब्दावली आकार तक पहुंचने तक सबसे अधिक बार वर्ण जोड़े के लोभी मिश्रण। |
| बाइट स्तर BPE | कभी भी अज्ञात टोकन नहीं | BPE 256 बाइट से अधिक कच्चे; GPT-2 / Llama इस का उपयोग करें. |
| एक्र्क | संभावनावादी टोकनराइज़र | लॉग-संभावना का उपयोग करके एक बड़े उम्मीदवार सेट से खजूर; द्वारा उपयोग किया जाता है T5, जेम्मा. |
| SentencePiece | सफेद अंतरिक्ष एक | पुस्तकालय जो ट्रेन करता है BPE/कच्चा पाठ पर यूनोग्राम; स्थान को कोडित किया गया `▁`. |
| टिक टॉक | सबसे तेज़ | OpenAIहै Rust- समर्थित BPE पूर्व निर्मित शब्दावली के लिए एन्कोडर। कोई प्रशिक्षण नहीं। |
| विलय सूची | जादू अंक | क्रमबद्ध सूची `(a, b) → ab` विलय; अनुक्रम में निष्कर्ष लागू होता है। |
| चरित्र कवरेज | कितना दुर्लभ बहुत दुर्लभ है? | प्रशिक्षण कॉर्पस में वर्णों का अंश टोकनराइज़र को कवर करना चाहिए; ~ 0.9995 विशिष्ट। |

## आगे पढ़ना

- [Sennrich, Haddow, Birch (2015) । उपशब्द इकाइयों के साथ दुर्लभ शब्दों का तंत्रिका मशीन अनुवाद](https://arxiv.org/abs/1508.07909)  BPE कागज।
- [कुडो (2018). यूनोग्राम भाषा मॉडल के साथ उपशब्द विनियमन](https://arxiv.org/abs/1804.10959) यूनिग्राम पेपर।
- [कुडो, रिचर्डसन (2018). SentencePiece: एक सरल और भाषा स्वतंत्र उपशब्द टोकन](https://arxiv.org/abs/1808.06226) पुस्तकालय।
- [गले लगाते हुए चेहरा  टोकन बनाने वालों का सारांश](https://huggingface.co/docs/transformers/tokenizer_summary) संक्षिप्त संदर्भ।
- [OpenAI टिक टॉक रेपो](https://github.com/openai/tiktoken) रसोई पुस्तक + कोडिंग सूची।
