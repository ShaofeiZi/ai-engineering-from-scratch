# ऑडियो मूल्यांकन WER, MOS, UTMOS, MMAU, FAD, और ओपन लीडरबोर्ड

> आप जो नहीं माप सकते उसे भेज सकते हैं। इस पाठ में प्रत्येक ऑडियो कार्य के लिए 2026 मीट्रिक का नाम दिया गया हैः ASR (WER, CER, RTFx), TTS (MOS, UTMOS, SECS, WER-on-ASR-round-trip), ऑडियो-भाषा (MMAU, LongAudioBench), संगीत (FAD, CLAP), और स्पीकर (EER) और रैंकिंग बोर्ड जहां आप तुलना करते हैं।

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time:** ~60 minutes

## समस्या

प्रत्येक ऑडियो कार्य में कई मीट्रिक होते हैं, प्रत्येक एक अलग अक्ष को मापता है। गलत मीट्रिक का उपयोग करके आप एक मॉडल कैसे भेजते हैं जो आपके डैशबोर्ड पर बहुत अच्छा दिखता है और उत्पादन में भयानक है। 2026 की कैनोनिकल सूचीः

| कार्य | प्राथमिक | माध्यमिक |
|------|---------|-----------|
| ASR | WER | CER · RTFx · पहले टोकन की लटेंसी |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| आवाज क्लोनिंग | SECS (ECAPA कॉसिन) | MOS · CER |
| स्पीकर सत्यापन | EER | minDCF · FAR / FRR परिचालन बिंदु पर |
| डायरीकरण | DER | JER · स्पीकर भ्रम |
| ऑडियो वर्गीकरण | शीर्ष-1 · mAP | मैक्रो F1 · प्रति वर्ग रिकॉल |
| संगीत पीढ़ी | FAD | CLAP · श्रवण पैनल MOS |
| ऑडियो भाषा मॉडल | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| स्ट्रीमिंग S2S | विलंबता P50/P95 | WER · MOS |

## अवधारणा

![ऑडियो मूल्यांकन मैट्रिक्स  मेट्रिक्स बनाम कार्य बनाम 2026 रैंकिंग बोर्ड](../assets/eval-landscape.svg)

### ASR माप

**WER (शब्द त्रुटि दर)** `(S + D + I) / N`. कम अक्षर, अंकन, अंकन से पहले संख्याओं को सामान्य करें. `jiwer` या OpenAIहै `whisper_normalizer`. &lt;5% = मानव-समानता भाषण पढ़ना.

**CER (वर्ण त्रुटि दर)** एक ही सूत्र, वर्ण-स्तर। स्वर भाषाओं (मान्डारीन, कैंटोन) के लिए उपयोग किया जाता है जहां शब्द विभाजन अस्पष्ट है।

**RTFx (परवा वास्तविक समय कारक) ।** प्रति वॉल-घड़ी सेकंड ऑडियो सेकंड संसाधित. उच्च बेहतर है.TDT 3380x पर हिट करता है.v3 यह ~30× है।

**पहले टोकन विलंबता.** ऑडियो इनपुट से पहले ट्रांसक्रिप्ट टोकन तक दीवार घड़ी स्ट्रीमिंग के लिए महत्वपूर्ण है।

### TTS माप

**MOS (मतलब राय स्कोर) ।** 1-5 मानव रेटिंग. स्वर्ण मानक लेकिन धीमा. प्रति नमूना 20+ श्रोताओं, प्रति मॉडल 100+ नमूने एकत्र.

**UTMOS (2022-2026).** सीखा MOS भविष्यवाणीकर्ता. मानव के साथ ~0.9 के साथ संबद्ध MOS मानक बेंचमार्क पर। F5-TTS: UTMOS 3.95; मूल सत्य: 4.08.

**SECS (स्पीकर एन्कोडर कॉसिन समानता)** आवाज क्लोनिंग के लिए। ECAPA संदर्भ और क्लोन आउटपुट के बीच कोसिन को एम्बेड करना। &gt; 0.75 = पहचान योग्य क्लोन।

**WER-on-ASR-round-trip.** विस्पर के पास चलाओ TTS आउटपुट, गणना WER इनपुट पाठ के खिलाफ। समझदारी regressions पकड़ता है. 2026 SOTA: &lt; 2% CER.

**TTFA (समय से पहले ऑडियो)** वॉल क्लॉक लेटेन्स. कोकोरो-82एम: ~ 100 ms; F5-TTS: ~ 1 से.

### आवाज क्लोनिंग-विशिष्ट

**SECS + MOS + CER** एक तिगुना के रूप में। जो उच्च स्कोर करता है SECS लेकिन कम MOS इसका अर्थ है- सही-परंतु-अप्राकृतिक; विपरीत का अर्थ है- प्राकृतिक आवाज लेकिन गलत वक्ता।

### स्पीकर सत्यापन

**EER (समान त्रुटि दर)** ऐसी सीमा जहां गलत स्वीकार दर गलत अस्वीकार दर के बराबर है। ECAPA पर VoxCeleb1-O: 0.87%.

**minDCF (मिनेट डिटेक्शन लागत) ।** चयनित परिचालन बिंदु पर वजन की गई लागत (अक्सर FAR=0.01) उत्पादन से अधिक प्रासंगिक EER.

### डायरीकरण

**DER (ज्वलन त्रुटि दर)** `(FA + Miss + Confusion) / total_speaker_time`. चूक बोली + झूठी अलार्म बोली + स्पीकर-गंभीरता, प्रत्येक अंश के रूप में। AMI बैठकेंः DER ~10-20% यथार्थवादी है। pyannote 3.1 + सटीकता-2 वाणिज्यिकः &lt;10% DER अच्छी तरह से रिकॉर्ड किए गए ऑडियो पर।

**JER (जैकार्ड त्रुटि दर)** वैकल्पिक DER, मजबूत से लघु खंड पूर्वाग्रह.

### ऑडियो वर्गीकरण

बहु-लेबलः **mAP (मध्यम औसत सटीकता)** सभी वर्गों पर। AudioSet: 0.548 mAP के लिए BEATs-iter3.

बहु-वर्ग विशेषः **शीर्ष-1, शीर्ष-5 सटीकता**. भाषण आदेश v2: 99.0% शीर्ष-1 (ऑडियो-MAE).

असंतुलित: **मैक्रो F1** + **प्रति वर्ग याद**. प्रति वर्ग रिपोर्ट  समग्र सटीकता छिपाता है कि कौन से वर्ग विफल होते हैं।

### संगीत पीढ़ी

**FAD (फ्रेचेट ऑडियो दूरी)** दूरी के बीच VGGish-embedding वास्तविक बनाम उत्पन्न ऑडियो का वितरण। MusicGen-small पर MusicCaps: 4.5. MusicLM4.0. नीचे बेहतर।

**CLAP स्कोर.** पाठ-ऑडियो संरेखण स्कोर का उपयोग करके CLAP एम्बेडेड. &gt; 0.3 = उचित संरेखण.

**श्रवण पैनल MOS.** उपभोक्ता स्तर के संगीत के लिए अभी भी अंतिम शब्द है। v5 ELO 1293 on TTS एरेना (मानव की जोड़ी पसंद से)

### ऑडियो भाषा बेंचमार्क

**MMAU (महामात्रा मल्टी ऑडियो समझ)** 10k ऑडियो-QA जोड़े।

**MMAU-Pro.** 1800 हार्ड आइटम, चार श्रेणियाँः भाषण / ध्वनि / संगीत / बहु-ऑडियो। Gemini 2.5 प्रो कुल मिलाकर ~ 60%; सभी मॉडलों में मल्टी ऑडियो ~ 22%।

**LongAudioBench.** अर्थिक प्रश्नों के साथ मल्टी मिनट क्लिप। Gemini 2.5 Pro.

**AudioCaps -क्लोथो.** संदर्भ मानकों का उपशीर्षक। SPICE, CIDEr, FENSE मेट्रिक्स।

### भाषण-भाषण स्ट्रीमिंग

**विलंबता P50 / P95 / P99.** उपयोगकर्ता के अंत-उपयोगकर्ता भाषण से पहली श्रव्य प्रतिक्रिया तक दीवार घड़ी। GPT-4o वास्तविक समय: 300 ms.

**WER / MOS** आउटपुट पर।

**प्रवेश करने की प्रतिक्रियाशीलता।** उपयोगकर्ता के अंतराल से सहायक मूक तक समय। लक्ष्य &lt; 150 ms.

### 2026 के शीर्ष स्थान

| रैंडरबोर्ड | ट्रैक | URL |
|------------|--------|-----|
| खुला ASR रैंडरबोर्ड (HF) | English + multilingual + long-form | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS आर्ना (HF) | अंग्रेजी TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, ELO जोड़ी हुई वोटों से | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM तर्क | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | स्पीकर मान्यता | `voxsrc.github.io` |
| MMAU संगीत उपसमूह | संगीत LALM | (अंतर्गत MMAU) |
| HEAR बेंचमार्क | स्व-निरीक्षण ऑडियो | `hearbenchmark.com` |

```figure
sp-wer-align
```

## इसे बनाओ

### चरण 1: WER सामान्यीकरण के साथ

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### चरण 2: TTS यात्रा WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### चरण 3: SECS आवाज क्लोनिंग के लिए

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### चरण 4: FAD संगीत पीढ़ी के लिए

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### चरण 5: EER स्पीकर सत्यापन के लिए (सीमा 6) के समान कोड

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## इसका प्रयोग करें

प्रत्येक तैनाती को एक निश्चित मूल्यांकन हर्नस के साथ जोड़ा जो प्रत्येक मॉडल अपडेट पर चलता है। तीन मुख्य नियमः

1. **स्कोर करने से पहले सामान्य हो जाओ।** लघु अक्षर, अंकन पट्टी, संख्या विस्तार। सामान्यीकरण नियम रिपोर्ट।
2. **औसत नहीं, वितरण की रिपोर्ट करें।** P50/P95/P99 विलंबता के लिए प्रति वर्ग वर्गीकरण के लिए याद MMAU.
3. **एक सार्वजनिक मानक मानक चलाएं।** यहां तक कि अगर आपके उत्पादन डेटा अलग हैं, तो ओपन पर रिपोर्टिंग ASR / TTS एरेना / MMAU समीक्षाओं को सेब से सेब की तुलना करने दें।

## फंदे

- **UTMOS निष्कर्षण।** प्रशिक्षण VCTK-style स्वच्छ भाषण; शोर / क्लोन / भावनात्मक ऑडियो खराब स्कोर करता है।
- **MOS पैनल पूर्वाग्रह।** 20 अमेज़ॅन मैकेनिकल टर्क कर्मचारी ≠ 20 लक्षित उपयोगकर्ता। यदि दांव उच्च हैं तो डोमेन पैनल के लिए भुगतान करें।
- **FAD संदर्भ सेट पर निर्भर करता है।** मॉडल के बीच समान संदर्भ वितरण के साथ तुलना करें।
- **संश्लेषित WER.** A 5% WER कुल मिलाकर 30% छिपा सकता है WER जनसांख्यिकीय स्लाइस के अनुसार रिपोर्ट।
- **सार्वजनिक बेंचमार्क संतृप्ति।** अधिकांश सीमा मॉडल मानक बेंचमार्क पर छत के पास हैं। एक घर में रखा सेट बनाएं जो आपके ट्रैफ़िक को प्रतिबिंबित करता है।

## इसे भेजें

के रूप में सहेजें `outputs/skill-audio-evaluator.md`किसी भी ऑडियो मॉडल रिलीज के लिए मेट्रिक्स, बेंचमार्क और रिपोर्टिंग प्रारूप चुनें।

## व्यायाम

1. **- आराम से।** दौड़ें `code/main.py`गणना WER / CER / EER / SECS / FAD-ish / MMAU-ish खिलौना इनपुट पर।
2. **मध्यम।** एक निर्माण TTS यात्रा WER अपने Kokoro या F5-TTS विस्पर के माध्यम से आउटपुट. गणना WER 50 से अधिक संकेत। WER &gt; 10%
3. **कठिन.** 10वीं कक्षा का स्कोर LALM विकल्प पर MMAU-Pro भाषण + बहु-ऑडियो उपसमूह (50 वस्तुओं प्रत्येक) प्रति श्रेणी सटीकता रिपोर्ट और प्रकाशित संख्या के साथ तुलना करें।

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| WER | ASR स्कोर | `(S+D+I)/N` सामान्यीकरण के बाद शब्द स्तर पर। |
| CER | चरित्र WER | स्वर भाषाओं या चार-स्तर प्रणालियों के लिए। |
| MOS | मानव राय | 1-5 रेटिंग; 20+ श्रोता × 100 नमूने। |
| UTMOS | ML MOS भविष्यवाणी | सीखा गया मॉडल; मानव के साथ ~0.9 संबद्ध MOS. |
| SECS | आवाज-क्लोन समानता | ECAPA संदर्भ और क्लोन के बीच कॉसिन। |
| EER | स्पीकर सत्यापन स्कोर | सीमा जहां FAR = FRR. |
| DER | डायरीकरण स्कोर | (FA + Miss + Confusion) / total. |
| FAD | संगीत-जन गुण | पर फ्रेचेट दूरी VGGish सम्मिलित करना। |
| RTFx | पारगमन | प्रति दीवार घड़ी सेकंड ऑडियो सेकंड. |

## आगे पढ़ना

- [ज्वार](https://github.com/jitsi/jiwer) — WER/CER सामान्यीकरण उपयोगिताओं के साथ पुस्तकालय।
- [UTMOS (सैकी और अन्य 2022)](https://arxiv.org/abs/2204.02152) सीखा MOS भविष्यवाणी।
- [फ्रेचेट ऑडियो दूरी (किल्गुर एट अल 2019)](https://arxiv.org/abs/1812.08466) संगीत-जन मानक।
- [खुला ASR रैंडरबोर्ड](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) 2026 लाइव रैंकिंग।
- [TTS एरेना](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) मानव-मत TTS रैंकिंग बोर्ड।
- [MMAU-Pro बेंचमार्क](https://mmaubenchmark.github.io/) — LALM तर्क की रैंकिंग बोर्ड।
- [HEAR बेंचमार्क](https://hearbenchmark.com/) ऑडियो SSL बेंचमार्क।
