# स्पीकर पहचान और सत्यापन

> ASR "क्या कहा था? " स्पीकर पहचान पूछता है "कौन कहा था? " गणित एक ही लग रहा है  एम्बेडेड प्लस cosine  लेकिन प्रत्येक उत्पादन निर्णय एक एकल पर निर्भर करता है EER संख्या।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 22 (Embedding Models)
**Time:** ~45 minutes

## समस्या

एक उपयोगकर्ता एक पासवर्ड कहता है. आप जानना चाहते हैंः क्या यह वह व्यक्ति है जो वे दावा करते हैं कि वे हैं (*सत्यापन*, 1:1), या यह आपके नामांकन बैंक में पहली व्यक्ति है (*पहचान*या यह कोई अज्ञात वक्ता नहीं है*खुला सेट*)?

2018 से पहलेः GMM-UBM + i-vectors. Reasonable EER लेकिन चैनल शिफ्ट (फोन बनाम लैपटॉप) और भावना के लिए नाजुक। 20182022: एक्स-वेक्टर (TDNN कोणीय मार्जिन के साथ प्रशिक्षित रीढ़ की हड्डी) 2022+: ECAPA-TDNN और WavLM-large 2026 तक क्षेत्र में तीन मॉडल और एक मीट्रिक द्वारा वर्चस्व है।

मीट्रिक है **EER** समान त्रुटि दर. अपने निर्णय की सीमा निर्धारित करें ताकि गलत स्वीकार करें Rate = False अस्वीकार दर। क्रॉसओवर है EER. हर अखबार में इस्तेमाल किया, हर लीडरबोर्ड, हर खरीद कॉल.

## अवधारणा

![पंजीकरण + सत्यापन पाइपलाइन जिसमें एम्बेडिंग + कॉसिन + EER](../assets/speaker-verification.svg)

**पाइपलाइन.** नामांकनः लक्ष्य स्पीकर के रिकॉर्ड 530 सेकंड; निश्चित आयामी एम्बेडिंग (192-डी के लिए) की गणना करें ECAPA-TDNN, 256-डी के लिए WavLM-large) सत्यापनः परीक्षण कथन एम्बेडिंग प्राप्त करें; कॉसिन समानता की गणना करें; एक सीमा की तुलना करें।

**ECAPA-TDNN (2020, अभी भी 2026 में हावी है) ।** 1D conv ब्लॉक, स्क््रेस-इक्सिटेशन, मल्टी-हेड ध्यान बंडलिंग, इसके बाद 192-डी तक रैखिक परत। VoxCeleb 1+2 (2,700 स्पीकर, 1.1M बोलने) के साथ अतिरिक्त कोण मार्जिन हानि (AAM-softmax).

**WavLM-SV (2022+).** एक पूर्व प्रशिक्षित ठीक-ठीक WavLM-large SSL के साथ रीढ़ की हड्डी AAM उच्च गुणवत्ता लेकिन धीमी  300+ MB 15 के खिलाफ MB.

**एक्स-वेक्टर (बेसलाइन)** TDNN + सांख्यिकीय सामूहिकरण। क्लासिक; अभी भी उपयोगी CPU / किनारे.

**AAM-softmax.** अतिरिक्त मार्जिन के साथ मानक सॉफ्टमैक्स `m` कोणीय स्थान मेंः `cos(θ + m)` सही वर्ग के लिए। वर्गों के बीच कोणीय अलगाव बल. `m=0.2`, पैमाने `s=30`.

### स्कोरिंग

- **कॉसिन** प्रवेश और परीक्षा के बीच की सीमाओं पर आधारित निर्णय।
- **PLDA (संभवतः LDA).** एक लटेंट स्थान में परियोजना एम्बेडिंग जहां एक ही स्पीकर बनाम अलग-अलग स्पीकर में एक बंद-रूप संभावना अनुपात है। +1020% के लिए कॉसिन के शीर्ष पर जोड़ा गया EER 2020 से पहले का मानक; अब केवल बंद सेट सेट सेटअप में उपयोग किया जाता है।
- **स्कोर सामान्यीकरण.** `S-norm` या `AS-norm`: प्रत्येक स्कोर को धोखा देने वाले साधनों और अन्य लोगों के एक समूह के खिलाफ सामान्य बनाना।

### आपको जो संख्याएं पता होनी चाहिए (2026)

| मॉडल | VoxCeleb1-O EER | पाराम | पारगमन (A100) |
|-------|-----------------|--------|-------------------|
| एक्स-वेक्टर (क्लासिक) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV बड़ा | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### डायरीकरण

"कौन कब बोला" मल्टी स्पीकर क्लिप में। पाइपलाइनः VAD → खंड → प्रत्येक खंड → समूह (अग्लोमेरेटिव या स्पेक्ट्रल) → चिकनी सीमाएँ एम्बेड करें। आधुनिक स्टैकः `pyannote.audio` 3.1, जो एक कॉल के पीछे स्पीकर सेगमेंटेशन + एम्बेडिंग + क्लस्टरिंग को बंडल करता है। 2026 SOTA DER पर AMI ~ 15% (2022 में 23% से नीचे) है।

```figure
sp-eer-crossover
```

## इसे बनाओ

### चरण 1: खिलौना एम्बेडिंग से MFCC सांख्यिकी

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

नहीं SOTA केवल अध्यापन के लिए। `code/main.py` इसको सिंथेटिक स्पीकर डेटा पर अवधारणा के प्रमाण के रूप में उपयोग करता है।

### चरण 2: कॉसिन समानता + सीमा

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### चरण 3: EER समानता जोड़े से

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

रिटर्न (eer, threshold_at_eer) दोनों रिपोर्ट करें।

### चरण 4: उत्पादन SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### चरण 5: पियानोट के साथ डायरीज़ करें

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## इसका प्रयोग करें

2026 स्टैकः

| स्थिति | चुनें |
|-----------|------|
| बंद सेट 1:1 सत्यापन, किनारा | ECAPA-TDNN + cosine threshold |
| खुले सेट सत्यापन, क्लाउड | WavLM-SV + AS-norm |
| डायरीकरण (बैठक, पॉडकास्ट) | `pyannote/speaker-diarization-3.1` |
| स्पूफिंग के खिलाफ (पुनर्प्रसार / गहरे नकली का पता लगाना) | AASIST या RawNet2 |
| छोटे एम्बेडेड (KWS + enrollment) | Titanet-Small (NeMo) |

## फंदे

- **चैनल असंगत.** मॉडल पर प्रशिक्षित VoxCeleb (वेब वीडियो) ≠ फोन कॉल ऑडियो. हमेशा लक्ष्य चैनल पर मूल्यांकन.
- **संक्षिप्त बयान।** EER परीक्षण ऑडियो के 3 सेकंड से नीचे तेजी से गिराता है।
- **शोर के साथ पंजीकरण।** एक शोर भरा नामांकन एंकर को जहर देता है। ≥3 स्वच्छ नमूने और औसत का उपयोग करें।
- **शर्तों के बीच निश्चित सीमा।** लक्ष्य डोमेन से एक लंबे समय तक बनाए गए डेवलपर सेट पर हमेशा सीमा समायोजित करें।
- **गैर-सामान्य एम्बेडमेंट पर कोसिन।** L2-normalize पहले; अन्यथा परिमाण हावी होता है।

## इसे भेजें

के रूप में सहेजें `outputs/skill-speaker-verifier.md`. चयन मॉडल, नामांकन प्रोटोकॉल, सीमा समायोजन योजना, और धोखाधड़ी सुरक्षा।

## व्यायाम

1. **- आराम से।** दौड़ें `code/main.py`. सिंथेटिक "स्पीकर" (विभिन्न स्वर प्रोफाइल), रजिस्टर, कंप्यूटिंग EER 100 जोड़ी परीक्षण सूची पर.
2. **मध्यम।** उपयोग SpeechBrain ECAPA 30 पर VoxCeleb1 बयान (5 स्पीकर × 6 प्रत्येक) गणना EER कॉसिन बनाम PLDA.
3. **कठिन.** पूरा नामांकन → दैनिक → सत्यापन पाइपलाइन के साथ निर्माण `pyannote.audio`मूल्यांकन DER पर AMI डेव सेट।

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| EER | शीर्षक मेट्रिक | झूठी होने की सीमा Accept = False अस्वीकार करो। |
| सत्यापन | 1:1 | "क्या यह एलिस है? |
| पहचान | 1:N | "कौन बोल रहा है? |
| खुला सेट | अज्ञात संभव | परीक्षण सेट में अनारक्षित स्पीकर हो सकते हैं। |
| पंजीकरण | पंजीकरण | एक वक्ता के संदर्भ एम्बेडिंग की गणना। |
| AAM-softmax | नुकसान | अतिरिक्त कोणीय मार्जिन के साथ सॉफ्टमैक्स; क्लस्टर अलगाव बल। |
| PLDA | क्लासिक स्कोरिंग | संभावनावादी LDA; सम्भाव्यता अनुपात अंकन एम्बेडमेंट के शीर्ष पर। |
| DER | डायरीकरण मेट्रिक्स | डायरीकरण त्रुटि दर  चूक + झूठी अलार्म + भ्रम। |

## आगे पढ़ना

- [स्नाइडर et al. (2018). एक्स-वेक्टरः मजबूत DNN स्पीकर पहचान के लिए एम्बेड](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) क्लासिक गहरे सम्मिलित कागज।
- [डेप्लानकेस तथा अन्य (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) प्रमुख वास्तुकला 20202026
- [चेन और अन्य (2022). WavLM: पूर्ण स्टैक भाषण प्रसंस्करण के लिए बड़े पैमाने पर स्व-निरीक्षण पूर्व-प्रशिक्षण](https://arxiv.org/abs/2110.13900) — SSL के लिए रीढ़ की हड्डी SV और डायरीकरण।
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) उत्पादन डायरीकरण + एम्बेडिंग स्टैक।
- [VoxCeleb रैंकिंग बोर्ड (2026 में अद्यतन)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) वर्तमान EER मॉडल में क्रमबद्धता।
