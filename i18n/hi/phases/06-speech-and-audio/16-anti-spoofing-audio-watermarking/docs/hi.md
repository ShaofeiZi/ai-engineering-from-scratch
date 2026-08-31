# आवाज विरोधी स्पूफिंग और ऑडियो वॉटरमार्किंग ASVspoof 5, AudioSeal, WaveVerify

> 2026 उत्पादन आवाज प्रणालियों दो चीजों की जरूरत हैः एक डिटेक्टर (AASIST, RawNet2) जो वास्तविक और नकली भाषण को वर्गीकृत करता है, और एक वॉटरमार्क (AudioSealदोनों जहाज या नहीं जहाज आवाज क्लोनिंग।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time:** ~75 minutes

## समस्या

तीन संबंधित रक्षाएंः

1. **एंटी-स्पूफिंग / डीपफैक डिटेक्शन।** ऑडियो क्लिप को देखते हुए, क्या यह सिंथेटिक है या असली? ASVspoof बेंचमार्क (ASVspoof 2019 → 2021 → 5) स्वर्ण मानक हैं।
2. **ऑडियो वॉटरमार्किंग।** उत्पन्न ऑडियो में एक अदृश्य संकेत एम्बेड करें जिसे एक डिटेक्टर बाद में निकाल सकता है। AudioSeal (मेटा) और WavMark खुले विकल्प हैं।
3. **प्रवाहित मूल।** ऑडियो फ़ाइलों + मेटाडेटा के क्रिप्टोग्राफिक हस्ताक्षर। C2PA / सामग्री प्रामाणिकता पहल.

पता लगाने विरोधी को संभालने जो सहयोग नहीं करते हैं। वॉटरमार्किंग अनुपालन को संभालने AI-generated ऑडियो को इस तरह से पहचानने योग्य होना चाहिए। दोनों की आवश्यकता 2026 में है।

## अवधारणा

![एंटी-स्पूफिंग बनाम वॉटरमार्किंग बनाम मूल  तीन रक्षा परतें](../assets/spoofing-watermark.svg)

### ASVspoof 5  2024-2025 के लिए बेंचमार्क

पिछले संस्करणों से सबसे बड़ा परिवर्तनः

- **जनसांख्यिकीय स्रोत** (स्टूडियो साफ नहीं)  यथार्थवादी परिस्थितियों.
- **~ 2000 स्पीकर** (से पहले ~ 100 के खिलाफ) ।
- **32 हमले एल्गोरिदम.** TTS + voice conversion + adversarial perturbation.
- **दो निशान.** प्रति उपाय (CM) स्वतंत्र पहचान; स्पूफिंग-रोबस्ट ASV (SASV) बायोमेट्रिक प्रणालियों के लिए।

नवीनतम ASVspoof 5: ~7.23% EER. बुजुर्गों पर ASVspoof 2019 LA: 0.42% EER. वास्तविक दुनिया में तैनातीः 5-10% की उम्मीद करें EER जंगली क्लिप पर।

### AASIST और RawNet2 पता लगाने के मॉडल परिवार

**AASIST** (2021, 2026 तक अद्यतन) स्पेक्ट्रल विशेषताओं पर ग्राफ-अवलोकन। वर्तमान SOTA पर ASVspoof 5 प्रति उपाय कार्य।

**RawNet2.** Convolutional front-end over raw waveform + TDNN रीढ़ की हड्डी. सरल आधार रेखा; अभी भी बारीक समायोजन के साथ प्रतिस्पर्धी.

**NeXt-TDNN + SSL विशेषताएं।** 2025 संस्करणः ECAPA-style + WavLM विशेषताएं + फोकल हानि 0.42% तक पहुंचता है EER पर ASVspoof 2019 LA.

### AudioSeal 2024 वॉटरमार्क डिफ़ॉल्ट

मेटा **AudioSeal** (जनवरी 2024, v0.2 मुख्य डिजाइनः

- **स्थानीयकृत.** प्रति फ्रेम 16 पर वॉटरमार्क का पता लगाता है kHz नमूना संकल्प (1/16000 s)
- **Generator + detector jointly trained.** जनरेटर अदृश्य संकेत को एम्बेड करना सीखता है; डिटेक्टर इसे वृद्धि के माध्यम से ढूंढना सीखता है।
- **मजबूत।** जीवित रहे MP3 / AAC संपीड़न, EQ, गति-परिवर्तन ±10%, शोर मिश्रण +10 dB SNR.
- **जल्दी करो.** डिटेक्टर 485× वास्तविक समय पर चलाता है; 1000× से तेज WavMark.
- **क्षमता।** 16-बिट उपयोगिता लोड (कोडिंग मॉडल कर सकते हैं ID, पीढ़ी का समयशीर्षक, उपयोगकर्ता ID) प्रत्येक कथन में सम्मिलित किया जा सकता है।

### WavMark

पूर्व-AudioSeal मूल लाइन खोले. उल्टा तंत्रिका नेटवर्क, 32 बिट्स / सेकंड. समस्याएंः

- क्रूर बल सिंक्रनाइज़ेशन धीमा है।
- Gaussian शोर या MP3 संपीड़न।
- वास्तविक समय में दोस्ताना नहीं।

### WaveVerify (जुलाई 2025)

पते AudioSealविशेष रूप से समय के साथ हेरफेर (उपवर्तन, गति) । FiLM-based जनरेटर + विशेषज्ञों के मिश्रण डिटेक्टर। AudioSeal मानक हमलों पर; समय संपादन संभालता है।

### अंतर विरोधी शोषण

से AudioMarkBench: "पीच शिफ्ट के तहत, सभी वॉटरमार्क 0.6 से नीचे बिट रिकवरी सटीकता दिखाते हैं, जो लगभग पूर्ण हटाने का संकेत देता है। " **पिच-शिफ्ट सार्वभौमिक हमला है।** 2026 वॉटरमार्क आक्रामक पिच संशोधन के लिए पूरी तरह से मजबूत है।AASIST) के साथ जलचिह्नित करना।

### C2PA / सामग्री प्रामाणिकता पहल

नहीं ML तकनीक  एक स्पष्ट प्रारूप। ऑडियो फ़ाइलें निर्माण उपकरण, लेखक, तिथि के बारे में क्रिप्टोग्राफिक रूप से हस्ताक्षरित मेटाडेटा लेती हैं। ऑडबॉक्स / सीमलेस इसका उपयोग करें। उत्पत्ति के लिए अच्छा है; कुछ भी नहीं करता है यदि एक बुरा अभिनेता री-कोडिंग और मेटाडेटा स्ट्रिप्स करता है।

```figure
v4-audio-watermark
```

## इसे बनाओ

### चरण 1: एक सरल स्पेक्ट्रल फीचर डिटेक्टर (खेलौना)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

सिंथेटिक भाषण में अक्सर असामान्य रूप से उच्च आवृत्ति ऊर्जा होती है। उत्पादन डिटेक्टर उपयोग करते हैं AASISTलेकिन अंतर्ज्ञान सही है।

### चरण 2: AudioSeal embed + detect

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### चरण 3: मूल्यांकन EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### चरण 4: उत्पादन एकीकरण

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

प्रत्येक पीढ़ी के जहाजों मेंः (1) वॉटरमार्क, (2) हस्ताक्षरित मैनिफस्ट, (3) भंडारण नीति के अनुरूप लेखा परीक्षा लॉग।

## इसका प्रयोग करें

| उपयोग के मामले | रक्षा |
|----------|---------|
| शिपिंग TTS / आवाज क्लोनिंग | AudioSeal प्रत्येक आउटपुट पर एम्बेड (न-negotiable) |
| बायोमेट्रिक आवाज अनलॉक | AASIST + ECAPA समूह; जीवन चुनौती |
| कॉल सेंटर धोखाधड़ी का पता लगाना | AASIST 20% आमंत्रित कॉल के नमूने पर |
| पॉडकास्ट प्रामाणिकता | C2PA अपलोड पर हस्ताक्षर करना, AudioSeal यदि AI-generated |
| अनुसंधान/शिक्षण डिटेक्टर | ASVspoof 5 ट्रेन/डेव/एवल सेट |

## फंदे

- **पानी का निशान बिना डिटेक्टर कभी काम कर रहा है.** अपने में डिटेक्टर भेजें CI.
- **माप के बिना पता लगाना।** AASIST प्रशिक्षित ASVspoof LA ओवरफिट, वास्तविक दुनिया की सटीकता में गिरावट. अपने डोमेन पर मापें.
- **पिच-शिफ्ट अंतर.** आक्रामक पिच शिफ्ट अधिकांश जलचिह्नों को हटा देता है।
- **मेटाडेटा स्ट्रिप और होस्ट.** C2PA हमेशा क्रिप्टोग्राफिक + धारणा (वाटरमार्क) रक्षा को एक साथ जोड़ें।
- **पहचान के रूप में जीवंतता।** उपयोगकर्ता से एक यादृच्छिक वाक्यांश कहने के लिए कहें। दोहराव हमलों को रोकता है लेकिन वास्तविक समय में क्लोनिंग नहीं।

## इसे भेजें

के रूप में सहेजें `outputs/skill-spoof-defender.md`. वॉयस-जेन तैनाती के लिए डिटेक्शन मॉडल, वॉटरमार्क, उद्गम पत्र और ऑपरेशनल प्लेबुक चुनें।

## व्यायाम

1. **- आराम से।** दौड़ें `code/main.py`. खिलौना डिटेक्टर + खिलौना वॉटरमार्क सिंथेटिक ऑडियो पर एम्बेड/डिटेक्ट करें।
2. **मध्यम।** स्थापित करें `audioseal`, एक 16 बिट उपयोगिता लोड में एम्बेड TTS ध्वनि शोर के साथ भ्रष्ट और बिट रिकवरी सटीकता मापने.
3. **कठिन.** ठीक-ठीक ए RawNet2 या AASIST पर ASVspoof 2019 LA. उपाय EER. एक लंबे समय तक चलने वाले सेट पर परीक्षण F5-TTS-generated क्लिप  देखें कैसे OOD पता लगाने में गिरावट आती है।

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| ASVspoof | बेंचमार्क | द्विवार्षिक चुनौती; 2024 = ASVspoof 5. |
| CM (countermeasure) | डिटेक्टर | वर्गीकरणकर्ता: वास्तविक भाषण बनाम सिंथेटिक / परिवर्तित। |
| SASV | Speaker verif + CM | Integrated biometric + spoof detection. |
| AudioSeal | मेटा वॉटरमार्क | स्थानीयकृत, 16-बिट उपयोगिता लोड, 485x से तेज WavMark. |
| बिट रिकवरी सटीकता | जलचिह्न जीवित रहना | हमले के बाद बरामद उपयोगी लोड बिट्स का एक अंश। |
| C2PA | उत्पत्ति पत्र | सृजन/लेखकत्व के बारे में क्रिप्टोग्राफिक मेटाडेटा। |
| AASIST | डिटेक्टर परिवार | ग्राफ-आधारित ध्यान-आधारित एंटी-स्पाउफिंग SOTA. |

## आगे पढ़ना

- [टोडिस्को एट अल. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) वर्तमान बेंचमार्क।
- [डेफोसेस एट अल (2024). AudioSeal](https://arxiv.org/abs/2401.17264) डिफ़ॉल्ट वॉटरमार्क।
- [चेन और अन्य (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — MoE समय के हमले के लिए डिटेक्टर।
- [जंग और अन्य (2022). AASIST](https://arxiv.org/abs/2110.01200)  SOTA पता लगाने की रीढ़ की हड्डी।
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) मजबूती का मूल्यांकन।
- [C2PA विनिर्देश](https://c2pa.org/specifications/specifications/) प्रवासन प्रपत्र प्रारूप।
