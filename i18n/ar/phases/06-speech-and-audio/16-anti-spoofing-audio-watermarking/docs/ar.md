# مكافحة الكذب الصوتي والإشارة إلى المياه الصوتية ASVspoof 5, AudioSeal, WaveVerify

> إنّ عملية استنساخ الصوت تمّ إرسالها أسرع من الدفاعات.AASIST, RawNet2) التي تصنف الخطاب الحقيقي مقابل الكلام المزيف، و علامة مائية (AudioSeal() التي تتعدى الضغط والتحرير. شحن كل من أو لا شحن النسخ الصوتية.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time:** ~75 minutes

## المشكلة

ثلاثة دفاعات ذات صلة:

1. **مكافحة التزوير / الكشف عن مزيفة عميقة** بالنظر إلى شريط صوتي، هل هو مصنوعي أم حقيقي؟ ASVspoof المؤشرات المرجعية (ASVspoof 2019 → 2021 → 5) هي المعيار الذهبي.
2. **علامة مياه صوتية** تضمين إشارة غير واضحة في الصوت الذي يتم إنشاؤه يمكن أن يستخرجه جهاز كشف لاحقاً. AudioSeal (ميتا) و WavMark هذه الخيارات المفتوحة
3. **-أصل مؤكد** توقيع رمزي لملفات الصوت + البيانات المعدنية. C2PA مبادرة مصادقة المحتوى

الكشف يتعامل مع المعارضين الذين لا يتعاونونون معهم. AI-generated يجب أن يكون الصوت قابلا للتعرف عليه على أنه كذلك. كلاً منهما مطلوب في عام 2026.

## المفهوم

![مكافحة التزوير مقابل علامة المياه مقابل المأصل  ثلاث طبقات دفاع](../assets/spoofing-watermark.svg)

### ASVspoof 5  مقياس 2024-2025

أكبر تغيير من الإصدارات السابقة:

- **البيانات المستمدة من الجماهير** (ليس استوديو نظيف)  الظروف الواقعية.
- **~ 2000 متحدث** (مقابل 100 قبل).
- **32 خوارزمية هجوم** TTS + voice conversion + adversarial perturbation.
- **اثنين من المسارات.** التدابير المضادة (CM) الكشف عن نفسها؛ التخريب القوي ASV (SASV) للأنظمة البيومترية.

أحدث المعلومات ASVspoof 5: ~7.23% EER. على الأكبر سنا ASVspoof 2019 LA: 0.42% EER. النشر في العالم الحقيقي: توقع 5-10% EER على المقاطع البرية

### AASIST و RawNet2 عائلات نموذج الكشف

**AASIST** (2021، تم تحديثه حتى 2026). الاهتمام الرسمي على الخصائص الطيفية. SOTA على ASVspoof 5 مهمة التدابير المضادة

**RawNet2.** Convolutional front-end over raw waveform + TDNN العمود الفقري. خط أساسي بسيط؛ لا يزال تنافسيا مع ضبط دقيقة.

**NeXt-TDNN + SSL الميزات** طراز 2025: ECAPA-style + WavLM الميزات + فقدان التركيز. يصل إلى 0.42% EER على ASVspoof 2019 LA.

### AudioSeal الوضع المتباعد في علامة المياه 2024

الميتا **AudioSeal** (كانون الثاني 2024) v0.2 (ديسمبر 2024) التصميم الرئيسي:

- **محليّة** يكتشف علامة المياه لكل إطار عند 16 kHz حل العينة (1/16000s).
- **Generator + detector jointly trained.** يتعلم المولد إدراج إشارة غير مسموعة، ويتعلم الكاشف العثور عليها من خلال التكثيف.
- **قوية** نجا MP3 / AAC الضغط EQ، تغيير السرعة ±10٪ ، مزيج الضوضاء +10 dB SNR.
- **بسرعة** الكشف يعمل في 485 × في الوقت الحقيقي؛ 1000 × أسرع من WavMark.
- **-إمكانيات** الحمل المفيد 16 بت (يمكن تشفير النموذج ID, طابع زمني للإنتاج , المستخدم ID) يمكن دمجها في كل كلمة.

### WavMark

-الـAudioSeal إنّه مفتوح، شبكة عصبية قابلة للتعديل، 32 بت/ثانية.

- التزامن القوة الخام بطيئة
- يمكن إزالتها بواسطة ضجيج غوسيا أو MP3 الضغط
- ليس صديقاً في الوقت الحقيقي

### WaveVerify (يوليو 2025)

العناوين AudioSealنقاط الضعف  تحديدا التلاعبات الزمنية (العكس، السرعة). FiLM-based المولد + كاشف مزيج من الخبراء. AudioSeal على الهجمات القياسية؛ يتعامل مع التحريرات الزمنية.

### الفجوة التي يستغلونها خصومها

من AudioMarkBench: "في ظل تغيير اللحظة، تظهر جميع علامات المياه دقة استعادة البيت أقل من 0.6، مما يشير إلى إزالة شبه كاملة". **التحول هو الهجوم العالمي** علامة المائية رقم 2026 قوية تماما لتعديل ارتفاع عدواني. لهذا السبب تحتاج إلى الكشف (AASIST) إلى جانب علامة المياه.

### C2PA / مبادرة مصداقية المحتوى

لا ML تقنية  شكل واضح. الملفات الصوتية تحمل بيانات متفرغة موقعة عن أداة الإنشاء، المؤلف، التاريخ. أودوبوكس / سليم لا تستخدمها. جيد للمصدر؛ لا يفعل شيئا إذا كان الفاعل السيئ إعادة تشفير وتقطيع البيانات المعدنية.

```figure
v4-audio-watermark
```

## بناءها

### الخطوة الأولى: جهاز كشف أشكال الطيف بسيط (لعبة)

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

في كثير من الأحيان يكون للخط الألغام الصناعية طاقة عالية التردد بشكل غير عادي. AASISTليس هذا، لكن الحدس يُمْكِنُ.

### الخطوة الثانية: AudioSeal embed + detect

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

### الخطوة الثالثة: التقييم EER

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

### الخطوة الرابعة: تكامل الإنتاج

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

كل جيل من السفن: (1) علامة المائية، (2) مذكرة توقيع، (3) سجل مراجعة مطابقة بسياسة الاحتفاظ.

## استخدمها

| حالة الاستخدام | الدفاع |
|----------|---------|
| الشحن TTS / استنسخ الصوت | AudioSeal تم تضمين كل خروج (غير قابل للتفاوض) |
| إفتاح صوتي بيوميدي | AASIST + ECAPA المجموعة؛ تحدي الحياة |
| الكشف عن احتيال مركز الاتصال | AASIST على 20٪ من عينات المكالمات الداخلية |
| أصالة البودكاست | C2PA التوقيع على تحميل، AudioSeal إذا AI-generated |
| أجهزة تحديد البحوث / التدريب | ASVspoof 5 مجموعات قطارات/دب/طريق |

## الفخاخ

- **علامة مياه بدون جهاز كشف يعمل** لا فائدة من ذلك، أرسل الكاشف في CI.
- **الكشف دون تصويب** AASIST تدرب على ASVspoof LA التخطيطات، انخفاض دقة العالم الحقيقي، تحسّن على مستوى المجال الخاص بك.
- **فجوة في التحول** تغيير الوضع العنيف يزيل معظم علامات المياه
- **إضافة البيانات المعدنية** C2PA يمكن تجنبها بشكل بسيط عن طريق إعادة التشفير. دائماً أضف الدفاع الرموزي + التدريبي (العلامة المائية) معاً.
- **الحيوية كاكتشاف** اطلب من المستخدم قول عبارة عشوائية، يمنع هجمات التكرار ولكن لا التنسيق في الوقت الحقيقي.

## أرسله

إحتفظ بها `outputs/skill-spoof-defender.md`. اختر نموذج الكشف ، علامة مياه ، دليل المأصل ، و دليل تشغيل لتشغيل الجين الصوتي.

## التمارين

1. **-بسهولة** أركض `code/main.py`. جهاز كشف الألعاب + علامة مياه للاعب تضمين/كشف على الصوت الاصطناعي.
2. **متوسط** إثباط `audioseal`، تضمين حمولة مفيدة 16 بت في TTS أخرج، إعادة تشفير، إفسد الصوت بالضوضاء و قياس دقة استرداد البيت
3. **صعب** تحديد الميزات RawNet2 أو AASIST على ASVspoof 2019 LA. القياس EER. اختبار على مجموعة متواصلة من F5-TTS-generated المقاطع  انظر كيف OOD يقلل الاكتشاف

## الشروط الرئيسية

| المدة | ما يقوله الناس | ما يعنيه هذا في الواقع |
|------|-----------------|-----------------------|
| ASVspoof | المؤشر المرجعي | التحدي الثنائي؛ 2024 = ASVspoof 5. |
| CM (countermeasure) | الكاشف | التصنيف: الكلام الحقيقي مقابل الصناعي / المتحول. |
| SASV | Speaker verif + CM | Integrated biometric + spoof detection. |
| AudioSeal | علامة ميتا | محلية، 16 بت الحمل المفيد، 485 × أسرع من WavMark. |
| دقة استعادة القليل | البقاء على قيد الحياة | جزء من قطع الحمولة المفيدة التي تم استعادتها بعد الهجوم |
| C2PA | مذكرة المصل | البيانات المختفية حول الإبداع / المؤلفية. |
| AASIST | عائلة الكشف | مكافحة التزوير القائمة على الرسم البياني SOTA. |

## المزيد من القراءة

- [(Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) المعيار المرجعي الحالي.
- [ديفوسيز وآخرون (2024). AudioSeal](https://arxiv.org/abs/2401.17264) علامة المياه الافتراضية.
- [(تشن) وآخرون (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — MoE جهاز كشف الهجمات الزمنية
- [جونغ وغيره (2022). AASIST](https://arxiv.org/abs/2110.01200) المجلس SOTA العمود الفقري للكشف
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) تقييم الصمود
- [C2PA المواصفات](https://c2pa.org/specifications/specifications/) صيغة بيان المصلحة.
