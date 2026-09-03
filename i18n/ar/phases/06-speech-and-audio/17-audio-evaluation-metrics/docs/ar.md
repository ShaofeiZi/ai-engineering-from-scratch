# تقييم صوتي WER, MOS, UTMOS, MMAU, FAD، وخطوط الرؤية المفتوحة

> لا يمكنك شحن ما لا يمكنك قياسه. هذه الدروس تعدد المقاييس 2026 لكل مهمة صوتية: ASR (WER, CER, RTFx), TTS (MOS, UTMOS, SECS, WER-on-ASR-round-trip() اللغة الصوتية (MMAU, LongAudioBenchالموسيقىFAD, CLAP() و المتحدث (EERبالإضافة إلى قائمة النتائج التي تقارنها

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time:** ~60 minutes

## المشكلة

كل مهمة صوتية لديها مقاييس متعددة، كل قياس محور مختلف. باستخدام المقياس الخاطئ هو كيفية شحن نموذج يبدو رائعا على لوحة التحكم الخاص بك و رهيبة في الإنتاج. قائمة 2026 القنوني:

| المهمة | الأساسية | ثانوية |
|------|---------|-----------|
| ASR | WER | CER · RTFx · تأخير أول علامة |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| النسخ الصوتي | SECS (ECAPA (كوزين) | MOS · CER |
| التحقق من المتحدث | EER | minDCF · FAR / FRR في نقطة التشغيل |
| الإسهال | DER | JER • حيرة المتحدثين |
| تصنيف الصوت | أعلى - 1 mAP | الكليات F1 · استدعاء لكل فئة |
| جيل الموسيقى | FAD | CLAP • لوحة الاستماع MOS |
| نموذج لغة الصوت | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| التدفق S2S | التأخير P50/P95 | WER · MOS |

## المفهوم

![المصفوفة لتقييم الصوت  المقاييس مقابل المهام مقابل قائمة النتائج لعام 2026](../assets/eval-landscape.svg)

### ASR المقاييس

**WER (معدل خطأ الكلمات).** `(S + D + I) / N`الحروف الصغرى، التخطيط، تعاديل الأرقام قبل تسجيل النقاط `jiwer` أو OpenAI- نعم `whisper_normalizer`. &lt;5% = قراءة الكلام على قدم المساواة البشرية

**CER (معدل خطأ الشخصية).** نفس الصيغة، مستوى الأحرف. تستخدم لغات النغمات (الماندرين، الكانتوني) حيث التقسيم الكليمي غير واضح.

**RTFx (عامل الوقت الحقيقي المعاكس)** ثواني صوتية معالجة لكل ثانية من الساعة الجدارية أعلى أفضلTDT يصل إلى 3380×.v3 هو ~ 30 ×.

**تأخير أول رمز.** ساعة الحائط من إدخال الصوت إلى أول رمز النسخة حرجة للتسجيل

### TTS المقاييس

**MOS (متوسط درجة الرأي)** 1-5 تصنيف بشري، معيار الذهب لكن بطيء، جمع أكثر من 20 سمعًا لكل عينة، أكثر من 100 عينة لكل نموذج.

**UTMOS (2022-2026).** تعلمت MOS التنبؤ. يتوافق مع ~ 0.9 مع البشر MOS على المعايير المعتادة. F5-TTS: UTMOS 3.95، الحقيقة الأساسية: 4.08.

**SECS (مُشاركة الكوزين في تشبيه المتحدثين)** لتنسيق الصوت ECAPA إضافة كوسين بين النتائج المرجعية والمتكررة. &gt; 0.75 = الكلون المعترف به.

**WER-on-ASR-round-trip.** إركض على "سيسبر" TTS الناتج، الحساب WER يلتقط رجعات التفاهم SOTA: &lt;2٪ CER.

**TTFA (أوقات إلى أول صوت).** تأخير الساعة الجدارية كوكورو-82م: ~100 ms F5-TTS: ~ 1 ثانية

### خاصة في عملية استنسخ الصوت

**SECS + MOS + CER** مثل ثلاثية. التنسيق الذي يسجل عالية SECS لكن منخفض MOS يعني الاهتمام بالصوت الصحيح ولكن غير الطبيعي، والعكس يعني الصوت الطبيعي ولكن المتحدث الخطأ.

### التحقق من المتحدث

**EER (معدل الأخطاء المتساوي)** العدالة التي يُعادل فيها معدل قبول كاذب معدل رفض كاذب. ECAPA على VoxCeleb1-O: 0.87%.

**minDCF (معدل أقل من تكلفة الكشف)** تكلفة معينة في نقطة تشغيل مختارة (غالباً ما تكون FAR=0.01) أكثر صلة بالإنتاج من EER.

### الإسهال

**DER (معدل خطأ الإفراط في التخدير).** `(FA + Miss + Confusion) / total_speaker_time`. غياب الكلام + كذب إيقاع الكلام + المتحدث-الارتباك، كل كجزء. AMI الاجتماعات: DER ~ 10-20% هو واقعي. ملاحظة 3.1 + دقة-2 تجارية: &lt;10% DER على صوت مسجل جيد

**JER (معدل خطأ جاكارد)** بديل DER، قوية إلى التأثير القصير.

### تصنيف الصوت

متعددة العلامات: **mAP (متوسط دقة)** على جميع الفصول AudioSet: 0.548 mAP لـ BEATs-iter3.

حصرية متعددة الفئات: **الدقة الأولى، الأولى الخامسة**. أوامر الكلام v2: 99.0% أعلى 1 (صوتي-MAE).

غير متوازن: **الكليات F1** + **إعادة التذكر لكل فئة**. تقرير لكل فئة  الدقة الإجمالية تخفي فئات الفئة الفاشلة.

### جيل الموسيقى

**FAD (مدى صوتي (فريشيت)** المسافة بين VGGish-embedding توزيعات الصوت الحقيقي مقابل الصوت المولود. MusicGen-small على MusicCaps: 4.5. MusicLMأضعها أفضل

**CLAP -نقطة** درجة التنحية النصية الصوتية باستخدام CLAP التوابل. &gt; 0.3 = التوجه المعقول.

**لوحة الاستماع MOS.** لا تزال الكلمة الأخيرة للموسيقى المستهلكة v5 ELO 1293 on TTS أرا (من تفضيلات الإنسان المزدوجة).

### معايير اللغة الصوتية

**MMAU (فهم متعدد الصوت)** صوت 10KQA أزواج

**MMAU-Pro.** 1800 عنصر صلب، أربعة فئات: الكلام / الصوت / الموسيقى / متعددة الصوت. فرصة عشوائية 25% على طريق أربع. Gemini 2.5 إعلان عام ~ 60%؛ متعددة الصوت ~ 22% على جميع الطرازات.

**LongAudioBench.** مقاطع متعددة الدقائق مع استفسارات معنوية. Gemini 2.5 Pro.

**AudioCaps -كلوتو** إضافة أسماء للمعايير SPICE, CIDEr, FENSE المقاييس

### التدفق من حديث إلى حديث

**التأخير P50 / P95 / P99.** ساعة الحائط من نهاية المستخدم إلى الاستجابة السمعة الأولى GPT-4o الوقت الحقيقي: 300 ms

**WER / MOS** على الخروج

**استجابة متطرفة** وقت من توقف المستخدم إلى مساعد صامت الهدف 150 ثانية

### قائمة اللائحة لعام 2026

| اللوحة الرائدة | أثر | URL |
|------------|--------|-----|
| مفتوح ASR اللوحة الرابطة (HF) | English + multilingual + long-form | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS (أرينا)HF) | الإنجليزية TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, ELO من الأصوات المزدوجة | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM التفكير | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | الاعتراف بالمتحدثين | `voxsrc.github.io` |
| MMAU مجموعة موسيقية | الموسيقى LALM | (في MMAU) |
| HEAR مقياس | صوتي مرصد ذاتي | `hearbenchmark.com` |

```figure
sp-wer-align
```

## بناءها

### الخطوة الأولى: WER مع التطبيع

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

### الخطوة الثانية: TTS رحلة ذهاب و ذهاب WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### الخطوة الثالثة: SECS للتنسخ الصوتي

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### الخطوة الرابعة: FAD لإنتاج الموسيقى

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### الخطوة الخامسة: EER للتحقق من المتحدث (مثل الرمز الذي يستخدم في الدروس 6)

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

## استخدمها

إزواج كل عملية نشر مع حزمة تقييم ثابتة التي تعمل على كل تحديث نموذج.

1. **عادي قبل أن تسجل** الحروف الصغرى، شريط النقاط، رقم التوسع، إبلغ عن قاعدة التطبيع
2. **تقرير التوزيعات، وليس المتوسطات.** P50/P95/P99 لخفض التخميس. استدعاء لكل فئة للتصنيف. لكل فئة لل MMAU.
3. **إشغال مقياس عام واحد.** حتى لو كانت بيانات الإنتاج الخاصة بك تختلف، تقرير على Open ASR / TTS أراينا MMAU يسمح للمراجعين بمقارنة التفاح مع التفاح

## الفخاخ

- **UTMOS الاستخراج** تدريب على VCTK-style كلام نظيف؛ تسجل صوت ضجيج / مستخدما / عاطفي بشكل سيء.
- **MOS تحيز اللوحة** 20 عامل في شركة أمازون ميكانيكال تورك ≠ 20 مستخدم هدف. ادفع للوحة النطاق إذا كانت المخاطر مرتفعة.
- **FAD يعتمد على مجموعة المرجعية.** مقارنة مع نفس التوزيع المرجعي بين النماذج.
- **المجموعة WER.** A 5% WER في المجموع يمكن أن تخفي 30٪ WER على الخطاب المجهد، تقرير حسب النقطة الديموغرافية
- **التشبع في المؤشر العام** معظم الطرازات الحدودية قريبة من السقف على مقارنات قياسية. قم ببناء مجموعة محمولة داخلية تعكس حركة المرور الخاصة بك.

## أرسله

إحتفظ بها `outputs/skill-audio-evaluator.md`. اختر المقاييس والمؤشرات المرجعية و تنسيق التقارير لأي إصدار من نماذج الصوت

## التمارين

1. **-بسهولة** أركض `code/main.py`-حساب WER / CER / EER / SECS / FAD-ish / MMAU-ish على مدخلات الألعاب
2. **متوسط** بناء TTS رحلة ذهاب و ذهاب WER أطلقوا على كوكورو أو F5-TTS النتائج من خلال Whisper WER أكثر من 50 إشارة. WER 10%
3. **صعب** احصل على دروسك 10 LALM الخيار MMAU-Pro الكلام + مجموعة فرعية متعددة الصوت (50 عنصر لكل). تقرير دقة لكل فئة ومقارنة مع الرقم المنشور.

## الشروط الرئيسية

| المدة | ما يقوله الناس | ما يعنيه هذا في الواقع |
|------|-----------------|-----------------------|
| WER | ASR النتيجة | `(S+D+I)/N` على مستوى الكلمات بعد التطبيع |
| CER | شخصية WER | لغات النغمات أو أنظمة مستوى الكار. |
| MOS | رأي البشر | تصنيف 1-5؛ 20 + سمع × 100 عينات. |
| UTMOS | ML MOS الوقود | النموذج المتعلم؛ يتوافق مع الإنسان ~ 0.9 MOS. |
| SECS | تشابه النسخ الصوتية | ECAPA كوسين بين المرجع والإستنساخ |
| EER | مؤشر التحقق من المتحدث | الحد الأدنى حيث FAR = FRR. |
| DER | درجة الإسهال | (FA + Miss + Confusion) / total. |
| FAD | جين الموسيقى | مسافة فريشيه على VGGish التوابل |
| RTFx | التشغيل | ثواني صوتية لكل ثانية |

## المزيد من القراءة

- [الـ (جيوير)](https://github.com/jitsi/jiwer) — WER/CER مكتبة مع خدمات التطبيع
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) تعلمت MOS الوقائف
- [مسافة صوتية Fréchet (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) معيار الموسيقى
- [مفتوح ASR اللوحة الرائدة](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) 2026 ترتيبات حية.
- [TTS أراينا](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) صوت البشر TTS قائمة النسب
- [MMAU-Pro مقياس](https://mmaubenchmark.github.io/) — LALM قائمة التفكير
- [HEAR مقياس](https://hearbenchmark.com/) صوتي SSL المعايير
