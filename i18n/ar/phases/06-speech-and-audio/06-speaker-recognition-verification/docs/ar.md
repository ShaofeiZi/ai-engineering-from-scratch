# التعرف على المتحدثين والتحقق منها

> ASR يسأل "ما الذي قالوه؟" يطلب التعرف على المتحدث "من قال ذلك؟" تبدو الرياضيات نفسها  التوابع زائد الكوسين  ولكن كل قرار إنتاج يعتمد على واحد EER رقم

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 22 (Embedding Models)
**Time:** ~45 minutes

## المشكلة

يستخدم مستخدم كلمة مرور. تريد أن تعرف: هل هذا الشخص الذي يدعي أنه (*التحقق*، 1:1) ، أو هو أول شخص في بنك التسجيل الخاص بك (*التعرف*أو لا  هذا هو المتحدث المجهول (*مفتوحة*)?

قبل عام 2018: GMM-UBM + i-vectors. Reasonable EER ولكن هشة لتحويل القناة (الهاتف مقابل الكمبيوتر المحمول) والعاطفة. 20182022: المتجهات x (TDNN العمود الفقري المدرب مع الحافة الزاوية). 2022+: ECAPA-TDNN و WavLM-large بحلول عام 2026، ستهيمن على هذا المجال ثلاثة نماذج ومتريكة واحدة.

المقياس هو **EER** معدل الأخطاء المتساوي. حدد عتبة قرارك حتى تقبل الكذب Rate = False معدل رفض، التقاطع هو EER. استخدمت في كل صحيفة، كل قائمة، كل دعوة مشتريات.

## المفهوم

![خط التسجيل + التحقق مع إدراج + كوسين + EER](../assets/speaker-verification.svg)

**خط الأنابيب** التسجيل: تسجيل 530 ثانية من مكبر الصوت المستهدف؛ حساب إضافة ذات الأبعاد الثابتة (192-د لل ECAPA-TDNN، 256-د لل WavLM-largeالتحقق: الحصول على إضافة بيانات الاختبار؛ حساب تشابه كوسين؛ مقارنة مع عتبة.

**ECAPA-TDNN (2020، لا يزال هيمناً 2026).** أكد الاهتمام القناة، والانتشار والجمع - شبكة عصبية تأخير الوقت. كتلة مغلقة 1D مع إثارة الضغط، جمع الاهتمام متعدد الرؤوس، تليها طبقة خطية إلى 192d. تدرب على VoxCeleb 1+2 (2،700 متحدث، 1.1 مليون كلمة) مع خسارة هامش زاوية إضافية (AAM-softmax).

**WavLM-SV (2022+).** -أحسن من قبل WavLM-large SSL العمود الفقري مع AAM الخسارة. جودة أعلى ولكن أبطأ  300+ MB مقابل 15 MB.

**المتجهة (الخط الأساسي).** TDNN + جمع الإحصاءات. CPU -حافة

**AAM-softmax.** المعدل القياسي للنحو المضاف مع الهامش المضاف `m` في المساحة الزاوية: `cos(θ + m)` للطبقة الصحيحة. قوى الفصل الزاوي بين الفئات. `m=0.2`، مقياس `s=30`.

### تسجيل النتيجة

- **كوزين** بين التسجيل والإختبار. القرار القائم على العدوان.
- **PLDA (ربما) LDA).** إضافة المشروع إلى مساحة مختفية حيث يكون لدى المتحدث نفسه مقابل المتحدث المختلف نسبة احتمالية في شكل مغلق. EER الاختزال: معيار قبل عام 2020؛ يستخدم الآن فقط في إعدادات المجموعات المغلقة.
- **-تطبيع النتائج** `S-norm` أو `AS-norm`: تعاديل كل نتيجة مقابل مجموعة من الوسائل والمعدات المزيفة.

### الأرقام التي يجب أن تعرفها (2026)

| النموذج | VoxCeleb1-O EER | "بارامز" | (الدرجةA100) |
|-------|-----------------|--------|-------------------|
| متجهات (كلاسيكية) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV كبيرة | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### الإسهال

"من تحدث متى" في شريط متعدد المتحدثين VAD → القطعة → تضمين كل قطعة → مجموعة (مجموعية أو طيفية) → حدود سلسة. كومة حديثة: `pyannote.audio` 3.1 ، الذي يجمع قسم المتكبرين + إضافة + تجميع خلف مكالمة واحدة. 2026 SOTA DER على AMI هو ~15% (انخفض من 23% في 2022).

```figure
sp-eer-crossover
```

## بناءها

### الخطوة الأولى: إضافة الألعاب من MFCC الإحصاءات

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

لا , لا SOTA على بعد ميل فقط للتدريس `code/main.py` يستخدم هذا كدليل على المفهوم على بيانات مكبرات صوتية اصطناعية.

### الخطوة الثانية: تشابه الكوسين + عتبة

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### الخطوة الثالثة: EER من أزواج التشابه

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

العائدات (eer، threshold_at_eer).

### الخطوة الرابعة: إنتاج مع SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### الخطوة 5: قم بتدوين يومياتك مع بيانوت

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## استخدمها

"مجموعة 2026"

| الوضع | اختر |
|-----------|------|
| التحقق من المجموعة المغلقة 1: 1 ، الحافة | ECAPA-TDNN + cosine threshold |
| التحقق من المجموعة المفتوحة، السحابة | WavLM-SV + AS-norm |
| التسجيلات (الاجتماعات، البودكاست) | `pyannote/speaker-diarization-3.1` |
| مكافحة التزوير (إعادة تشغيل / الكشف عن التزوير العميق) | AASIST أو RawNet2 |
| (مُدمج صغير)KWS + enrollment) | Titanet-Small (NeMo) |

## الفخاخ

- **عدم مطابقة القناة** النموذج المدرب على VoxCeleb (فيديو على شبكة الإنترنت) ≠ صوت المكالمة الهاتفية. دائما تقييم على قناة الهدف.
- **كلمات قصيرة** EER يحتقر بشكل حاد أقل من 3 ثوان من صوت الاختبار.
- **التسجيل مع الضوضاء.** واحد من المشاركات الضوضاء تسمم المُرسومة. استخدم ≥3 عينات نظيفة ومتوسط.
- **عتبة ثابتة عبر الشروط** دائماً ضبط العد على مجموعة من المطورين المحتملين من النطاق المستهدف.
- **كوزين على التوابل غير الطبيعية.** L2-normalize أولاً، وإلاً الحجم سيطر على الأمر.

## أرسله

إحتفظ بها `outputs/skill-speaker-verifier.md`- نموذج اختيار، بروتوكول التسجيل، خطة تحديد العدوان، وحماية الاحتيال.

## التمارين

1. **-بسهولة** أركض `code/main.py`.بناء "المكبرات" الاصطناعية (ملفات صوت مختلفة) ، تسجيلات، الحسابات EER في قائمة تجريبية من 100 زوج
2. **متوسط** الاستخدام SpeechBrain ECAPA في 30 VoxCeleb1 التعبيرات (كل 5 متحدثين × 6). EER مع كوسين vs PLDA.
3. **صعب** بناء التسجيل الكامل → يومي → التحقق من خط الأنابيب مع `pyannote.audio`- تقييم DER على AMI مجموعة التطوير

## الشروط الرئيسية

| المدة | ما يقوله الناس | ما يعنيه هذا في الواقع |
|------|-----------------|-----------------------|
| EER | المقياس العنوان | العدالة حيث كاذبة Accept = False رفض |
| التحقق | 1:1 | "هل هذه أليس؟" |
| التعرف | 1:N | "من يتحدث؟" |
| مفتوحة | ممكنة غير معروفة | مجموعة الاختبار يمكن أن تحتوي على مكبرات صوت غير مسجلة. |
| التسجيل | التسجيل | تحسّب إدراج مرجعية المتحدث |
| AAM-softmax | الخسارة | Softmax مع هامش زاوية إضافية؛ قوى فصل الكلاستير. |
| PLDA | النتيجة الكلاسيكية | احتمالية LDA· تسجيل النسبة من الاحتمالات فوق التوابل. |
| DER | مقياس الإسهال | معدل خطأ الإغلاق  غياب + إنذار خاطئ + ارتباك. |

## المزيد من القراءة

- [Snyder et al. (2018). X-Vectors: قوية DNN إدخال لتحديد المتحدثين](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) ورقة عميقة
- [(Desplanques et al. (2020) ECAPA-TDNN](https://arxiv.org/abs/2005.07143) الهندسة المعمارية المهيمنة 20202026.
- [(تشن) وآخرون (2022). WavLM: تدريب مسبق على نطاق واسع تحت إشراف ذاتي لمعالجة الكلام الكامل](https://arxiv.org/abs/2110.13900) — SSL العمود الفقري SV و التأجيل
- [برينين وآخرون (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) إعادة التأجيل في الإنتاج + وضع كومة
- [VoxCeleb قائمة النسب (تحدثت في عام 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) الحالي EER التصنيف على بين النماذج.
