# التصنيف الصوتي  من k-NN على MFCCs إلى AST و BEATs

> كل شيء من "كلب يلعن ضد السيرين" إلى "أي لغة هي هذه" هو تصنيف الصوت. AUC, F1، و تذكير لكل فئة

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 3 · 06 (CNNs), Phase 5 · 08 (CNNs & RNNs for Text)
**Time:** ~75 minutes

## المشكلة

تحصل على شريط لمدة 10 ثواني. تريد أن تعرف: "ما هو؟" صوت حضري (سيرين، حفرة، كلب) ، أمر الكلام (نعم / لا / توقف) ، اللغة ID (en/es/ar) ، عاطفة المتحدث (غضب / محايد) ، أو صوت بيئي (داخل / خارج المنزل ، البابل). *تصنيف الصوت*، وفي عام 2026 تكون الهندسة المعمارية الأساسية نضج: CNN أو Transformer → softmax

الصعوبة الأساسية ليست الشبكة. إنها البيانات. مجموعة البيانات الصوتية لديها عدم توازن فئات وحشي ، وتحويل مستوى قوي (نظيف مقابل ضجيج) ، وضجيج العلامات التجارية (من قرر "المرحلة الحضرية" مقابل "ضجيج المطعم"؟ 80% من المشكلة هي التركيب والتكبير والتقييم ، وليس التبادل CNN لـ (ترانسفورمير)

## المفهوم

![سلم تصنيف الصوت: k-NN على MFCCs إلى AST إلى BEATs](../assets/audio-classification.svg)

**كـNN على MFCCs (حوالي الأساس في التسعينيات).** مسطح MFCCs لكل مقطع، حساب تشابه كوسين مع البنك الملصق، عودة صوت الأغلبية من أعلى K. قوية بشكل مفاجئ على مجموعة بيانات نظيفة، صغيرة (أوامر الكلام، ESC-50) يدير بدون أي GPU.

**2D CNN بشأن المياه المزروعة (2015-2019).** معالجة `(T, n_mels)` التطبيق على الرسائل التسجيلية كصورة ResNet-18 أو VGG-style. متوسط العالم يجمع محور الوقت، و يُساعد على التدريب على المرحلة، ما زال قائماً في معظم مسابقات كاغل عام 2026.

**محول الطيف الصوتي AST (2021-2024).** إصبع الملفات (مثل 16 × 16 ملصقات) ، إضافة إضافة وضعيات، إطعام إلى ViT. حالة الفن AudioSet (mAP 0.485) للتعلم المراقب.

**BEATs و WavLM-base (2024-2026).** التدريب المباشر على الذات على ملايين الساعات. ضبط مهمتك مع 1-10% من البيانات المراقبة التي كنت ستحتاجها. في عام 2026 هذه هي نقطة البداية الافتراضية للصوت غير الكلام. BEATs-iter3 النبضات AST من 1 إلى 2 mAP على AudioSet بينما تستخدم 1/4 الحساب.

**كشف فسخ كعمود الفقري المجمد (2024).** خذ جهاز "ويسبر" للتشفير، اترك جهاز التشفير، وربط مصنف خطيSOTA عن اللغة ID و تصنيف حدث بسيط مع زيادة الصوت صفر. "غداء مجاني"

### عدم توازن الطبقات هو التحدي الحقيقي

ESC-50: 50 درجة، 40 شريطا كل  متوازنة، سهلة. UrbanSound8K: 10 فئات، عدم التوازن 10:1. AudioSet632 فئة مع ذيل طويل 100,000: 1 تقنيات تعمل:

- أخذ العينات المتوازن أثناء التدريب (ليس في التقييم).
- الاختلاط: التقاطع خطيا بين شريطين (وتصريحاتهم) كإضاف.
- SpecAugment: قناع وقت عشوائي ومجموعات تردد بسيطة، حرجة.

### التقييم

- حصرية متعددة الفئات (أوامر الكلام): دقة من أعلى 1، دقة من أعلى 5.
- (معدل الفئات متعددة العلامات)AudioSet, UrbanSound-style): متوسط دقة (mAP).
- عدم التوازن الكبير: استدعاء لكل فئة + كلية F1.

أرقام 2026 يجب أن تعرفها:

| المرجعية | خط الأساس | SOTA 2026 | المصدر |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs ورق (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR قائمة النسب 2026 |
| أوامر الكلام v2 | 98% (CNN) | 99.0% (صوتيMAE) | HEAR v2 النتائج |

```figure
mfcc-pipeline
```

## بناءها

### الخطوة 1: التميز

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### الخطوة الثانية: ملخص طول ثابت

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

بسيط ولكن قوي: المتوسط + التباين عبر الزمن يعطي إضافة ثابتة 26 طولة لـ 13 كوف MFCC. يُجري على الفور، يُغلب على أحدث التطورات NN خطوط أساسية على ESC-50 في وقت قريب من عام 2017.

### الخطوة الثالثة:NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### الخطوة الرابعة: تحديث إلى CNN على المياه

في PyTorch:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

ملامح 3 م. القطارات في ~ 10 دقيقة على ESC-50 مع واحدة RTX 4090، دقة 80٪ +.

### الخطوة 5: التأثير الروحي الروحي 2026 BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

ل: BEATs، استخدام `microsoft/BEATs-base` عبر `beats` المكتبة، المحولات API هو نفس الشكل.

## استخدمها

"مجموعة 2026"

| الوضع | ابدأ |
|-----------|-----------|
| Tiny dataset (<1000 clips) | كـNN على MFCC means (your baseline) + audio augmentation |
| مجموعة بيانات متوسطة (1K100K) | BEATs أو AST المزق |
| Large dataset (>100K) | القطار من الصفر أو التنسيق الدقيق |
| في الوقت الحقيقي، الحافة | 40-MFCC CNN، مقياسياً إلى int8 (KWS-style) |
| (متعددة العلامات)AudioSet) | BEATs-iter3 مع BCE loss + mixup + SpecAugment |
| اللغة ID | MMS-LID, SpeechBrain VoxLingua107 خط الأساس |

قاعدة القرار: **تبدأ مع العمود الفقري المجمد، وليس نموذج جديد**. تحسين BEATs رأسك يحصل لك 95٪ من SOTA في ساعات وليس أسابيع

## أرسله

إحتفظ بها `outputs/skill-classifier-designer.md`. اختيار الهندسة المعمارية، وتعزيزات، استراتيجية توازن الطبقات، وتقييم المقاييس لمهمة تصنيف الصوت معينة.

## التمارين

1. **-بسهولة** أركض `code/main.py`إنه يدرب الك...NN MFCC خط أساسي على مجموعة بيانات اصطناعية من 4 فئات (أصوات نقية في مستويات مختلفة).
2. **متوسط** استبدال `summarize` مع [متوسط، var، منحرف، كورتوس] هل 4 اللحظات تجمع ضرب المتوسط + var على نفس مجموعة البيانات الاصطناعية؟
3. **صعب** استخدام `torchaudio`، تدريب 2D CNN على ESC-50 المثبتة 1 تقرير 5 مرات دقة التحقق المتقاطع. إضافة SpecAugment (الوقت mask = 20, freq mask = 10) و أبلغ عن الدلتا

## الشروط الرئيسية

| المدة | ما يقوله الناس | ما يعنيه هذا في الواقع |
|------|-----------------|-----------------------|
| AudioSet | المُسَمِع ImageNet الصوت | شريط جوجل 2M، درجة 632 ضعيفة YouTube مجموعة بيانات |
| ESC-50 | مقياس تصنيف صغير | 50 فئة × 40 شريط من الأصوات البيئية. |
| AST | محول للكتابات الصوتية | ViT على ملصقات الملفات المطبوعة في الملفات المطبوعة؛ 2021 SOTA. |
| BEATs | صوتي مرصد ذاتي | نموذج مايكروسوفت iter3 الإتجاهات AudioSet اعتبارا من عام 2026 |
| المزيج | زيادة الأزواج | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| SpecAugment | التوسع القائم على القناع | فصيلة وقت عشوائية و ترددات الطيف |
| mAP | المقياسات الرئيسية متعددة العلامات | متوسط دقة بين الفئات والعدوان |

## المزيد من القراءة

- [غونغ، تشونغ، جلاس (2021). AST: محول للكتابات الصوتية](https://arxiv.org/abs/2104.01778) الهندسة المعمارية السجلية من 20212024.
- [تشين وغيرها (2022، rev. 2024). BEATs: صوتي التدريب المسبق مع الـ "توكينيزر" الصوتي](https://arxiv.org/abs/2212.09058) الاكتفاء في 2024+
- [(بارك وغيره) (2019). SpecAugment](https://arxiv.org/abs/1904.08779) التوسع السمعي المهيمن.
- [بيكزاك (2015) ESC-50 مجموعة بيانات](https://github.com/karolpiczak/ESC-50) مقياس 50 فئة التي تعيش على.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) فئة 632 YouTube التخسيس، لا يزال المعيار الذهبي.
