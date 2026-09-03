# بناء خط أنابيب مساعد الصوت  مرحلة 6 Capstone

> كل شيء من الدروس 01-11 ، تم خياطة كل شيء معاً. بناء مساعد صوتي يستمع ، ويعبر ، ويتحدث مرة أخرى. في عام 2026 هذه مشكلة هندسية حل ، وليس مشكلة بحثية  ولكن تفاصيل التكامل تقرر ما إذا كانت تسير.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time:** ~120 minutes

## المشكلة

بناء مساعد من نهاية إلى نهاية:

1. يلتقط المدخلات الميكروفونية (16 kHz (مونو)
2. يكتشف بداية/نهاية خطاب المستخدم.
3. ينسخ التدفق
4. يمر النص إلى LLM التي يمكن أن تدعو الأدوات (التوقيت، الطقس، التقويم).
5. التيارات LLM نص إلى TTS.
6. يعيد الصوت إلى المستخدم
7. يتوقف إذا قام المستخدم بتقاطع الرد في منتصفها.

هدف التأخير: أولاً TTS البايت الصوتي في غضون 800 ميس من الانتهاء من المستخدم من كلامه على جهاز كمبيوتر محمول CPU. هدف الجودة: لا كلمات مفقودة، لا ترجمات هالوسينات على الصمت، لا تسرب نسخ الصوت، لا نجاح في الحقن السريع.

## المفهوم

![خط أنابيب المساعد الصوتي: ميكروفون → VAD → STT → LLM+tools → TTS المتحدث](../assets/voice-assistant.svg)

### المكونات السبعة

1. **-التقاط الصوت** ميك → 16 kHz واحد → 20 ms قطع. عادة `sounddevice` في Python أو أصلي AudioUnit/ALSA/WASAPI في الإنتاج
2. **VAD (الدرس 11)** سيلرو VAD @ عتبة 0.5، تقرير الدقيقة 250 ms، صمت التوقف 500 ms. إشارات "بدء" و "نهاية".
3. **التدفق STT (الدرس 4-5).** -تصفيق الشائعات، (باراكيت)TDTأو ديبجرام نوفا-3 (API). Partial + final transcripts.
4. **LLM مع الاتصال بالأدوات.** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. JSON مخطط للأدوات، إشارات التدفق
5. **التدفق TTS (الدرس السابع)** كوكورو-82M (أسرع فتح) أو كارتيسيا سونيك (التجارية). TTS بعد 20 LLM رموز
6. **التشغيل** المتكلم خارج، رمزية للشبكات ذات النطاق النطاق المنخفض.
7. **مدير التوقف** إذا VAD الحرائق خلال TTS التشغيل، توقف التشغيل، إلغاء LLM, إعادة تشغيل STT.

### الوضعين الثلاثة التي ستضربها

1. **كلمة كلمة أولى** VAD يبدأ النبض متأخراً جداً، "هي" المستخدم مفقود، حد البدء عند 0.3، وليس 0.5.
2. **الرد المتوسط يقاطع الارتباك** LLM يستمر في توليد بعد انقطاع المستخدم، ويقول المساعد عن المستخدم. VAD → إلغاءLLM.
3. **الهلوسة الصمت** "تخرج النصائح "شكراً على مشاهدتك على الإطار الصامت VAD-gate.

### 2026 مستويات الإشارة الإنتاجية

| الـ"كثيرة" | التأخير | رخصة | ملاحظات |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | التجارية API | الصناعة المتخلفة 2026 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 سم | معظمها مفتوح | DIY-friendly |
| موشي (مكون كامل) | 200-300 سم | CC-BY 4.0 | نموذج واحد، بنية مختلفة، الدروس 15 |
| Vapi / Retell (مدار) | 300-500 سم | التجارية | أسرع إطلاق؛ محدودة التخصيص |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | خارج الاتصال | مفتوح | الخصوصية / الحافة |

```figure
v4-voice-latency
```

## بناءها

### الخطوة الأولى: التقاط الميكروفون مع التقطيع (مخططات مزيفة)

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### الخطوة الثانية: VAD-gated التقاط المدير

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### الخطوة الثالثة: التدفق STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### الخطوة الرابعة: أداة الدعوة داخل LLM حلقة

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### الخطوة 5: التعامل مع المقاطعة

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## استخدمها

انظر `code/main.py` لتنمية قابلة للتشغيل التي تصل جميع المكونات السبعة مع نماذج القطع، حتى تتمكن من رؤية شكل خط الأنابيب حتى دون أجهزة.

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` أو `openai-whisper`
- `openai` (`gpt-4o`) أو `anthropic`
- `kokoro` أو `cartesia`
- `sounddevice` لـ (I/O)

## الفخاخ

- **التقطيع PII إلى الأبد** الصوت الكامل التحول هو PII في معظم الولايات القضائية، احتفاظ لمدة 30 يوماً، مشفرة في حالة الراحة.
- **لا تدخل** المستخدمون سيقاطعون، يجب أن يتوقف مساعدك عن الحديث
- **TTS هذا يمنع** متزامن TTS يمنع حلقة الحدث. استخدم التزامن أو خيط منفصل.
- **لا يوجد إصابة بالخطأ في الاتصال بالأدوات** الأدوات تفشل LLM يجب أن تعيد الخطأ + محاولة مرة أخرى ، ثم تخفيض gracefully.
- **ملفات الهلوسة المفرطة الحماس** "فلتراً أكثر" ويقول المساعد "لا أستطيع المساعدة" "فلتراً أقل" ويقول أي شيء
- **لا خيار للكلمة الاستيقاظ.** الاستماع دائما هو مسؤولية خصوصية. openWakeWord).

## أرسله

إحتفظ بها `outputs/skill-voice-assistant-architect.md`. بالنظر إلى القيود المفروضة على الميزانية + الحجم + اللغة + الامتثال، قم بإعداد تحديد كامل للمجموعة.

## التمارين

1. **-بسهولة** أركض `code/main.py`إنه يحاكي دور كامل واحد من نهاية إلى نهاية مع وحدات الصفوف والطبعات في كل مرحلة
2. **متوسط** استبدال STT معصم مع نموذج فيسبر الحقيقي على تسجيل مسبق `.wav`- قياس WER و التأخير من نهاية إلى نهاية
3. **صعب** إضافة دعوة الأداة: تنفيذ `get_weather` (أي API) و `set_timer`- إرشاد الطريق LLM عبر الأدوات والتحقق من أنه عندما يقول المستخدم "وضع توقيت 5 دقائق" يتم تشغيل الوظيفة الصحيحة والرد المتكلم يؤكد ذلك.

## الشروط الرئيسية

| المدة | ما يقوله الناس | ما يعنيه هذا في الواقع |
|------|-----------------|-----------------------|
| إلتفت | A user + assistant round-trip | واحد VAD-bounded user speech + one LLM-TTS رد فعل |
| -إختراق | الإقلاع | يتحدث المستخدم بينما يتحدث المساعد، يتوقف المساعد. |
| أيقظوا | "مرحباً مساعدتي" | كلمات مفاتيح قصيرة الكشف، البوركوبين، سنو بوي، openWakeWord. |
| الإشارة النهائية | نهاية الجولة | VAD + min-silence decision that user has finished. |
| المقبل | عازف قبل الكلام | إبق 200-400 ms من الصوت قبل VAD حرائق لتجنب التقاط الكلمة الأولى |
| دعوة الأدوات | دعوة الوظيفة | LLM الإصدارات JSON· إرسال الوقت؛ إرسال النتيجة في الدائرة. |

## المزيد من القراءة

- [LiveKit وكيل الصوت سريع](https://docs.livekit.io/agents/) إشارة إلى مستوى الإنتاج
- [مثال على وكيل صوتي](https://github.com/pipecat-ai/pipecat) — DIY-friendly الإطار
- [OpenAI الوقت الحقيقي API](https://platform.openai.com/docs/guides/realtime) المسار المُدار الصوتي الأصلي
- [كيوتاي موشي](https://github.com/kyutai-labs/moshi) إشارة كاملة (الدرس 15).
- [كلمة استيقظ الخنزير](https://picovoice.ai/products/porcupine/) -إغلاق الكلمات المُستيقظة
- [Anthropic دليل استخدام الأدوات](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM الدعوة الوظيفية
