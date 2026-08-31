# एक आवाज सहायक पाइपलाइन बनाएं  चरण 6 कैपस्टोन

> सब कुछ सब कुछ से कक्षा 01-11, एक साथ सिलाई। एक आवाज सहायक का निर्माण जो सुनता है, तर्क देता है, और बात करता है। 2026 में यह एक समाधान इंजीनियरिंग समस्या है, एक अनुसंधान समस्या नहीं है  लेकिन एकीकरण विवरण तय करते हैं कि यह जहाज है या नहीं।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time:** ~120 minutes

## समस्या

एक अंत-से-अंत सहायक बनाएंः

1. माइक्रो इनपुट कैप्चर करता है (16 kHz मोनो) ।
2. उपयोगकर्ता भाषण की शुरुआत/अंत का पता लगाता है।
3. स्ट्रीमिंग का अनुवाद करता है।
4. एक पर ट्रांसक्रिप्ट पारित करता है LLM जो उपकरण (टाइमर, मौसम, कैलेंडर) बुला सकता है।
5. धाराएं LLM एक को पाठ TTS.
6. उपयोगकर्ता को ऑडियो वापस चलाता है।
7. यदि उपयोगकर्ता मध्य-उत्तर में बाधित करता है तो रोकता है।

विलंबता लक्ष्य: प्रथम TTS उपयोगकर्ता द्वारा लैपटॉप पर अपना भाषण समाप्त करने के 800 एमएस के भीतर ऑडियो बाइट CPU. गुणवत्ता लक्ष्य: कोई याद किए गए शब्द, कोई भ्रमपूर्ण उपशीर्षक चुपचाप, कोई आवाज क्लोनिंग लीक, कोई त्वरित इंजेक्शन सफलता नहीं।

## अवधारणा

![आवाज सहायक पाइपलाइनः माइक्रो → VAD → STT → LLM+tools → TTS → वक्ता](../assets/voice-assistant.svg)

### सात घटक

1. **ऑडियो कैप्चर.** माइक → 16 kHz mono → 20 ms टुकड़े। आमतौर पर `sounddevice` में Python या स्वदेशी AudioUnit/ALSA/WASAPI उत्पादन में।
2. **VAD (पाठ 11)** सिलेरो VAD @ threshold 0.5, मिन भाषण 250 एमएस, मौन लटकन 500 एमएस. संकेत "शुरू" और "अंत"
3. **स्ट्रीमिंग STT (पाठ ४-५)** चुप्पी प्रवाह, परकीट-TDT, या डीपग्राम नोवा-3 (API). Partial + final transcripts.
4. **LLM उपकरण कॉल के साथ।** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. JSON उपकरण के लिए योजना. स्ट्रीम टोकन.
5. **स्ट्रीमिंग TTS (पाठ 7)** कोकोरो-82एम (सबसे तेज़ खोलने) या कार्टेशिया सोनिक (व्यावसायिक) शुरू करें TTS 20 के बाद LLM टोकन.
6. **प्लेबैक.** स्पीकर आउट; कम बैंडविड्थ नेटवर्क के लिए अपस-कोड।
7. **विराम के प्रबंधक.** यदि VAD आग के दौरान TTS प्लेबैक, प्लेबैक बंद, रद्द LLM, पुनः आरंभ STT.

### तीन विफलता मोड आप हिट करेंगे

1. **पहले शब्द क्लिप.** VAD एक धड़कन बहुत देर से शुरू होता है. उपयोगकर्ता का "हे" गायब है. 0.3 पर शुरू करने की सीमा, 0.5 नहीं.
2. **मध्य प्रतिक्रिया भ्रम को बाधित करती है।** LLM उपयोगकर्ता के अंतराल के बाद उत्पन्न करता रहता है; सहायक उपयोगकर्ता के ऊपर बात करता है। तार VAD → रद्द-LLM.
3. **मौन भ्रम।** चुपके वार्मअप फ्रेम पर "देखने के लिए धन्यवाद" चुप्पी आउटपुट। हमेशा VAD-gate.

### 2026 उत्पादन संदर्भ स्टैक

| स्टैक | विलंबता | लाइसेंस | नोट्स |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | वाणिज्यिक API | उद्योग डिफ़ॉल्ट 2026 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | अधिकतर खुला | DIY-friendly |
| मोशी (पूर्ण-डप्लक्स) | 200-300 ms | CC-BY 4.0 | एकल मॉडल; विभिन्न वास्तुकला, पाठ 15 |
| Vapi / रिटेल (प्रबंधित) | 300-500 ms | वाणिज्यिक | लॉन्च करने के लिए सबसे तेज; सीमित अनुकूलन |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | ऑफ़लाइन | खुला | गोपनीयता / बढ़त |

```figure
v4-voice-latency
```

## इसे बनाओ

### चरण 1: चश्मा (पस्यूडोकोड) के साथ माइक्रोफ़ोन कैप्चर

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

### चरण 2: VAD-gated घुमाव पकड़े

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

### चरण 3: स्ट्रीमिंग STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### चरण 4: उपकरण अंदर कॉल LLM लूप

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

### चरण 5: विराम का संचालन

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

## इसका प्रयोग करें

देखिये `code/main.py` एक चलाने योग्य सिमुलेशन के लिए जो सभी सात घटकों को स्टब मॉडल के साथ तार करता है, ताकि आप हार्डवेयर के बिना भी पाइपलाइन आकार देख सकें। एक वास्तविक कार्यान्वयन के लिए, स्टब को स्विच करेंः

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` या `openai-whisper`
- `openai` (`gpt-4o`) या `anthropic`
- `kokoro` या `cartesia`
- `sounddevice` I/O के लिए

## फंदे

- **लकड़ी का उत्पादन PII हमेशा के लिए.** पूर्ण-टर्न ऑडियो है PII अधिकांश न्यायालयों में 30 दिनों के लिए भंडारण, आराम में एन्क्रिप्टेड।
- **कोई घुसपैठ नहीं.** उपयोगकर्ता बाधित करेंगे. आपके सहायक को बोलना बंद करना चाहिए.
- **TTS यह ब्लॉक करता है।** समकालिक TTS घटना लूप को अवरुद्ध करता है. असिनक्रोनस या एक अलग धागा का उपयोग करें.
- **कोई उपकरण कॉल त्रुटि हैंडलिंग नहीं।** उपकरण विफल हो जाते हैं। LLM त्रुटि + एक बार पुनः प्रयास करना चाहिए, फिर gracefully degraded.
- **अति उत्साही पगडंडी फिल्टर।** ओवर-फिल्टर और सहायक दोहराता है "मैं इसके साथ मदद नहीं कर सकता है।" नीचे-फिल्टर और यह कुछ भी कहता है. एक पकड़ सेट पर माप.
- **कोई चेतावनी नहीं है।** हमेशा सुनना एक गोपनीयता दायित्व है. एक जागृति शब्द गेट (पोर्किपिन या openWakeWord).

## इसे भेजें

के रूप में सहेजें `outputs/skill-voice-assistant-architect.md`. बजट + पैमाने + भाषा + अनुपालन संबंधी बाधाओं को देखते हुए, एक पूर्ण स्टैक विनिर्देश तैयार करें।

## व्यायाम

1. **- आराम से।** दौड़ें `code/main.py`यह स्टब मॉड्यूल और प्रति चरण विलंबता के साथ एक पूर्ण बारी अंत-से-अंत अनुकरण करता है।
2. **मध्यम।** प्रतिस्थापन STT पूर्व-रिकॉर्ड पर एक असली विस्पर मॉडल के साथ स्टब `.wav`. माप WER और अंत-से-अंत विलंबता.
3. **कठिन.** उपकरण कॉल जोड़ेंः लागू करें `get_weather` (किसी भी API) तथा `set_timer`. मार्ग LLM उपकरण के माध्यम से और सत्यापित करें कि जब उपयोगकर्ता कहता है "5 मिनट टाइमर सेट करें" सही फ़ंक्शन फायर करता है और बोली प्रतिक्रिया इसकी पुष्टि करती है।

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| घुमाओ | A user + assistant round-trip | एक VAD-bounded user speech + one LLM-TTS प्रतिक्रिया। |
| नौकायन | विराम | उपयोगकर्ता बोलता है जबकि सहायक बोलता है; सहायक रुकता है। |
| जागने की खबर | "हे सहायक" | लघु कीवर्ड डिटेक्टर; पोर्किपिन, स्नोबॉय, openWakeWord. |
| अंत की ओर संकेत | मोड़ समाप्त | VAD + min-silence decision that user has finished. |
| पूर्व-रोल | भाषण से पहले बफर | पहले 200-400 एमएस ऑडियो रखें VAD पहले शब्द क्लिप से बचने के लिए आग लगाना। |
| उपकरण कॉल | फ़ंक्शन का आह्वान | LLM उत्सर्जन JSON; रनटाइम डिस्पैच; परिणाम लूप में वापस फीड करता है। |

## आगे पढ़ना

- [LiveKit आवाज एजेंट त्वरित प्रारंभ](https://docs.livekit.io/agents/) उत्पादन स्तर का संदर्भ।
- [पिकपेट  आवाज एजेंट उदाहरण](https://github.com/pipecat-ai/pipecat) — DIY-friendly ढांचा।
- [OpenAI वास्तविक समय API](https://platform.openai.com/docs/guides/realtime) प्रबंधित आवाज-मौलिक पथ।
- [क्युताई मोशी](https://github.com/kyutai-labs/moshi) पूर्ण डुप्लेक्स संदर्भ (पाठ 15)
- [सुअर की जागृति शब्द](https://picovoice.ai/products/porcupine/) जागने के शब्द गेटिंग।
- [Anthropic उपकरण उपयोग गाइड](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM कार्य कॉल।
