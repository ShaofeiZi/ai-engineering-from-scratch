# GPU सेटअप और क्लाउड

> प्रशिक्षण CPU वास्तविक प्रशिक्षण के लिए एक GPU.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## सीखने के लक्ष्य

- स्थानीय सत्यापित करें GPU उपयोग करके उपलब्धता `nvidia-smi` और PyTorchहै CUDA API
- Google Colab को एक T4 GPU निःशुल्क क्लाउड आधारित प्रयोगों के लिए
- बेंचमार्क मैट्रिक्स गुणन पर CPU vs GPU और गति को मापें
- अपने आकार में सबसे बड़ा मॉडल का अनुमान लगाएं VRAM उपयोग करके fp16 अंगूठे का नियम

## समस्या

चरण 1-3 में अधिकांश पाठ ठीक से चलते हैं CPU. लेकिन एक बार जब आप प्रशिक्षण शुरू CNNs, ट्रांसफार्मर, या LLMs (phases 4+), you need GPU एक प्रशिक्षण दौड़ जो 8 घंटे तक चलती है CPU 10 मिनट लगते हैं GPU.

आपके पास तीन विकल्प हैंः स्थानीय GPU, बादल GPU, या गूगल कोलाब (मुक्त) ।

## अवधारणा

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

```figure
s0-gpu-dispatch
```

## इसे बनाओ

### विकल्प 1: स्थानीय NVIDIA GPU

जाँचें कि क्या आपके पास एक हैः

```bash
nvidia-smi
```

स्थापित करें PyTorch के साथ CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### विकल्प 2: गूगल कोलाब

1. जाओ [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change चलना समय type > T4 GPU
3. दौड़ें `!nvidia-smi` सत्यापित करने के लिए

इस कोर्स से नोटबुक सीधे कोलाब में अपलोड करें।

### विकल्प 3: बादल GPU

लैम्ब्डा लैब्स के लिए, RunPod, या Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### नहीं GPUकोई समस्या नहीं है।

अधिकांश पाठों पर काम करता है CPU. जो लोग जरूरत है GPU यह कहेंगे और कोलाब लिंक शामिल करेंगे।

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## इसे बनाओः GPU vs CPU बेंचमार्क

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## व्यायाम

1. ऊपर बेंचमार्क चलाएँ और तुलना करें CPU vs GPU समय
2. यदि आपके पास एक नहीं है GPU, गूगल कोलाब पर चलाओ और तुलना
3. कितना जाँचें GPU स्मृति आप है और अनुमान है कि सबसे बड़ा मॉडल आप फिट कर सकते हैं (आंगूठे के नियमः 2 बाइट प्रति पैरामीटर के लिए fp16)

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|----------------|----------------------|
| CUDA | "GPU कार्यक्रम" | NVIDIAसमानांतर कंप्यूटिंग मंच है कि आप कोड पर चलाने की अनुमति देता है GPU |
| VRAM | "GPU स्मृति" | वीडियो RAM पर GPU, प्रणाली से अलग RAM. मॉडल आकार को सीमित करता है। |
| fp16 | "आधे सटीकता" | 16-बिट तैरने बिंदु, का आधा स्मृति का उपयोग करता है fp32 न्यूनतम सटीकता हानि के साथ |
| टेन्सर कोर | "फास्ट मैट्रिक्स हार्डवेयर" | विशेष GPU मैट्रिक्स गुणन के लिए कोर, नियमित कोर की तुलना में 4-8 गुना तेज |
