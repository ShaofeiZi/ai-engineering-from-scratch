# अनुक्रम से अनुक्रम मॉडल

> दो RNNs वे एक अनुवादक होने का नाटक करते हैं।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (PyTorch Intro)
**Time:** ~75 minutes

## समस्या

वर्गीकरण एक एकल लेबल पर एक चर लंबाई अनुक्रम का नक्शा बनाता है। अनुवाद एक चर लंबाई अनुक्रम को एक अन्य चर लंबाई अनुक्रम पर नक्शा बनाता है। इनपुट और आउटपुट अलग-अलग शब्दावली में रहते हैं, संभवतः अलग-अलग भाषाओं में, लंबाई समानता की कोई गारंटी नहीं है।

इन seq2seq वास्तुकला (Sutskever, Vinyals, Le, 2014) ने जानबूझकर सरल नुस्खा के साथ इसे क्रैक किया। RNNs. एक स्रोत वाक्य पढ़ता है और एक निश्चित आकार का संदर्भ वेक्टर उत्पन्न करता है। दूसरा उस वेक्टर को पढ़ता है और लक्ष्य वाक्य टोकन को टोकन द्वारा उत्पन्न करता है। उसी कोड को आपने पाठ 08 के लिए लिखा है, अलग तरह से चिपकाया गया।

यह दो कारणों से अध्ययन करने लायक है। पहला, संदर्भ-वेक्टर बोतल की गर्दन शिक्षा के लिए सबसे उपयोगी विफलता है NLP. यह ध्यान और ट्रांसफार्मर में अच्छे सभी चीजों को प्रेरित करता है। दूसरा, प्रशिक्षण नुस्खा (शिक्षक मजबूर, अनुसूचित नमूने लेने, निष्कर्ष पर बीम खोज) अभी भी प्रत्येक आधुनिक पीढ़ी प्रणाली पर लागू होता है जिसमें LLMs.

## अवधारणा

**एन्कोडर।** एक RNN जो स्रोत वाक्य पढ़ता है. इसकी अंतिम छिपी हुई स्थिति है **संदर्भ वेक्टर** पूरे इनपुट का एक निश्चित आकार का सारांश। स्रोत के अलावा कुछ भी नहीं खोना, माना जाता है।

**डिकोडर.** एक और RNN संदर्भ वेक्टर से शुरू किया गया है। प्रत्येक चरण में यह पहले उत्पन्न टोकन को इनपुट के रूप में लेता है और लक्ष्य शब्दावली पर एक वितरण का उत्पादन करता है। अगले टोकन को चुनने के लिए नमूना या argmax। इसे वापस फ़ीड करें। एक तक दोहराएं `<EOS>` टोकन उत्पन्न किया जाता है या अधिकतम लंबाई मारा जाता है।

**प्रशिक्षणः** प्रत्येक डिकोडर चरण में क्रॉस-एंट्रोपी हानि, क्रम में योग। दोनों नेटवर्क के माध्यम से समय के माध्यम से मानक बैकपॉड।

**शिक्षक मजबूर कर रहा है।** प्रशिक्षण के दौरान, डेकोडर का इनपुट चरण `t` है *मूल सत्य* स्थिति पर टोकन `t-1`, डिकोडर की अपनी पिछली भविष्यवाणी नहीं है. यह प्रशिक्षण को स्थिर करता है; इसके बिना, प्रारंभिक त्रुटियां कैस्केड होती हैं और मॉडल कभी नहीं सीखता है। निष्कर्ष पर, आपको मॉडल की अपनी भविष्यवाणी का उपयोग करना होगा, इसलिए हमेशा एक ट्रेन / इन्फेरेंस वितरण अंतर होता है। उस अंतर को कहा जाता है **जोखिम पूर्वाग्रह**.

**बोतल की गर्दन.** कोडर को स्रोत के बारे में जो कुछ भी सीखा है उसे उस संदर्भ वेक्टर में दबाया जाना चाहिए। लंबे वाक्य विवरण खो देते हैं। दुर्लभ शब्द धुंधला हो जाते हैं। पुनर्गठन (चैट नोअर बनाम ब्लैक कैट) को याद रखना चाहिए, गणना नहीं।

ध्यान (पाठ 10) यह ठीक करता है, यह डिकोडर को देखने के लिए *हर* एन्कोडर छिपे राज्य, केवल पिछले एक नहीं है. यह पूरी पिच है.

```figure
lstm-gates
```

## इसे बनाओ

### चरण 1: एक एन्कोडर

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` आकार है `[batch, seq_len, hidden_dim]` प्रत्येक इनपुट स्थिति के लिए एक छिपा हुआ राज्य। `hidden` आकार है `[1, batch, hidden_dim]` अंतिम चरण। पाठ 08 में कहा गया था "वर्गीकरण के लिए आउटपुट परpool।" यहाँ हम अंतिम छिपे हुए राज्य को संदर्भ वेक्टर के रूप में रखते हैं, और प्रति चरण आउटपुट को अनदेखा करते हैं।

### चरण 2: एक डिकोडर

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

इनपुटः एकल टोकन का एक बैच और वर्तमान छिपी हुई स्थिति। आउटपुटः अगले टोकन और अद्यतन छिपी हुई स्थिति के लिए शब्दावली लॉगिंग।

### चरण 3: शिक्षक द्वारा मजबूर किए जाने वाले प्रशिक्षण लूप

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

नाम देने लायक दो बटन। `ignore_index=0` पैडिंग टोकन पर नुकसान छोड़ देता है। `teacher_forcing_ratio` प्रत्येक चरण में मॉडल की भविष्यवाणी के खिलाफ वास्तविक टोकन का उपयोग करने की संभावना है। एक्सपोज़र-bias अंतर को बंद करने के लिए 1.0 (पूर्ण शिक्षक मजबूर) से शुरू करें और प्रशिक्षण के माध्यम से ~0.5 तक नीचे बढ़ें।

### चरण 4: निष्कर्ष लूप (लाभकारी)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

लालची डिकोडिंग हर कदम पर सबसे अधिक संभावना वाले टोकन को चुनती है। यह भटक सकता हैः एक बार जब आप एक टोकन के लिए प्रतिबद्ध होते हैं, तो आप इसे अनइंस्टॉल नहीं कर सकते। **बीम खोज** शीर्ष को बनाए रखता है-`k` आंशिक अनुक्रमों को जीवित और सबसे अधिक स्कोरिंग पूर्ण एक को चुनता है अंत में. बीम चौड़ाई 3-5 मानक है।

### चरण 5: बोतल की गर्दन, दिखाया गया

मॉडल को खिलौना कॉपी करने के लिए प्रशिक्षित करेंः स्रोत `[a, b, c, d, e]`, लक्ष्य `[a, b, c, d, e]`अनुक्रम की लंबाई बढ़ाएँ। सटीकता का निरीक्षण करें।

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

एकल GRU गुप्त राज्य 40 टोकन इनपुट याद रखने के लिए खोने के बिना नहीं कर सकते. जानकारी वहाँ है प्रत्येक एन्कोडर चरण पर, लेकिन डिकोडर केवल अंतिम राज्य देखता है. ध्यान सीधे इसे ठीक करता है.

## इसका प्रयोग करें

PyTorch है `nn.Transformer` और `nn.LSTM`-आधारित seq2seq टेम्पलेट्स. गले लगाना चेहरा `transformers` पुस्तकालय जहाजों पूर्ण एन्कोडर-डेकोडर मॉडल (BART, T5, mBART, NLLB) को अरबों टोकन पर प्रशिक्षित किया गया है।

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

आधुनिक एन्कोडर-डेकोडर गिर गए RNNs उच्च स्तर का आकार (एन्कोडर, डिकोडर, जेनरेट टोकन-दर-टोकन) 2014 के समान है seq2seq प्रत्येक ब्लॉक के अंदर तंत्र अलग है।

### कब तक पहुंचना है RNN-based seq2seq

नए परियोजनाओं के लिए लगभग कभी नहीं।

- स्ट्रीमिंग अनुवाद जहां आप सीमित स्मृति के साथ एक समय में एक टोकन इनपुट का उपभोग करते हैं।
- डिवाइस पर पाठ उत्पन्न करना जहां ट्रांसफार्मर मेमोरी की लागत निषेधात्मक है।
- शैक्षणिक. एन्कोडर-डेकोडर की गड़बड़ी को समझना यह समझने का सबसे तेज़ तरीका है कि ट्रांसफार्मर क्यों जीते हैं।

### जोखिम पूर्वाग्रह और इसके उन्मूलन

- **निर्धारित नमूनाकरण।** प्रशिक्षण के दौरान शिक्षक बल अनुपात के साथ-साथ, मॉडल अपनी गलतियों से उबरना सीखता है।
- **न्यूनतम जोखिम प्रशिक्षण।** वाक्य स्तर पर ट्रेन BLEU टोकन स्तर के क्रॉस-एंट्रोपी के बजाय स्कोर. आप वास्तव में क्या चाहते हैं के करीब.
- **सुदृढीकरण सीखने की बारीक-टीप।** एक मीट्रिक के साथ अनुक्रम जनरेटर इनाम. LLM RLHF.

तीनों ही अभी भी ट्रांसफार्मर आधारित पीढ़ी पर लागू होते हैं।

## इसे भेजें

के रूप में सहेजें `outputs/prompt-seq2seq-design.md`:

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## व्यायाम

1. **- आराम से।** खिलौना कॉपी करने का कार्य करें। GRU seq2seq इनपुट-आउटपुट जोड़े पर जहां लक्ष्य स्रोत के बराबर है। लंबाई 5, 10, 20 पर सटीकता मापें। बोतल गला को पुनः प्रस्तुत करें।
2. **मध्यम।** बीम की चौड़ाई के साथ बीम खोज डिकोडिंग जोड़ें 3. मापें BLEU लालच के खिलाफ एक छोटे समानांतर corpus पर। दस्तावेज जहां बीम खोज जीत (आमतौर पर अंतिम टोकन) और जहां यह कोई फर्क नहीं पड़ता।
3. **कठिन.** ठीक-ठीक `facebook/bart-base` 10k जोड़ी पैराफ्रेस डेटासेट पर। बारीक-ट्यून मॉडल के बीम-4 आउटपुट की तुलना आधार मॉडल पर रखा इनपुट पर. रिपोर्ट BLEU और 10 गुणात्मक उदाहरण चुनें।

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|-----------------|-----------------------|
| एन्कोडर | इनपुट RNN | स्रोत पढ़ता है. प्रति चरण छिपे हुए राज्यों और एक अंतिम संदर्भ वेक्टर उत्पन्न करता है. |
| डिसीडर | आउटपुट RNN | संदर्भ वेक्टर से शुरू. लक्ष्य टोकन एक समय में उत्पन्न करता है. |
| संदर्भ वेक्टर | सारांश | अंतिम एन्कोडर छिपा हुआ राज्य, फिक्स्ड आकार, बोतल गला ध्यान हल करता है। |
| शिक्षक जबरन | वास्तविक टोकन का उपयोग करें | प्रशिक्षण के समय मूल सत्य के पहले टोकन को खिलाएं। सीखने को स्थिर करता है। |
| जोखिम पूर्वाग्रह | ट्रेन/परीक्षण अंतर | सच्चे टोकन पर प्रशिक्षित मॉडल ने कभी अपनी गलतियों से उबरने का अभ्यास नहीं किया। |
| बीम खोज | बेहतर डिकोडिंग | हर कदम पर शीर्ष-के आंशिक अनुक्रमों को जीवित रखें लालची से प्रतिबद्ध करने के बजाय। |

## आगे पढ़ना

- [सुट्सकेवर, विनाइल, ले (2014) । तंत्रिका नेटवर्क के साथ अनुक्रम से अनुक्रम सीखने](https://arxiv.org/abs/1409.3215) मूल seq2seq चार पृष्ठ।
- [Cho et al. (2014) । प्रयोग करके वाक्यांश प्रतिनिधित्व सीखना RNN सांख्यिकीय मशीन अनुवाद के लिए एन्कोडर-डेकोडर](https://arxiv.org/abs/1406.1078)  GRU और एन्कोडर-डेकोडर फ्रेमिंग।
- [बहदानू, चो, बेंगियो (2014) । संरेखण और अनुवाद करने के लिए संयुक्त रूप से सीखने के द्वारा तंत्रिका मशीन अनुवाद](https://arxiv.org/abs/1409.0473) ध्यान पत्र. इस पाठ के तुरंत बाद पढ़ें।
- [PyTorch NLP स्क्रैच ट्यूटोरियल से](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) निर्माण योग्य seq2seq + attention code.
