# अर्थशास्त्र खंडन  यू-नेट

> U-Net इसे एक डाउनसैम्पलिंग एन्कोडर को अपसैम्पलिंग डिकोडर के साथ जोड़े और उनके बीच के कनेक्शन को सिंक करने से काम करता है।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 4 Lesson 04 (Image Classification)
**Time:** ~75 minutes

## सीखने के लक्ष्य

- अर्थशास्त्र, उदाहरण और पैनप्टिक विभाजन में अंतर करें और किसी दिए गए समस्या के लिए सही कार्य चुनें
- एक यू-नेट को खरोंच से बनाएं PyTorch एन्कोडर ब्लॉक, एक बोतल गला, एक ट्रांसपोस्टेड घुमावदार के साथ एक डिकोडर, और स्kip कनेक्शन
- पिक्सेल-वार क्रॉस-एंट्रोपी, डैस हानि, और संयुक्त हानि को लागू करें जो चिकित्सा और औद्योगिक विभाजन के लिए वर्तमान डिफ़ॉल्ट है
- पढ़िए IoU और प्रति वर्ग के लिए डैस मीट्रिक और निदान करें कि क्या एक बुरा स्कोर छोटे वस्तुओं को याद करने, सीमा सटीकता, या वर्ग असंतुलन से आता है

## समस्या

वर्गीकरण प्रति छवि एक लेबल आउटपुट करता है। पता लगाने प्रति छवि कुछ बॉक्स आउटपुट करता है। विभाजन प्रति पिक्सेल एक लेबल आउटपुट करता है। आकार के इनपुट के लिए `H x W`, आउटपुट आकार का एक tensor है `H x W` (सार्थक) या `H x W x N_instances` (उदाहरण) यह प्रति छवि लाखों भविष्यवाणियां हैं, एक नहीं।

विभाजन की संरचना इस कारण से लगभग हर घने भविष्यवाणी दृष्टि उत्पाद को संचालित करती हैः चिकित्सा इमेजिंग (ट्यूमर मास्क), स्वायत्त ड्राइविंग (सड़क, लेन, बाधा), उपग्रह (निर्माण पदचिह्न, फसल सीमाएं), दस्तावेज़ विश्लेषण (लेआउट जोन), रोबोटिक्स (पकड़ने योग्य क्षेत्र) । उन कार्यों में से कोई भी वस्तु के चारों ओर एक बॉक्स डालकर हल नहीं किया जा सकता है; उन्हें सटीक प्रतिरूप की आवश्यकता है।

वास्तुकला की समस्या को कहना आसान है और हल करना आसान नहीं हैः आपको एक छवि के वैश्विक संदर्भ (यह किस तरह का दृश्य है) और स्थानीय पिक्सेल विवरण (सही तरह से कौन सा पिक्सेल सड़क बनाम गलियारा है) को एक साथ देखने के लिए नेटवर्क की आवश्यकता है। CNN संदर्भ प्राप्त करने के लिए स्थानिक रूप से संपीड़ित, जो विवरण फेंक देता है. यू-नेट डिजाइन था कि दोनों मिला.

## अवधारणा

### अर्थिक बनाम उदाहरण बनाम पैनप्टिक

```mermaid
flowchart LR
    IN["Input image"] --> SEM["Semantic<br/>(pixel → class)"]
    IN --> INS["Instance<br/>(pixel → object id,<br/>only foreground classes)"]
    IN --> PAN["Panoptic<br/>(every pixel → class + id)"]

    style SEM fill:#dbeafe,stroke:#2563eb
    style INS fill:#fef3c7,stroke:#d97706
    style PAN fill:#dcfce7,stroke:#16a34a
```

- **अर्थशास्त्र** यह कहता है "यह पिक्सेल सड़क है, वह पिक्सेल कार है". दो कारें एक दूसरे के बगल में एक ब्लाब में गिर जाते हैं।
- **प्रथा** यह कहता है "यह पिक्सेल कार # 3 है, यह पिक्सेल कार # 5 है। " पृष्ठभूमि सामग्री ("सामग्री" = आकाश, सड़क, घास) को अनदेखा करता है।
- **पैनोप्टिक** दोनों को एक करता हैः प्रत्येक पिक्सेल एक वर्ग लेबल मिलता है, प्रत्येक उदाहरण एक अद्वितीय आईडी मिलता है, सामान और सामान दोनों खंडित.

इस पाठ में अर्थशास्त्र को शामिल किया गया है। अगला पाठ (मास्क आर-CNN) उदाहरण को कवर करता है।

### यू-नेट का आकार

```mermaid
flowchart LR
    subgraph ENC["Encoder (contracting)"]
        E1["64<br/>H x W"] --> E2["128<br/>H/2 x W/2"]
        E2 --> E3["256<br/>H/4 x W/4"]
        E3 --> E4["512<br/>H/8 x W/8"]
    end
    subgraph BOT["Bottleneck"]
        B1["1024<br/>H/16 x W/16"]
    end
    subgraph DEC["Decoder (expanding)"]
        D4["512<br/>H/8 x W/8"] --> D3["256<br/>H/4 x W/4"]
        D3 --> D2["128<br/>H/2 x W/2"]
        D2 --> D1["64<br/>H x W"]
    end
    E4 --> B1 --> D4
    E1 -. skip .-> D1
    E2 -. skip .-> D2
    E3 -. skip .-> D3
    E4 -. skip .-> D4
    D1 --> OUT["1x1 conv<br/>classes"]

    style ENC fill:#dbeafe,stroke:#2563eb
    style BOT fill:#fef3c7,stroke:#d97706
    style DEC fill:#dcfce7,stroke:#16a34a
```

एन्कोडर चार बार स्थानिक रिज़ॉल्यूशन को आधा करता है और चैनल को दोगुना करता है। डिकोडर उल्टा करता हैः चार बार स्थानिक रिज़ॉल्यूशन को दोगुना करता है और चैनल को आधा करता है। स्kip कनेक्शन प्रत्येक रिज़ॉल्यूशन पर डिकोडर सुविधाओं के साथ मेल खाने वाले एन्कोडर सुविधाओं को एक साथ जोड़ते हैं। अंतिम 1x1 conv मानचित्र `64 -> num_classes` पूर्ण संकल्प पर।

स्किप कनेक्शन आवश्यक क्यों हैंः डिकोडर ने पिक्सेल-स्तर की भविष्यवाणियों को आउटपुट करने का प्रयास करने तक केवल छोटे फीचर मैप्स देखे हैं। स्किप के बिना यह किनारों को सटीक रूप से स्थान नहीं दे सकता है क्योंकि उस जानकारी को एन्कोडर में संपीड़ित किया गया था। स्किप कनेक्शन उसे उच्च रिज़ॉल्यूशन फीचर मैप्स प्रदान करता है। नीचे जाने के रास्ते में गणना किए गए एन्कोडर को मैप्स।

### ट्रांसपोस्टेड बनाम द्विआधारी उप-सैंपल

डेकोडर को अंतरिक्ष आयामों का विस्तार करना है। दो विकल्पः

- **ट्रांसपोस्ड कन्वॉल्यूशन** (`nn.ConvTranspose2d`)  अपरेसन योग्य नमूना. ऐतिहासिक यू-नेट डिफ़ॉल्ट। यदि चरण और नाभिक आकार समान रूप से विभाजित नहीं होते हैं तो चेकरबोर्ड कलाकृतियां उत्पन्न कर सकते हैं।
- **Bilinear upsample + 3x3 conv** एक संकुल के बाद एक चिकनी अपसैम्पल। कम कलाकृतियों, कम मापदंडों, अब आधुनिक डिफ़ॉल्ट।

दोनों ही प्राकृतिक रूप से दिखाई देते हैं. पहली यू-नेट के लिए, द्विआधारी अधिक सुरक्षित है.

### पिक्सेल ग्रिड पर क्रॉस-एंट्रोपी

सी वर्गों के साथ अर्थिक खंडन के लिए, मॉडल आउटपुट है `(N, C, H, W)`लक्ष्य है `(N, H, W)` पूर्णांक वर्ग के साथ IDs. क्रॉस-एंट्रोपी वर्गीकरण मामले के समान है, बस प्रत्येक स्थानिक स्थिति पर लागू किया जाता हैः

```text
Loss = mean over (n, h, w) of -log( softmax(logits[n, :, h, w])[target[n, h, w]] )
```

`F.cross_entropy` में PyTorch इस आकार को मूल रूप से संभालता है. कोई पुनर्विकृति की आवश्यकता नहीं है.

### ड्यूस हार और आपको इसकी आवश्यकता क्यों है

क्रॉस-एंट्रोपी हर पिक्सेल को समान रूप से व्यवहार करती है। यह गलत है जब एक वर्ग फ्रेम पर हावी होता है (चिकित्सा इमेजिंगः 99% पृष्ठभूमि, 1% ट्यूमर) । नेटवर्क हर जगह पृष्ठभूमि की भविष्यवाणी करके 99% सटीकता प्राप्त कर सकता है और फिर भी बेकार हो सकता है।

डस हानि प्रत्यक्ष रूप से भविष्यवाणी और वास्तविक मुखौटा के बीच ओवरलैप को अनुकूलित करके इस हल करता हैः

```text
Dice(p, y) = 2 * sum(p * y) / (sum(p) + sum(y) + epsilon)
Dice_loss = 1 - Dice
```

जहां `p` एक वर्ग के लिए सिग्मोइड/सॉफ्टमैक्स संभावना मानचित्र है और `y` यह द्विआधारी आधार सत्य मुखौटा है। हानि शून्य है केवल जब ओवरलैप सही है। क्योंकि यह अनुपात आधारित है, वर्ग असंतुलन irrelevant है।

अभ्यास में, **संयुक्त हानि**:

```text
L = L_cross_entropy + lambda * L_dice       (lambda ~ 1)
```

क्रॉस-एंट्रोपी प्रशिक्षण के शुरुआती चरण में स्थिर ग्रेडिएंट देती है; डैस प्रशिक्षण के पूंछ को वास्तव में मुखौटा के आकार से मेल खाने पर केंद्रित करता है। यह संयोजन चिकित्सा-छविकरण डिफ़ॉल्ट है और किसी भी वर्ग-असम संतुलित डेटासेट पर हराया जाना मुश्किल है।

### मूल्यांकन मेट्रिक्स

- **पिक्सेल सटीकता** प्रतिशत पिक्सल सही ढंग से भविष्यवाणी की। सस्ता. वर्गीकरण में सटीकता के समान कारण के लिए असंतुलित डेटा पर टूट गया।
- **IoU प्रति वर्ग** प्रत्येक वर्ग के मुखौटे के लिए संघ पर चौराहे; औसत पार classes = mIoU.
- **डाइस (F1 पिक्सेल पर)** समान IoU; `Dice = 2 * IoU / (1 + IoU)`चिकित्सा इमेजिंग डाइस पसंद करता है, ड्राइविंग समुदाय पसंद करता है IoU; वे एकतरफा रूप से संबंधित हैं।
- **सीमा F1** यह मापता है कि अनुमानित सीमाएं जमीन-सत्य सीमाओं के कितने करीब हैं, यहां तक कि छोटे बदलावों को दंडित करना।

रिपोर्ट IoU प्रति वर्ग, न केवल mIoU. अर्थ IoU एक वर्ग 15% पर है जबकि नौ अन्य 85% पर हैं।

### इनपुट रिज़ॉल्यूशन ट्रेडऑफ

यू-नेट का एन्कोडर रिज़ॉल्यूशन को चार गुना आधा करता है, इसलिए इनपुट को 16 से विभाजित किया जाना चाहिए। चिकित्सा छवियां अक्सर 512x512 या 1024x1024 होती हैं। स्वायत्त ड्राइविंग फसलें 2048x1024 होती हैं। यू-नेट के मेमोरी लागत के साथ पैमाने `H * W * C_max`, और 1024x1024 के साथ 1024 बोतल गला चैनल आगे पास पहले से ही उपयोग करता है VRAM.

दो मानक उपायः
1. टाइल इनपुट  प्रक्रिया 256x256 टाइलों के साथ ओवरलैप और सिलाई।
2. फ्लैश को विस्तारित घुमावों से बदलें जो स्थानिक संकल्प को अधिक बनाए रखते हैं लेकिन रिसेप्टिव फ़ील्ड (प्रोफाइल) को व्यापक बनाते हैं। DeepLab परिवार) ।

एक पहले मॉडल के लिए, एक 256x256 इनपुट के साथ 64-चैनल-बेस यू-नेट 8 पर आराम से ट्रेन करता है GB VRAM.

```figure
segmentation-flood
```

## इसे बनाओ

### चरण 1: एन्कोडर ब्लॉक

बैच मान के साथ दो 3x3 कन्वे और ReLU. पहला कन्वे बदलता है चैनल की गिनती; दूसरा इसे रखता है।

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
```

इस ब्लॉक का उपयोग पूरे समय किया जाता है। `bias=False` क्योंकि BNबीटा पूर्वाग्रह को संभालता है।

### चरण 2: नीचे और ऊपर ब्लॉक

```python
class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_c, out_c),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
```

केवल स्थानिक रूप की जांच (`shape[-2:]`) इनपुट को संभालता है जिसका आयाम 16 से विभाजित नहीं होता है; `F.interpolate` पूर्ण आकार की तुलना करने से चैनल-कंटेंट में अंतर भी होगा, जो एक जोरदार त्रुटि होनी चाहिए, एक मौन इंटरपोलेट नहीं।

### चरण 3: यू-नेट

```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.d4 = Down(base * 8, base * 16)
        self.u1 = Up(base * 16 + base * 8, base * 8)
        self.u2 = Up(base * 8 + base * 4, base * 4)
        self.u3 = Up(base * 4 + base * 2, base * 2)
        self.u4 = Up(base * 2 + base, base)
        self.outc = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)

net = UNet(in_channels=3, num_classes=2, base=32)
x = torch.randn(1, 3, 256, 256)
print(f"output: {net(x).shape}")
print(f"params: {sum(p.numel() for p in net.parameters()):,}")
```

आउटपुट आकार `(1, 2, 256, 256)` इनपुट के समान स्थानिक आकार, `num_classes` 7.7M के बारे में पैरामीटर `base=32`.

### चरण 4: नुकसान

```python
def dice_loss(logits, targets, num_classes, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * targets_one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1 - dice.mean()


def combined_loss(logits, targets, num_classes, lam=1.0):
    ce = F.cross_entropy(logits, targets)
    dc = dice_loss(logits, targets, num_classes)
    return ce + lam * dc, {"ce": ce.item(), "dice": dc.item()}
```

Dice प्रति वर्ग की गणना की जाती है और फिर औसत (मैक्रो Dice) । `eps` बैच से अनुपस्थित वर्गों पर शून्य से विभाजन को रोकता है।

### चरण 5: IoU मेट्रिक

```python
@torch.no_grad()
def intersection_union_per_class(logits, targets, num_classes):
    preds = logits.argmax(dim=1)
    intersections = torch.zeros(num_classes, device=logits.device)
    unions = torch.zeros(num_classes, device=logits.device)
    for c in range(num_classes):
        pred_c = (preds == c)
        true_c = (targets == c)
        intersections[c] = (pred_c & true_c).sum()
        unions[c] = (pred_c | true_c).sum()
    return intersections, unions


def iou_from_counts(intersections, unions):
    ious = torch.full_like(intersections, float("nan"), dtype=torch.float32)
    present = unions > 0
    ious[present] = intersections[present].float() / unions[present].float()
    return ious
```

प्रत्येक सत्यापन बैच पर चौराहे और संघ वेक्टरों को जमा करें, फिर कॉल करें `iou_from_counts` यह एक छोटे अंतिम बैच के बजाय एक पूरे बैच के रूप में एक ही प्रभाव देता है. `nan` मूल्यांकन किए गए संपूर्ण डेटासेट में अनुपस्थित वर्ग को दर्शाता है।

### चरण 6: अंत-से-अंत सत्यापन के लिए सिंथेटिक डेटासेट

स्वतंत्र रूप से यादृच्छिक रंगीन पृष्ठभूमि पर एक से तीन आकार उत्पन्न करें। आकार रंगों को सर्कल / वर्ग वर्ग से स्वतंत्र रूप से नमूना लिया जाता है, इसलिए मॉडल एक निश्चित पैलेट को याद करके कार्य को हल नहीं कर सकता है। एक दृश्य दोनों वर्गों को शामिल कर सकता है।

```python
import numpy as np
from torch.utils.data import Dataset, DataLoader

def synthetic_segmentation(num_samples=200, size=64, seed=0):
    if size < 16:
        raise ValueError("size must be at least 16 pixels")

    rng = np.random.default_rng(seed)
    images = np.zeros((num_samples, size, size, 3), dtype=np.float32)
    masks = np.zeros((num_samples, size, size), dtype=np.int64)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    min_radius = max(3, size // 16)
    max_radius = max(min_radius + 1, size // 5)
    for i in range(num_samples):
        images[i] = rng.uniform(0.1, 0.9, size=3)
        num_shapes = int(rng.integers(1, 4))
        for _ in range(num_shapes):
            cls = int(rng.integers(1, 3))
            color = rng.uniform(0.05, 0.95, size=3)
            radius = int(rng.integers(min_radius, max_radius + 1))
            cx = int(rng.integers(radius, size - radius))
            cy = int(rng.integers(radius, size - radius))
            if cls == 1:
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 < radius ** 2
            else:
                mask = (np.abs(xx - cx) < radius) & (np.abs(yy - cy) < radius)
            images[i][mask] = color
            masks[i][mask] = cls
        images[i] += rng.normal(0, 0.02, images[i].shape)
        images[i] = np.clip(images[i], 0, 1)
    return images, masks


class SegDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i]).permute(2, 0, 1).float()
        mask = torch.from_numpy(self.masks[i]).long()
        return img, mask
```

तीन वर्गः पृष्ठभूमि (0), वृत्त (1), वर्ग (2)। नेटवर्क को आकार को अलग करना सीखना चाहिए।

### चरण 7: प्रशिक्षण लूप

```python
def train_one_epoch(model, loader, optimizer, device, num_classes):
    model.train()
    loss_sum, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss, _ = combined_loss(logits, y, num_classes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
    return loss_sum / total


@torch.no_grad()
def evaluate_iou(model, loader, device, num_classes):
    model.eval()
    intersections = torch.zeros(num_classes, device=device)
    unions = torch.zeros(num_classes, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        batch_intersections, batch_unions = intersection_union_per_class(
            model(x), y, num_classes
        )
        intersections += batch_intersections
        unions += batch_unions
    return iou_from_counts(intersections, unions)
```

सिंथेटिक डेटासेट पर 10-30 युगों के लिए इसे चलाएं और घड़ी mIoU आकार वर्गों के लिए सुधार करें। डेटासेट स्तर पर गणना संश्लेषण सुनिश्चित करता है कि बैच आकार और वर्ग की अनुपस्थिति रिपोर्ट किए गए IoU.

## इसका प्रयोग करें

उत्पादन के लिए, `segmentation_models_pytorch` ("smp") किसी भी टॉर्चविजन या टिमम रीढ़ की हड्डी के साथ प्रत्येक मानक विभाजन वास्तुकला को लपेटता है। तीन पंक्तियाँः

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3,
)
```

असली काम के लिए भी जानने लायकः
- **DeepLabV3+** अधिकतम पूल आधारित डाउनसैम्पलिंग को विस्तारित कन्वर्स से बदल देता है ताकि बोतल गला को संकल्प बनाए रखा जा सके; उपग्रह और ड्राइविंग डेटा पर तेज सीमाएं।
- **SegFormer** एक पदानुक्रमिक ट्रांसफार्मर के लिए कन्वि कोडर को स्विच करता है; वर्तमान SOTA कई बेंचमार्क पर।
- **Mask2Former** / **OneFormer** एक ही वास्तुकला में अर्थिक, उदाहरण और पैनप्टिक विभाजन को एकीकृत करें।

तीनों में से एक में गिरावट प्रतिस्थापन हैं `smp` या `transformers` एक ही डेटा लोडर के साथ।

## इसे भेजें

इस पाठ से उत्पन्न होता हैः

- `outputs/prompt-segmentation-task-picker.md` एक प्रॉम्प्ट जो अर्थिक, इंस्टेंस और पैनप्टिक सेगमेंट के बीच चुनता है और किसी दिए गए कार्य के लिए वास्तुकला का नाम देता है।
- `outputs/skill-segmentation-mask-inspector.md` एक कौशल जो कक्षा वितरण, भविष्यवाणी-मास्क आंकड़ों और उन कक्षाओं की रिपोर्ट करता है जो कम भविष्यवाणी या सीमा-अवशिष्ट हैं।

## व्यायाम

1. **(Easy)** कार्यान्वयन `bce_dice_loss` एक द्विआधारी विभाजन कार्य के लिए (पूर्वभूमि बनाम पृष्ठभूमि) सिंथेटिक दो वर्ग डेटासेट पर सत्यापित करें कि संयुक्त हानि से अधिक तेजी से अभिसरण BCE अकेले जब अग्रभूमि पिक्सल का 5% है।
2. **(Medium)** प्रतिस्थापन `nn.Upsample + conv` एक के साथ अप-ब्लॉक `nn.ConvTranspose2d` संश्लेषण डेटा सेट पर दोनों को प्रशिक्षित करें और तुलना करें mIoU. ट्रांसपोस्टेड-कन्व संस्करण में चेकरबोर्ड कलाकृतियों के स्थानों पर ध्यान दें।
3. **(Hard)** एक वास्तविक खंडन डेटासेट लें (ऑक्सफोर्ड-IIIT पालतू जानवरों, सिटीस्केप्स मिनी स्प्लिट, या एक चिकित्सा उपसमूह) और 2 के भीतर U-नेट को प्रशिक्षित करें IoU इन बिंदुओं के `smp.Unet` संदर्भ प्रति वर्ग रिपोर्ट IoU और पता लगाएं कि किस वर्ग को नुकसान में डैस जोड़ने से सबसे अधिक लाभ होता है।

## प्रमुख शर्तें

| अवधि | लोग क्या कहते हैं | इसका क्या मतलब है |
|------|----------------|----------------------|
| अर्थिक खंडन | "हर पिक्सेल को लेबल करें" | प्रति पिक्सेल वर्गीकरण C वर्गों में; एक ही वर्ग के उदाहरण एक साथ मिलें |
| इंस्टेंस सेगमेंट | "हर वस्तु को लेबल करें" | एक ही वर्ग के अलग-अलग उदाहरणों को अलग करता है; केवल अग्रभूमि |
| पैनोपिक खंडन | "सिमेटिक + उदाहरण" | प्रत्येक पिक्सेल एक वर्ग मिलता है; प्रत्येक वस्तु उदाहरण भी एक अद्वितीय आईडी प्राप्त करता है |
| स्किप कनेक्शन | "यू-नेट ब्रिज" | एन्कोडर सुविधाओं को मिलान-रिज़ॉल्यूशन डेकोडर सुविधाओं में जोड़ना; उच्च आवृत्ति विवरण को संरक्षित करना |
| स्थानांतरित किया गया | "डिस्कॉन्विल्यूशन" | सीखने योग्य अपसैम्पलिंग; शतरंज बोर्ड कलाकृतियां उत्पन्न कर सकती है |
| दाल खोना | "ओवरलैप हानि" | 1 - 2|ए  बी| / (|A| + |B|); मास्क ओवरलैप को सीधे अनुकूलित करता है और वर्ग असंतुलन के लिए मजबूत है |
| mIoU | "संघ के ऊपर औसत चौराहे" | औसत IoU वर्गों के बीच; खंडन के लिए सामुदायिक मानक मीट्रिक |
| सीमा F1 | "सीमित सटीकता" | F1 केवल सीमा पिक्सेल पर गणना किए गए स्कोर; सटीक-महत्वपूर्ण कार्यों के लिए मामले |

## आगे पढ़ना

- [यू-नेटः बायोमेडिकल इमेज सेगमेंट के लिए कन्व्यूशनल नेटवर्क (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597) मूल कागज; प्रत्येक व्यक्ति का प्रतिलिपि बनाने का आंकड़ा पृष्ठ 2 पर है
- [पूर्ण रूप से संवर्धित नेटवर्क (Long et al., 2015)](https://arxiv.org/abs/1411.4038) पेपर जो पहली बार विभाजन को एक अंत-से-अंत कन्वि समस्या बना दिया
- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) उत्पादन खंडन के लिए संदर्भ; प्रत्येक मानक वास्तुकला प्लस प्रत्येक मानक हानि
- [प्रशिक्षण से सीखे गए पाठ SOTA खंडन (kaggle.com प्रतियोगिताएं)](https://www.kaggle.com/code/iafoss/carvana-unet-pytorch) क्यों के बारे में एक मार्गदर्शक TTA, छद्म लेबलिंग, और वर्ग वजन वास्तविक डेटा पर मायने रखते हैं
