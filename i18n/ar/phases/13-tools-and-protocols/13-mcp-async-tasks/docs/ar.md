# MCP توسيع المهام: عمل مستمر على أساس بلا جنسية

> بدون جنسية MCP لا يعني أن كل عملية يجب أن تنتهي في طلب واحد. تمديد المهام الرسمية يعطي العمل الطويل المشي مسدسة دائمة صريحة. يمكن للخادم إعادة هذا المكافئ من `tools/call`، أي حالة يمكن أن تجيب `tasks/get`ومدخل العميل يصل من خلال `tasks/update` بدون إحياء جلسات البروتوكول

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 11 (stateless MRTR), Phase 13 · 12 (elicitation)
**Time:** ~90 minutes

## أهداف التعلم

- تمييز النقل البروتوكول بدون ولاية عن حالة مهمة التطبيق الدائمة.
- التفاوض `io.modelcontextprotocol/tasks` توسيع قدرات الطلب `server/discover`.
- إرجاع إرسال المستخدم `CreateTaskResult` مع `resultType: "task"` إلا بعد خلق دائم
- استطلاع مع `tasks/get`، إنجاز مدخل المهمة `tasks/update`وطلب إلغاء التعاون مع `tasks/cancel`.
- إزالة الأكبر سنا `tasks/status`, `tasks/result`و `tasks/list` الافتراضات
- الاشتراك في إشعارات المهمة الاختيارية من خلال `subscriptions/listen` على POST ردود فعل SSE -تدفق
- انتهاء صلاحية المهام النموذجية، وإعادة تشغيل الاسترداد، وتقليل النسخة في مفتاح المدخل، وأخطاء التنفيذ بشكل صحيح.

## لماذا المهام هي امتداد

ظهرت المهام لأول مرة كجزء تجريبي أساسي في 2025-11-25. `io.modelcontextprotocol/tasks` التوسع حتى يتمكن العملاء والسيرفر من اختيار دورة حياة إضافية دون توسيع بروتوكول الأساس للجميع.

تُبقى مواصفات التوسع مسودة سطحية على الرغم من أنها هي الموقع الرسمي الحالي للمهام. SDK، إشغال سيناريوهات التوافق، وعزل مُعايير الأسلاك من عاملك ومجال التخزين

استخدم المهمة عندما يكون للعملية واحدة أو أكثر من هذه الخصائص:

- قد تفوق وقت الطلب العادي
- نظام عمل عمل خارجي يمتلك بالفعل تنفيذ.
- العميل بحاجة إلى التعافي بعد إعادة تشغيله
- يتم وقف العملية لإدخال المستخدم أو النموذج أثناء التنفيذ.
- الإلغاء والحصول على النتائج الدائمة هي متطلبات المنتج.

لا تخلق مهمة للبحث التحديدي الرخيص. التدخل، الاستمرار، الاستطلاع، انتهاء الصلاحية، والإلغاء تعقيدا حقيقيا.

## أساسية بدون جنسية، تطبيق دولي

MCP 2026-07-28 إزالة `initialize`, `notifications/initialized`، جلسات البروتوكول، `Mcp-Session-Id`هذا لا يحظر المنتجات الحكومية

تُعد هوية المهمة حالة تطبيق صريحة:

- الخادم يصرّ عليه قبل إعادتها
- العميل يمكنه تخزينها وتجربة مرة أخرى بعد إعادة تشغيلها
- يمكن أن يُوجّه الهوية إلى أي نسخة مدعومة من نفس المتجر الدائم.
- يتم التحقق من الائتمان على كل طريقة المهمة.
- يتم تعريف انتهاء الصلاحية والحذف من خلال حقل المهام، وليس مدة حياة النقل.

هذا مختلف عن الحالة الخفية المرتبطة بالاتصال.

أبقوا أربع حياة منفصلة

| الدولة | طوال الحياة | حيث ينتمي |
|---|---|---|
| البيانات المعدنية للبروتوكول | طلب واحد | `params._meta`، يتم التحقق من جديد في كل مكالمة |
| أعمال النقل | طلب واحد من الاستديو أو HTTP ردود فعل | منسق الطيران مع مدة محددة |
| MRTR الاستمرار | تسلسل إعادة المحاولة | حماية النزاهة `requestState`، بالإضافة إلى التحكم في إعادة التشغيل عند الحاجة |
| مهمة دائمة | عبر الطلبات، النسخ، إعادة تشغيل، وإعادة الاتصال | متجر التطبيقات المشتركة المفتاح من قبل المرخص `taskId` |

نقل سجل المهمة إلى ذاكرة العملية لا يجعل MCP هذا يجعل التطبيق غير موثوق به البروتوكول يبقى بلا بيان، ولكن `tasks/get` لا يمكن استرداد السجل. استمر قبل إعادة المقبض، ثم جعل كل طريقة المهمة لحل نفس السجل المشترك تحت الفائز والفحص الرئيسي.

## التفاوض حول القدرة

يعلن العميل عن الدعم على كل طلب مؤهل:

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "extensions": {
        "io.modelcontextprotocol/tasks": {}
      }
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "lesson-client",
      "version": "1.0.0"
    }
  }
}
```

الخادم يعود بالضبط `supportedVersions`، القدرات `ttlMs`و `cacheScope` من `server/discover`حيث أنها تعلن عن الأدوات، فإنها تنفيذ أيضاً الإجبارية `tools/list`هذا النتيجة تعود إلى تحديد `generate_report` وصف، كائن صالح `inputSchema`, `resultType: "complete"`، بيانات الهوية الخادم، وتلميحات التخزين العام.

طريقة مهمة من عميل لم يعلن عن إرجاع التوسع `-32021`, المطلوب من الموظفين المفقودين , مع `data.requiredCapabilities` المحددة إلى `{"extensions":{"io.modelcontextprotocol/tasks":{}}}`. تعود سلسلة بروتوكول غير مدعومة `-32022` مع دقة `supported` و `requested` البيانات؛ إعادة نسخة مفقودة أو غير سلسلة `-32602`.

غلاف بدون JSON-RPC `id` هو إشعار. يمكن للمستلم معالجته، لكنه لا ينشر JSON-RPC نتيجة أو خطأ. HTTP إرجاع المعدل `202 Accepted` بدون هيئة للإخطار المقبول.

في الوقت الحالي، فقط `tools/call` يدعم تنفيذ مهام معززة. تصميم التجريد الداخلي الخاص بك بحيث أن أنواع الطلبات المستقبلية لا تتطلب إعادة كتابة التخزين.

## إنشاء المهام الموجهة إلى الخادم

العلم القديم للعميل `params._meta.task.required` يُعلن العميل دعم التوسع، ثم يقرر الخادم ما إذا كان `tools/call` يصبح مهمة

الطلب:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "generate_report",
    "arguments": {"size": "large"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

رد:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "task",
    "taskId": "tsk_786512e29e0d",
    "status": "working",
    "statusMessage": "Preparing report outline.",
    "createdAt": "2026-08-21T10:30:00Z",
    "lastUpdatedAt": "2026-08-21T10:30:00Z",
    "ttlMs": 900000,
    "pollIntervalMs": 1000
  }
}
```

يجب على الخادم عدم إعادة هذه المقبضة حتى `tasks/get` في متجر متسق في نهاية المطاف، انتظر رؤية القراءة قبل الإجابة. خلاف ذلك يمكن للعميل الحصول على هوية صالحة تبدو و تحصل على "لا يمكن العثور على" على الفور.

لا يتم طلب استجابة للمهمة بمعنى أن العميل لا يطلب وضع المهمة. لا يتم التفاوض عليه: لا يزال على الطلب الحالي إعلان التوسع.

## شكل المهمة

كل مهمة تحمل:

- `taskId`: معرف مستقر منخفض الخادم
- `status`: `working`, `input_required`, `completed`, `cancelled`أو `failed`;
- `createdAt` و `lastUpdatedAt`: ISO 8601 طوابع زمنية
- `ttlMs`: مدة انتهاء الصلاحية من الابتكار، أو `null` بدون حد إعلاني
- اختيارية `pollIntervalMs`: الحد الأدنى المفترض للحصول على الانتخابات من الخادم
- اختيارية `statusMessage`: سياق يستهدف المستخدم أو النموذج.

تظهر الحقول الخاصة بالحالة فقط عندما تكون ذات صلة:

- `input_required` يشمل `inputRequests`.
- `completed` يتضمن طلب الأصلي `result` الشكل
- `failed` يتضمن JSON-RPC `error` -أجسام

يجب على العميل أن يحترم `pollIntervalMs`قد يحدد الخادم المعدلات التي تستهدف استطلاعات أكثر عدوانية ويمكن أن يغير الفاصل على مدى عمر المهمة.

## استطلاع مع `tasks/get`

العميل يطلب صورة حالية:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/get
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tasks/get",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

`tasks/get` نفسها قد اكتملت، لذلك نتيجة لها دائما `resultType: "complete"`المهمة المُعقدة لا تزال قد تكون `status: "working"` أو `status: "input_required"`.

هذا التمييز يمنع حدوث خطأ عام في المصفح:

```text
result.resultType = complete    means the tasks/get RPC finished
result.status = working        means the represented job is still running
```

لا يوجد `tasks/result` عندما تنتهي المهمة، `tasks/get` رد يظهر في الأصلية `CallToolResult` تحت `result`:

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "completed",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:34:12Z",
  "ttlMs": 900000,
  "result": {
    "resultType": "complete",
    "content": [
      {"type": "text", "text": "Generated large report with approved outline."}
    ],
    "structuredContent": {"size": "large", "approved": true},
    "isError": false,
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "tasks-demo",
        "version": "1.0.0"
      }
    }
  },
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "tasks-demo",
      "version": "1.0.0"
    }
  }
}
```

الخارجي `resultType` يقول `tasks/get` RPC تم الانتهاء من ذلك `result.resultType` يقول أن الاتصال الأصلي الأداة قد اكتملت. هذا التمييز المتعقد مطلوب. `CallToolResult` SHOULD أيضا تحمل الخاصة به `io.modelcontextprotocol/serverInfo`هذا الدروس يتضمنها بدلا من تخزين حمولة مفيدة غير محددة.

لا يوجد `tasks/list`. لا يمكن لخادمات غير الجلسة استنتاج الآمن عن المهام التي تنتمي إلى قائمة محددة عن الاتصال. يجب على التطبيقات التي تحتاج إلى تاريخ الكشف عن أداة نطاق مصرح بها مع مرشحات صريحة وقواعد الملكية.

## إدخال أثناء تنفيذ المهمة

مدخل المهام والقاعدة MRTR تبدو متشابهة ولكن تستخدم مواصلات مختلفة

### الإدخال المطلوب قبل إنشاء المهمة

النواة العائدة `resultType: "input_required"` من الأصلي `tools/call`العميل يقوم به ويحاول مرة أخرى تلك المكالمة الأصلية فقط إخلال المهمة بعد تلك المكالمات المزمنة MRTR -تنتهي الرصاصات

### الإدخال المطلوب بعد إنشاء المهمة

حدد المهمة إلى `input_required`. `tasks/get` يُكشف عن المشكلة `inputRequests`، ويرسل العميل الردود عبر `tasks/update`العميل لا يحاول إعادة كتابة الأصلية `tools/call`.

صورة سريعة:

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "input_required",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:31:00Z",
  "ttlMs": 900000,
  "inputRequests": {
    "approve_outline": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Approve the generated report outline?",
        "requestedSchema": {
          "type": "object",
          "properties": {"approved": {"type": "boolean"}},
          "required": ["approved"]
        }
      }
    }
  }
}
```

تحديث:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/update
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tasks/update",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "inputResponses": {
      "approve_outline": {
        "action": "accept",
        "content": {"approved": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

رد النجاح هو اعتراف فارغ زائد `resultType: "complete"`قد يكون التغيير في الدولة متسقًا في النهاية، لذا يواصل العميل إجراء الاستطلاع أو الاستماع.

كل واحد `inputRequests` يجب أن تكون مفتاحها فريدة طوال عمر المهمة. `tasks/get` يمكن أن تظهر اللقطات نفس المفتاح المميز ؛ العملاء يكررون UI والخادمات تتجاهل الردود على المفاتيح غير المعروفة أو المبدلة أو المكتملة بالفعل. `input_required` حتى يتم الإجابة على جميع المفاتيح المطلوبة

## الإلغاء هو تعاون

`tasks/cancel` إشارات النية وتعطي إقرارًا فارغًا كاملًا. هذا الإقرار لا يضمن توقف العامل. قد ينتهي العمل أولاً، أو يتجاهل الإلغاء أو الانتقال في وقت لاحق.

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/cancel
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tasks/cancel",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

لكل أساليب المهمة الثلاثة `Mcp-Name` المرايا `params.taskId`. لا تكرر JSON-RPC اسم الطريقة `code/main.py` مركزية هذه القاعدة في `make_http_request`.

يقوم العامل في الدروس بتشرف الإلغاء على الفور، ويعمل على المكالمات المتكررة غير قابلة. لا يزال على عميل الإنتاج أن يعامل الإلغاء كمتعاون بدلاً من استنتاج حالة المهمة النهائية من التأكيد.

لا تستخدم `notifications/cancelled` لتحل مهمة، هذا الإخطار ينتمي إلى طلب إلغاء، وليس مهمات دائمة.

التمييز مهم في حدود التوجه طلب إلغاء يهدف إلى واحد في الرحلة JSON-RPC العملية أو مستوى الطلب HTTP الإجابة `tools/call` لقد عاد بالفعل `resultType: "task"`، أن الطلب قد اكتمل و لا يمكن لإغلاق النقل أن يذكر أو يوقف العمل الدائم. `tasks/cancel` هو مرخص جديد RPC. إنه يحمل `params.taskId`, مرآة تلك الهوية `Mcp-Name`، يحل الخلفية المملكة للمهمة، ويُسجل نية إلغاء التعاون، ويُعيد تأكيدًا دون أن يدعي العامل أنه توقف.

وبالتالي يجب على البوابة أن تحتفظ بمنسقات الطلبات وطرق المهام في جداول مختلفة. يمكن أن تختفي جدول الطلبات عند انتهاء الاستجابة. يجب أن تبقى مسار المهام حتى تنتهي الحالة النهائية والاحتفاظ بها. [الدروس 29: MCP موثوقية، إلغاء، ومراقبة التدفق](../../29-mcp-reliability-cancellation-and-flow-control/docs/en.md) يُبني السباق، وقتاً وقفياً، وفرصاً، و ضغوطاً، و إعادة محاولة القواعد لكلا الطرق.

## الإخطارات الاختيارية

استطلاعات الرأي هي الخط الأساسي. العميل الذي يريد تحديثات دفع يرسل `subscriptions/listen` مع هويات المهام. HTTP، هذا هو POST والذي يقدم استجابة محددة حسب الطلب SSE لا يوجد مجرى GET سلسلة الأحداث و لا جلسة بروتوكول للحفاظ على الحياة.

الجهاز يعترف بالهوية المقبولة مع `notifications/subscriptions/acknowledged` ويمكن بعد ذلك إرسال اللقطات الفورية الكاملة `notifications/tasks`. الإقرار وكل إشعار مهمة تحمل `io.modelcontextprotocol/subscriptionId` في `_meta`, مساوية `subscriptions/listen` كل إشعار مهمة يعادل ما `tasks/get` سيعود في تلك اللحظة

يجب على العملاء أن يعلنوا على طول المهام. يجب أن يعيدوا الاتصال واستئناف من أوراق تعريف المهام الدائمة بدلاً من الاعتماد على إعادة تشغيل الأحداث أو `Last-Event-ID`.

## النقص في النطقية

استخدم طبقتين الخطأ بشكل صحيح.

### خطأ بروتوكول

تعود معايير طريقة غير صالحة أو معرف مهم مجهول JSON-RPC الخطأ، عادة `-32602`. إعلانات الدعم المفقودة `-32021` مع جسم القدرة المطلوب.

### نتائج تنفيذ المهمة

- نتيجة أداة طبيعية مع `isError: true` ما زالت `completed` المهمة لأن دعوة الأداة قدمت نتيجة محددة.
- A JSON-RPC خطأ أثناء التنفيذ المؤجل يجعل المهمة `failed` ويتخزن ذلك JSON-RPC خطأ في `error`.
- الرفض المستخدم يمكن أن يؤدي `cancelled`نتيجة رفض كاملة أو نتيجة آمنة أخرى محددة للمجال.

## استمرارية، انتهاء صلاحية، وملكية

استمر على الأقل في تحديد اسم المهمة ، والحالة ، والخوابات الزمنية ، ttl ، فترة الاستطلاع ، والمتلكية الأصلية للعملية ، والنتيجة أو الخطأ ، والطلبات المتبقية للمدخول ، وجميع مفاتيح المدخل الصادرة.

يجب أن يحتوي مفتاح التخزين على مستأجر ومدير مصرح أو يحل ذلك. لا يجب أن يمنح معرفة هوية المهمة الوصول. تحقق ملكية كل `tasks/get`, `tasks/update`, `tasks/cancel`، والإشتراك

`ttlMs` يمكن للمستخدمين أن يستخدموا هذه المعلومات في التطبيقات، ويمكن أن يستخدمها في التطبيقات، ويمكن أن يتم قياسها من وقت إنشاءها ويمكن أن يتغير. يمكن للمستخدمين التعامل معها كمساعدة خلفية عندما توقفت المهمة عن إنتاج تحديثات قابلة للملاحظة. قد يفشل الخادم ويمحو في وقت لاحق مهمة انتهت صلاحيتها. لا تصفها كوعد بالاحتفاظ بالنتيجة المكتملة لعدة ملثانية بعد الانتهاء.

استخدام الكتب الذرية أو المعاملات. الدرس يكتب ملفًا مؤقتًا ويُعيد تسميته بشكل ذري. يجب أن تستخدم خدمة متعددة النسخة متجرًا دائمًا مشتركًا وترخيص عامل أو تحكم متزامن معادلة.

```figure
tp-task-lifecycle
```

## بناءها

`code/main.py` تنفيذ خدمة مهمة تحديدية:

- `server/discover` العائدات `supportedVersions`، إشارات التخزين، ومدّة المهام
- `tools/list` يعود تحديد، قابلة للتخفيض `generate_report` وصف مع مخطط إدخال صالح.
- `tools/call` يخلق و يستمر المهمة قبل العودة `resultType: "task"`.
- مثال خدمة جديد يُعيد تحميل نفس المهمة، يُظهر إعادة تشغيل التعافي.
- `tasks/get` يعيد صور اللقطات الكاملة للمهمة.
- العامل ينتقل من `working` إلى `input_required`.
- `tasks/update` يقبل استجابة من النموذج ويرد إقرارًا كاملًا فارغًا.
- العامل يحتفظ بالعش `CallToolResult` مع نفسها `resultType` و هوية الخادم ، ثم الانتقال إلى `completed`.
- `tasks/cancel` لا يُمكن أن يكون ذلك ممكناً في هذا التنفيذ.
- المُسَمِع HTTP مجموعات البناء `Mcp-Name` إلى `params.taskId` لـ `tasks/get`, `tasks/update`و `tasks/cancel`.
- استخدام مساعدي الإخطار `notifications/subscriptions/acknowledged` و `notifications/tasks`، كلاهما مع علامة اسم طلب الاستماع.
- الإخطارات بدون رقم تعطي لا JSON-RPC رد فعل

يعمل العامل بشكل صريح بدلاً من النوم في خيط خلفي، مما يجعل كل انتقال حالة محدداً ويحافظ على مثال البروتوكول منفصل عن ميكانيكا الصف.

## استخدمها

من جذور المخبأ:

```bash
cd phases/13-tools-and-protocols/13-mcp-async-tasks/code
python3 main.py
python3 -m unittest discover tests -v
```

تسلسل النتائج المتوقع:

```text
id=0 resultType=complete status=ack
id=1 resultType=task status=working
id=2 resultType=complete status=working
id=3 resultType=complete status=input_required
id=4 resultType=complete status=ack
id=5 resultType=complete status=completed
```

أيضا التحقق من أن `tasks/status`, `tasks/result`و `tasks/list` طريقة العودة لا توجد في الخدمة الحديثة.
تأكد من ذلك `tools/list` هو تحديدية وكل تيار HTTP طريقة المهمة تعكس اسم المهمة من خلال `Mcp-Name`.

## أرسله

`outputs/skill-task-store-designer.md` الآن تنتج تصميمًا واعًا للتوسع: تفاوض القدرات، إنشاء استمرار قبل العودة، والطرق الحالية، وتدفق تحديث المدخلات، والمالك، والانتهاء من الصلاحية، والإلغاء، والإشتراك، والهجرة من الطرق التجريبية المزودة.

## التمارين

1. إضافة مفتاح إدخال ثانيا غير مؤقتة. إرسال مفتاح جزئي `tasks/update` و إثبات أن المهمة لا تزال قائمة `input_required` حتى يتم الإجابة على كلتا المفاتيح
2. إضافة ملكية المستأجر إلى المتجر ورفض هوية مهمة صالحة قدمتها رئيس المصادقة الخطأ.
3. إضافة عقد تأجير العمال مع انتهاء صلاحيتها. إثبات أن حالات الخدمة لا يمكن أن تقوم بنفس المهمة في وقت واحد.
4. تنفيذ POST-response SSE المعدل `subscriptions/listen`لا تضيف GET, `Last-Event-ID`أو عنوان جلسة
5. إضافة تنظيف انتهاء الصلاحية. تمييز مهمة انتهت الصلاحية من هوية مهمة خاطئة دون تسريب وجود المتبادل المستأجر.

## الشروط الرئيسية

| المدة | المعنى في التوسع الحالي |
|------|----------------------------------|
| تمديد المهام | اختياري `io.modelcontextprotocol/tasks` القدرة على العمل في التزامن المستدام |
| `CreateTaskResult` | المستهلك `resultType: "task"` رد على طلب مؤهل |
| `tasks/get` | استقصاء صورة مفاجأة كاملة للعمل الحالي، بما في ذلك النتيجة النهائية أو المدخلات المنتظرة |
| `tasks/update` | إرسال الردود على المهمة المعلقة `inputRequests` |
| `tasks/cancel` | الاعتراف بنية الإلغاء التعاونية |
| `input_required` | حالة المهمة التي تشير إلى إدخال العميل غير متوقع |
| `pollIntervalMs` | التأخير الحد الأدنى المقترح من الخادم قبل استطلاع آخر |
| `ttlMs` | مدة انتهاء الصلاحية التي تم قياسها من إنشاء المهمة |
| استمرارية قبل العودة | قاعدة يجب أن يحل اسم المهمة قبل إرسال مسدسه |
| `notifications/tasks` | صورة مفاجأة كاملة اخيارية يتم تسليمها على اشتراك SSE ردود فعل |

## التوافق مع التراث

تستخدم السطح التجريبي 2025-11-25 زيادة المهام التي يطلبها العميل، `tasks/status`, `tasks/result`، و اختياري `tasks/list`. إبقوا هذه الأسماء فقط داخل مُعدّل إرث مُثبت . العميل الحالي يستخدم القدرة على التوسع ، يقبل المُسَلّطات المُوجّهة إلى الخادم ، استطلاعات الرأي `tasks/get`، وتزويد المدخلات `tasks/update`، ويقرأ النتيجة النهائية من صورة المهام.

## المزيد من القراءة

- [رسمية MCP تمديد المهام](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
- [MCP 2026-07-28 طلبات رحلة متعددة](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 قابل للتسجيل HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
