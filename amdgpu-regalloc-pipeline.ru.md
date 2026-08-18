# Регистровый аллокатор в бэкенде AMDGPU: конвейер, liveness, работа аллокатора и спилл

Документ подробно разбирает, **как устроен конвейер кодогенерации AMDGPU (GCN)**, **где в нём
сидит регистровый аллокатор**, **какой анализ времени жизни (liveness) готовит для него данные**,
**как работает сам аллокатор** и **как происходит спилл**. Всё привязано к реальному коду
(`file:line`) на дереве `/home/mrsidims/LLVM-work/llvm-project`.

Терминология MFMA/WMMA/occupancy предполагается известной (см.
`amdgpu-coexec-scheduler-overview.ru.md` и `amdgpu-backend-autotuning.ru.md`). Здесь фокус —
именно на регистрах.

---

## 0. TL;DR

- Регистровый аллокатор стоит **после ISel и после pre-RA планировщика (`MachineScheduler`,
  в т.ч. coexec/GCN-стратегия)** и **до post-RA стадий**. То есть весь coexec-тюнинг из
  соседних документов работает **над виртуальными регистрами**, а RA превращает их в физические.
- AMDGPU **не использует один `-regalloc`**. Вместо этого — **три отдельных прохода greedy**:
  сначала SGPR, потом WWM-регистры, потом VGPR (`AMDGPUTargetMachine.cpp:1909`
  `addRegAssignAndRewriteOptimized`). Флаг `-regalloc` для amdgcn запрещён
  (`:1879-1881`), есть `-sgpr-regalloc`/`-wwm-regalloc`/`-vgpr-regalloc`.
- Liveness для RA — это **`LiveIntervals`**: полуоткрытые сегменты `[start,end)` над номерами
  слотов (`SlotIndexes`), с номерами значений `VNInfo` и по-полосными подсегментами `SubRange`.
  Старый `LiveVariables` (def+kills, без числового порядка) используется вспомогательно.
- Аллокатор — **`RAGreedy`** (`RegAllocGreedy.cpp`). На каждый виртуальный регистр — каскад
  `tryAssign → tryEvict → trySplit → spill` (`selectOrSplitImpl:2647`), очередь по приоритету
  (крупные/глобальные раньше мелких/локальных), стадии `RS_New … RS_Done`.
- **Вес спилла** (`CalcSpillWeights.cpp`) = сумма частот use/def, делённая на размер интервала;
  ремат-дешёвые интервалы получают ×0.5 (кандидаты на спилл), «бесконечный» вес = не спиллить.
- **Спилл** делает `InlineSpiller`: сначала **ремат** (пересчёт вместо спилла), потом
  вставка store после def и reload перед use, с попыткой `foldMemoryOperand`. У AMDGPU **два
  уровня**: SGPR спиллятся **в полосы (lanes) VGPR** через `V_WRITELANE`/`V_READLANE`
  (`SILowerSGPRSpills`), а VGPR — в **scratch-память** через `SCRATCH_STORE`/`BUFFER_STORE`
  (`SIRegisterInfo::eliminateFrameIndex`). Каждый лишний spill-VGPR давит на occupancy.

---

## 1. Где в конвейере стоит регистровый аллокатор

Общий вид пути от IR до ассемблера (только релевантные фазы):

```
   LLVM IR
     │
     ▼
  ISel  (SelectionDAG / GlobalISel)          ← ещё виртуальные регистры, SSA
     │
     ▼
  Machine SSA оптимизации                     addMachineSSAOptimization()
     │
     ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  addOptimizedRegAlloc()   (TargetPassConfig.cpp:1485,                 │
 │                            переопределён в GCNPassConfig:1785)        │
 │                                                                       │
 │   DetectDeadLanes → InitUndef → ProcessImplicitDefs                   │
 │   UnreachableMachineBlockElim → LiveVariables                         │
 │   MachineLoopInfo → PHIElimination (+ SILowerControlFlow у AMDGPU)    │
 │   TwoAddressInstruction → RegisterCoalescer  (требует LiveIntervals)  │
 │   RenameIndependentSubregs                                            │
 │   ╔═══════════════════════════════════════════════════════════════╗  │
 │   ║  MachineScheduler  ← PRE-RA ПЛАНИРОВЩИК                        ║  │
 │   ║  сюда врезаны GCNPreRAOptimizations, SIWholeQuadMode,          ║  │
 │   ║  SIOptimizeExecMaskingPreRA, SIFormMemoryClauses               ║  │
 │   ║  здесь же живёт coexec-стратегия (AMDGPUCoExecSchedStrategy)   ║  │
 │   ╚═══════════════════════════════════════════════════════════════╝  │
 │                              │                                        │
 │                              ▼                                        │
 │   ╔═══════════════════════════════════════════════════════════════╗  │
 │   ║  addRegAssignAndRewriteOptimized()  ← РЕГИСТРОВЫЙ АЛЛОКАТОР    ║  │
 │   ║  (GCNPassConfig, AMDGPUTargetMachine.cpp:1909)                 ║  │
 │   ║   1. SGPR greedy → rewrite → StackSlotColoring                 ║  │
 │   ║   2. SILowerSGPRSpills  (SGPR → полосы VGPR)                   ║  │
 │   ║   3. WWM: PreAllocateWWMRegs → WWM greedy → LowerWWMCopies     ║  │
 │   ║   4. VGPR greedy → preRewrite(AGPR/NSA) → VirtRegRewriter      ║  │
 │   ╚═══════════════════════════════════════════════════════════════╝  │
 │                              │                                        │
 │                              ▼                                        │
 │   StackSlotColoring → MachineCopyPropagation → MachineLICM            │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
   addPostRegAlloc()  (SIFixVGPRCopies, SIOptimizeExecMasking)   ← POST-RA
                                │
                                ▼
   post-RA scheduler → PEI (frame) → ассемблер
```

Ключевые факты:

1. **RA — граница между виртуальным и физическим миром.** До него всё — виртуальные
   регистры, и pre-RA планировщик (включая coexec) двигает инструкции, не зная финальной
   раскраски. Именно поэтому в `amdgpu-coexec-scheduler-overview.ru.md` подчёркивалось, что
   coexec «видит» давление регистров лишь через `GCNRegPressure`, а не через реальные физрегистры.
2. **`LiveIntervals` строится к моменту `RegisterCoalescer`** (коалесинг требует его как анализ)
   и живёт дальше, пока его потребляет RA. `LiveVariables` считается ещё раньше
   (`TargetPassConfig.cpp:1503`) и нужен PHI-элиминации, two-address и др.
3. **Обёртки вокруг планировщика.** `GCNPassConfig::addOptimizedRegAlloc`
   (`AMDGPUTargetMachine.cpp:1785`) врезает `SIWholeQuadMode`, `SIOptimizeExecMaskingPreRA`,
   `SIFormMemoryClauses` **после** `MachineScheduler`, затем зовёт базовый
   `TargetPassConfig::addOptimizedRegAlloc()` (`:1819`), который и вызывает
   `addRegAssignAndRewriteOptimized()`.

---

## 2. Особенность AMDGPU: три фазы аллокации вместо одной

На большинстве целей `-regalloc` — один проход. AMDGPU так нельзя: у GCN **три разных
регистровых файла** с разной семантикой.

```
   ┌────────────┬───────────────────────────────┬─────────────────────────┐
   │ Класс      │ Что это                        │ Как спиллится           │
   ├────────────┼───────────────────────────────┼─────────────────────────┤
   │ SGPR       │ скалярные, одно значение на    │ в ПОЛОСУ VGPR           │
   │ (scalar)   │ всю волну (uniform)            │ (writelane/readlane)    │
   ├────────────┼───────────────────────────────┼─────────────────────────┤
   │ VGPR       │ векторные, своё значение в     │ в SCRATCH-память        │
   │ (vector)   │ каждой полосе (per-lane)       │ (scratch/buffer store)  │
   ├────────────┼───────────────────────────────┼─────────────────────────┤
   │ AGPR       │ аккумуляторы MFMA              │ обмен с VGPR / в scratch│
   │ (accum)    │                                │                         │
   └────────────┴───────────────────────────────┴─────────────────────────┘
```

Поэтому `addRegAssignAndRewriteOptimized` (`AMDGPUTargetMachine.cpp:1909`) выстраивает
последовательность из **отдельных greedy-проходов**:

```
 addRegAssignAndRewriteOptimized()                       AMDGPUTargetMachine.cpp:1909
 ────────────────────────────────
   GCNPreRALongBranchReg                                 :1913
   ┌─ ФАЗА SGPR ──────────────────────────────────────────────────┐
   │ createSGPRAllocPass(true) = greedy(onlyAllocateSGPRs)   :1915 │  :1830
   │ createVirtRegRewriter(false)   зафиксировать SGPR       :1921 │
   │ StackSlotColoring   уплотнить SGPR-spill слоты          :1926 │
   │ SILowerSGPRSpills   «PEI для SGPR»: SGPR → полосы VGPR   :1929 │
   └──────────────────────────────────────────────────────────────┘
   ┌─ ФАЗА WWM (whole-wave-mode) ─────────────────────────────────┐
   │ SIPreAllocateWWMRegs   преназначить физ-VGPR WWM-регам   :1932 │
   │ createWWMRegAllocPass(true) = greedy(WWM)                :1935 │
   │ SILowerWWMCopies                                         :1936 │
   │ createVirtRegRewriter(false)                            :1937 │
   │ AMDGPUReserveWWMRegs                                     :1938 │
   └──────────────────────────────────────────────────────────────┘
   ┌─ ФАЗА VGPR ──────────────────────────────────────────────────┐
   │ createVGPRAllocPass(true) = greedy(VGPR)                :1941 │  :1845
   │ addPreRewrite():  GCNNSAReassign + RewriteAGPRCopyMFMA  :1943 │  :1822
   │ VirtRegRewriter                                          :1944 │
   │ AMDGPUMarkLastScratchLoad                               :1946 │
   └──────────────────────────────────────────────────────────────┘
```

Почему такой порядок:

- **SGPR раньше VGPR**, потому что SGPR-спилл *создаёт* новые VGPR (полосы для writelane).
  Сначала надо узнать, сколько SGPR не влезло, выделить под них VGPR-полосы, и лишь потом
  раскрашивать VGPR с учётом этого давления.
- **`SILowerSGPRSpills` между фазами** — это «PEI для SGPR»: он материализует спилл-полосы
  до того, как обычный VGPR-аллокатор начнёт работу.
- **WWM в середине**, потому что whole-wave-mode регистры (в т.ч. полосы SGPR-спилла) должны
  получить физ-VGPR раньше обычных per-lane VGPR, иначе обычная раскраска их затрёт
  (`SIPreAllocateWWMRegs` через `LiveRegMatrix`, `rewriteRegs` — `SIPreAllocateWWMRegs.cpp:123`).
- **`addPreRewrite` перед последним rewrite** — здесь AGPR-переписывание MFMA
  (`AMDGPURewriteAGPRCopyMFMA`) и переупаковка NSA. Это тот самый слой, к которому относится
  стадия `RewriteMFMAForm` из документов про авто-тюнинг.

Каждый `createSGPR/VGPR/WWMAllocPass(Optimized=true)` — это `createGreedy…RegisterAllocator`
с фильтром класса регистров (`AMDGPUTargetMachine.cpp:1839-1856`). При `-O0` вместо greedy
берётся `RegAllocFast` (`:1842`, `:1857`).

---

## 3. Liveness-анализ до аллокатора

Аллокатору нужно точно знать, **где именно** (с точностью до позиции инструкции) каждый
виртуальный регистр «жив». Это делает `LiveIntervals`, стоящий на трёх китах: нумерация слотов
(`SlotIndexes`), представление интервала (`LiveInterval`/`VNInfo`), и построение из SSA
(`LiveIntervalCalc`/`LiveRangeCalc`).

### 3.1. `SlotIndexes` — нумерация инструкций

Каждой «настоящей» инструкции присваивается номер (базовый индекс), а конкретная позиция —
это базовый индекс плюс **2-битный под-слот** (`SlotIndexes.h:93`, `PointerIntPair`). Под-слотов
четыре (`enum Slot`, `SlotIndexes.h:69-91`):

```
   базовый номер инструкции N
        │
        ├── Slot_Block         (B) граница блока / точка PHI-def
        ├── Slot_EarlyClobber  (E) early-clobber use/def
        ├── Slot_Register      (r) обычный use/def регистра
        └── Slot_Dead          (d) точка «смерти» мёртвого def
```

Печатаются буквами из строки `"Berd"` (`SlotIndexes.cpp:295`). Полуоткрытость `[start,end)`
(конец **исключается**) позволяет def на `Slot_Register` и kill предыдущего значения на том же
номере не конфликтовать.

**Зачем зазоры между инструкциями.** Шаг нумерации — не 1, а `InstrDist = 4*Slot_Count = 16`
(`SlotIndexes.h:112-116`); в цикле нумерации индекс растёт на `InstrDist` на каждую инструкцию
(`SlotIndexes.cpp:103`) плюс пустая запись между блоками (`:111`). Пустое числовое пространство
между соседями нужно, чтобы **спилл и расщепление live-range могли вставлять новые инструкции
между существующими без полной перенумерации**: новая инструкция получает слот на середине
зазора — `dist = ((next-prev)/2) & ~3u` (`SlotIndexes.h:564-565`). Если зазор исчерпан
(`dist == 0`), делается локальная `renumberIndexes` (`:572-573`, `SlotIndexes.cpp:170-195`),
а при churn >20% — уплотнение `packIndexes()` (`:268-271`).

### 3.2. `LiveInterval` / `LiveRange` / сегменты / `VNInfo`

- **`VNInfo` (номер значения)** — `LiveInterval.h:54-89`. Пара `(unsigned id, SlotIndex def)`:
  идентификатор значения и точка, где оно определено. `isPHIDef()` = `def.isBlock()` (`:79`):
  PHI-значение «определено» на границе блока.
- **`LiveRange::Segment`** — `LiveInterval.h:171-207`. Полуоткрытый интервал `[start, end)` с
  указателем на `VNInfo *valno` — то есть «здесь жив, и это вот такое значение». `LiveRange` —
  упорядоченный вектор сегментов плюс список `valnos` (`:209-214`); в диапазоне могут быть дыры.
- **`LiveInterval`** — `LiveInterval.h:705`, наследник `LiveRange`, добавляет `Reg` (`:732`) и
  `float Weight` (`:733`) — тот самый вес спилла. Спиллабельность закодирована в весе:
  `isSpillable()` = `Weight != huge_valf` (`:844`), `markNotSpillable()` ставит `huge_valf` (`:847`).
- **`SubRange` — по-полосная liveness** (`LiveInterval.h:712-727`). Широкий/супер-регистр
  (например пара/кортеж VGPR или regmask-полосы) раскладывается на подсегменты, каждый с маской
  полос `LaneBitmask LaneMask` (`:715`). Это позволяет отслеживать жизнь **отдельных
  подрегистров** — критично для AMDGPU, где кортежи VGPR/AGPR и subreg-обращения повсеместны.
  Управление: `createSubRange` (`:810`), `refineSubRanges` (`:896`).

```
  vreg %10 (кортеж из 4 VGPR):

  main LiveRange:   [16 ───────────────── 96)   valno0
   SubRange lane0:  [16 ─── 48)  [80 ─ 96)       (sub0 жив тут)
   SubRange lane1:  [16 ─────────────── 96)      (sub1 жив дольше)
   ...
```

### 3.3. `LiveIntervals` — как строятся из SSA

Проход-драйвер: `LiveIntervals::analyze` (`LiveIntervals.cpp:159-181`) заводит
`LiveIntervalCalc` и вызывает `computeVirtRegs()` (`:171`), `computeRegMasks()`,
`computeLiveInRegUnits()`. `computeVirtRegs` (`:236-248`) обходит все виртуальные регистры,
пропускает те, у кого нет не-debug использования, и для каждого зовёт `computeVirtRegInterval`.
Физрегистрам при создании ставится `huge_valf` (неспиллабельны), vreg — `0.0` (`:222-225`).

Ядро — `LiveIntervalCalc::calculate` (`LiveIntervalCalc.cpp:41-103`), двухшаговое SSA-построение:

```
  ШАГ 1: минимальные «мёртвые» def-сегменты        LiveIntervalCalc.cpp:53-81
    для каждого операнда-определения Reg:
      createDeadDef на его getRegSlot(EarlyClobber)  — крошечный сегмент [def, def+)
      если трекаются subreg → маршрут в SubRange по маске полос (refineSubRanges)

  ШАГ 2: продлить до использований                 extendToUses  :134-194
    для каждого операнда-чтения Reg:
      вычислить SlotIndex использования
      extend(...) — «дорасти» интервал назад до достигающего def
    ОСОБЫЙ СЛУЧАЙ PHI (:171-175):
      операнд PHI используется не в самой PHI, а в КОНЦЕ соответствующего
      блока-предшественника → UseIdx = getMBBEndIdx(pred)
```

Собственно обратный поток данных — в `LiveRangeCalc` (`LiveRangeCalc.cpp`). `extend`
(`:85-108`) сначала пробует `extendInBlock` (дойти до def в том же блоке), иначе
`findReachingDefs` (`:189-326`) идёт BFS назад по предшественникам, собирая живущие-на-выходе
значения. Если достигающее значение единственное — сегменты вливаются сразу через
`LiveRangeUpdater`; если разные значения сходятся в блоке — блок кладётся в worklist `LiveIn`,
и `updateSSA` (`:330-433`) спускается по дереву доминаторов, **вставляя PHI-def там, где у
предшественников разные значения** (`needPHI`, `:351-388`). Это классический SSA-updater,
переиспользующий готовый `MachineDominatorTree`.

### 3.4. `LiveVariables` — старый анализ (кратко)

`LiveVariables` (`LiveVariables.cpp`, `VarInfo` — `LiveVariables.h:78-98`) на каждый vreg хранит
`AliveBlocks` (битвектор блоков, сквозь которые значение живёт целиком) и `Kills` (инструкции —
последние использования в своём блоке). Это представление **def + kills, без числового порядка**
инструкций и без сегментов/номеров значений. Оно дешевле и достаточно для PHI-элиминации,
two-address и т. п., но **не** годится для аллокатора: тому нужен точный ответ «жив ли %X в
слоте N и с каким именно значением», а это даёт только `LiveIntervals` c `SlotIndex`-порядком.

Различие в одну строку:

```
  LiveVariables : «в каких блоках жив + где убит»     (грубо, без порядка)
  LiveIntervals : [start,end) сегменты + VNInfo        (точно, с порядком слотов)
```

### 3.5. Оценка веса спилла (spill weight)

Вес интервала — это **оценка «насколько дорого его спиллить»**; чем больше вес, тем сильнее
аллокатор хочет удержать его в регистре. Считает `VirtRegAuxInfo::weightCalcHelper`
(`CalcSpillWeights.cpp:232-392`):

```
  1. Унаследовать неспиллабельность от родителя-сплита            :243-252
  2. Просуммировать частоты use/def:                              :279-336
        для каждой инструкции с Reg (кроме identity-copy):
            w_i = (isDef + isUse) * freq(block)      LiveIntervals.cpp:911-922
        блок-выход цикла, пишущий live-out, ×3       CalcSpillWeights.cpp:323-327
        TotalWeight += w_i
  3. Подсказки-копии (copy hints) слабо усиливают ×1.01           :330-359
  4. Нулевой/пустой интервал без reg-mask/mem-fold → не спиллить  :373-378
  5. РЕМАТ-дешёвый (все def рематериализуемы) → ×0.5              :380-385
  6. Масштаб класса регистров                                     :388-389
  7. НОРМАЛИЗАЦИЯ по размеру:                                     :391
        normalize(TotalWeight, size, NumInstr)
```

Формула нормализации (`CalcSpillWeights.h:34-42`):

```
                    сумма частот use/def
   weight  =  ─────────────────────────────────
               size + 25 * SlotIndex::InstrDist
```

Смысл констант: `25*InstrDist` в знаменателе гасит влияние искусственных зазоров `SlotIndexes`
на короткие интервалы. Короткий интервал получает вес ~ числу использований, длинный —
приближается к **плотности** использований. Итог:

```
   высокий вес   → спиллить дорого   → держать в регистре
   низкий вес    → дешёвый кандидат на спилл
   × 0.5 (ремат) → предпочтительный кандидат: пересчитать дешевле, чем спиллить/грузить
   huge_valf     → НИКОГДА не спиллить (физрегистры, помеченные интервалы)
```

---

## 4. Как работает регистровый аллокатор (greedy)

Все три фазы AMDGPU — это `RAGreedy` (`RegAllocGreedy.cpp`). Общая идея: обрабатывать
виртуальные регистры **по одному, в порядке убывания приоритета**, и для каждого пройти каскад
«присвоить → вытеснить → расщепить → спиллить», пока регистр не будет раскрашен или спиллен.

### 4.1. Очередь приоритетов и стадии `LiveRangeStage`

Все vreg лежат в приоритетной очереди. `enqueue` (`RegAllocGreedy.cpp:424`) при первом попадании
переводит `RS_New → RS_Assign` (`:431-434`) и кладёт пару `(priority, ~Reg.id())`; инверсия id —
тай-брейк, чтобы младшие номера шли раньше. Приоритет даёт
`DefaultPriorityAdvisor::getPriority` (`:443`):

```
   RS_Split-остатки        → приоритет = только размер (в самый низ)      :449-452
   «гигантские»/GlobalPrio → форсируется глобальная эвристика             :457-460
   исходные ЛОКАЛЬНЫЕ (1 MBB, RS_Assign) → приоритет = дистанция по инстр :463-475
   ГЛОБАЛЬНЫЕ/сплит-продукты → приоритет = размер с GlobalBit=1           :476-482
```

Нетто-эффект: **крупные/глобальные интервалы раскрашиваются раньше мелких/локальных**, а те —
раньше расщеплённых остатков. Крупные важнее «застолбить» первыми, пока свободных физрегистров
много.

Стадия `LiveRangeStage` (`RegAllocEvictionAdvisor.h:51-72`) — это «сколько попыток уже потрачено»:

```
   RS_New    созданный, ни разу не в очереди                         :53
   RS_Assign только присвоение+вытеснение, затем requeue как RS_Split :56
   RS_Split  пробовать расщепление, если присвоить нельзя            :59
   RS_Split2 более агрессивное расщепление с гарантией прогресса      :64
   RS_Spill  будет спиллен, расщеплять больше не пробуем             :67
   RS_Done   ничего больше сделать нельзя; не раскрасился → ошибка    :71
```

Важно: **`RS_Memory` в этом коде нет** — «в памяти» = `RS_Spill`/`RS_Done`. Стадии хранит
`ExtraRegInfo` (`RegAllocGreedy.h:66-135`, старт `Stage = RS_New` на `:69`).

### 4.2. Каскад `selectOrSplitImpl`

`selectOrSplit` (`:2325`) — тонкая обёртка над `selectOrSplitImpl` (`:2647`). Сам каскад:

```
  selectOrSplitImpl(VirtReg)                          RegAllocGreedy.cpp:2647
  ──────────────────────────
    (1) tryAssign  ── есть свободный физрег? ──────► ДА: вернуть его      :2656
                                                    (спец-случай CSR)     :2661
         │ нет
         ▼
    Stage = getStage(VirtReg)                                             :2676
    (2) если Stage != RS_Split:
          tryEvict ── вытеснить менее ценный интервал? ──► ДА: вернуть   :2684
         │ нет
         ▼
    (3) если Stage < RS_Split (первая встреча):
          НЕ расщеплять и НЕ спиллить — поднять до RS_Split, requeue      :2704-2708
          «дождаться второго круга, когда мелочь уже расставлена»
         │
         ▼
    (4) если Stage < RS_Spill и интервал непуст:
          trySplit ── расщепить интервал/помехи ──► успех: вернуть        :2711-2716
         │ нет
         ▼
    (5) если Stage >= RS_Done или !isSpillable:
          tryLastChanceRecoloring (последний шанс перекраски)             :2721-2723
         │
         ▼
    (6) spiller().spill(LRE, &Order)   ← СПИЛЛ                            :2730
         пометить все новые vreg как RS_Done                              :2731
```

Обрати внимание на шаг (3) — **отложенность**. Первый раз интервал не расщепляется и не
спиллится: его переводят в `RS_Split` и возвращают в очередь. Идея — сперва расставить все
мелкие интервалы, чтобы получить точную картину интерференции, вокруг которой потом расщеплять.

### 4.3. `tryAssign` — прямое присвоение

`tryAssign` (`:536`) идёт по `AllocationOrder` и для каждого кандидата-физрега спрашивает
`Matrix->checkInterference(VirtReg, PhysReg)` (`:543`). Первый физрег без интерференции берётся;
хинтованный (по copy-hint) возвращается сразу (`:544-548`). Интерференцию хранит `LiveRegMatrix`
(поюнитные `LiveIntervalUnion`), а результат — `InterferenceKind` (`LiveRegMatrix.h:84`):

```
   IK_Free    = 0   свободно, assign легален
   IK_VirtReg       мешает виртуальный регистр  → его можно вытеснить/перекрасить
   IK_RegUnit       мешает физический reg-unit  → нельзя
   IK_RegMask       мешает regmask (вызов)      → нельзя
```

Если найденный физрег дорог по cost-per-use, `tryAssign` ещё раз пробует `tryEvict` в поиске
более дешёвого варианта (`:578-587`).

### 4.4. `tryEvict` — вытеснение и каскады

`tryEvict` (`:719`) тонкий: спрашивает у эдвайзера кандидата на вытеснение
(`tryFindEvictionCandidate`) и, если нашёл, зовёт `evictInterference` (`:730`). Вся логика
«кого можно вытеснить» — в `canEvictInterferenceBasedOnCost`
(`RegAllocEvictionAdvisor.cpp:256`):

```
   - интерференция не чисто виртуальная (> IK_VirtReg)?        → нельзя         :260
   - интерферентов >= порога (~10)?                            → скорее всего
                                                                 один тяжёлый    :277-280
   - интерферент в FixedRegisters или RS_Done?                → нельзя          :290-295
   - КАСКАДЫ (анти-зацикливание):                                               :298-309
        Cascade == IntfCascade         → нельзя (тот же круг)
        Cascade <  IntfCascade         → нельзя, кроме «urgent», и это дорого
        Cascade >  IntfCascade         → можно (вытесняем более СТАРЫЙ каскад)
   - СРАВНЕНИЕ ВЕСОВ:                                                           :316-319
        Cost.MaxWeight = max(..., вес интерферента) + штрафы за сломанные хинты
        прервать, как только Cost >= MaxCost
```

**Каскады** — это номера, гарантирующие завершимость: вытеснять разрешено только *более старые*
каскады, поэтому цепочка «A вытесняет B, B вытесняет A…» невозможна. `evictInterference`
(`:621`) присваивает/читает каскад (`:627`), снимает назначение у интерферентов (`:650`) и
проверяет инвариант каскадов ассертом (`:651-655`).

### 4.5. Расщепление live-range (SplitKit)

Если ни присвоить, ни выгодно вытеснить нельзя — интервал **расщепляют** на куски, часть которых
можно раскрасить, а неудобную часть — спиллить в узком месте. `trySplit` (`:1972`) выбирает
подстратегию:

```
   локальный интервал (1 MBB):  tryLocalSplit  :1985 → tryInstructionSplit :1988
   глобальный интервал:         tryRegionSplit :2000 (если Stage<RS_Split2)
                                иначе fallback  tryBlockSplit :2006
```

`tryRegionSplit` (`:1203`) — стоимостной: строит компактный регион (`calcCompactRegion`),
считает стоимость кандидатов (`calculateRegionSplitCost`) против базовой стоимости спилла всего
интервала, а границы региона задаёт `addSplitConstraints` (`:743`), скармливая частоты блоков
`SpillPlacer` (хопфилдовское размещение спиллов: по каждому блоку — держать в регистре или
спиллить на границе).

Механику переписывания делает **SplitKit** (`SplitKit.h/.cpp`):

- **`SplitAnalysis`** (`SplitKit.h:96`) сканирует один `LiveInterval`, классифицируя каждый блок
  (live-in/kill, def/live-out, live-through) — `analyze`/`calcLiveBlockInfo`.
- **`SplitEditor`** (`SplitKit.h:263`) — мутатор, реально режущий машинный код и `LiveIntervals`.
  Протокол `openIntv → enterIntv*/useIntv*/leaveIntv* → finish` (`:254-261`). Он держит
  **комплемент-интервал 0** для всего, что не вынесли в новый интервал, и вставляет копии/reload
  только на границе региона. Режим `ComplementSpillMode` (`:281`) регулирует, сколько
  перекрытий допустимо ради минимизации вставленных COPY.

Результирующие подынтервалы возвращаются в очередь (как `RS_Split2`, затем `RS_Spill`), что
**гарантирует прогресс** к раскраске или спиллу.

---

## 5. Как происходит спилл

Когда каскад доходит до `spiller().spill(...)` (`RegAllocGreedy.cpp:2730`), в дело вступает
общий `InlineSpiller`, а затем — AMDGPU-специфичное превращение спилл-псевдо в реальные
инструкции.

### 5.1. Общий спиллер `InlineSpiller`

Вход — `InlineSpiller::spill(LiveRangeEdit&, AllocationOrder*)` (`InlineSpiller.cpp:1442`).
Порядок (`:1460-1465`):

```
   spill()
     ├─ collectRegsToSpill()   собрать основной vreg + «сниппет»-копии       :389
     ├─ reMaterializeAll()     СНАЧАЛА попытаться РЕМАТ (пересчёт)            :816
     └─ if (остались regs) spillAll()   иначе спилл вообще не нужен          :1393
```

**Ремат первым.** `reMaterializeFor` (`:656`) для каждого использования проверяет, читает ли оно
значение, находит определяющий `VNInfo`/`DefMI` в *исходном* интервале и, если def
рематериализуем (`canRematerializeAt`), **клонирует def прямо перед использованием** вместо
загрузки из памяти. Заодно пробует `foldMemoryOperand`, чтобы вплавить ремат в потребителя
(`:758`). Если значение полностью рематериализовано — старый def становится мёртвым и удаляется.

**Store после def, reload перед use.** Если ремат не покрыл всё, `spillAll` (`:1393`) сливает
спиллящиеся регистры в один стек-слот (`:1394-1414`), а `spillAroundUses` (`:1297`) обходит
каждое use/def:

```
   для каждого обращения к vreg:
     пробуем foldMemoryOperand(Ops)  — вплавить память прямо в инструкцию     :1360
     не вышло → завести свежий vreg                                          :1363
        перед use:  insertReload  (loadRegFromStackSlot)                     :1229/:1368
        после def:  insertSpill   (storeRegToStackSlot, сразу после def)     :1259/:1388
```

`insertSpill` ставит store на `std::next(MI)` (сразу после определения, `:1267-1272`);
для undef-def реального store нет (эмитится `KILL`).

**Свёртка и хойстинг.** `foldMemoryOperand` (`:988`) просит `TII.foldMemoryOperand` вплавить
обращение к памяти прямо в пользователя (нельзя для tied-операндов, `:1040`), убирая отдельную
загрузку/сохранение. Хойстинг спиллов: `hoistSpillInsideBB` (`:441`) поднимает store к def
внутри блока, а межблочный хойстинг откладывается в `postOptimization → hoistAllSpills`
(`:1471`, `:1794`), сливающий одинаковые спиллы к общим доминаторам.

**`LiveRangeEdit`** (`LiveRangeEdit.h`) — объект, который аллокатор передаёт спиллеру: он владеет
множеством новых vreg, помнит родительский `Original`-интервал, даёт
`createFrom`/`rematerializeAt`/`canRematerializeAt` и держит `LiveIntervals`/`VirtRegMap`
согласованными по мере мутаций (через колбэки-делегаты `LRE_*`).

### 5.2. AMDGPU: SGPR спиллятся в полосы VGPR

SGPR — скалярные, **скалярная память не умеет адресовать per-lane scratch**. Поэтому SGPR
сохраняют не в память, а в **отдельную полосу (lane) векторного регистра**: `V_WRITELANE_B32`
пишет один SGPR в одну полосу VGPR, `V_READLANE_B32` читает обратно (`SIRegisterInfo.cpp:2414`,
`:2450`). Одна VGPR-полоса хранит по одному SGPR на каждую волновую полосу.

Это делает `SILowerSGPRSpills` (`SILowerSGPRSpills.cpp:425`, «PEI для SGPR»):

```
   run()                                                     SILowerSGPRSpills.cpp:425
     spillCalleeSavedRegs()                                                  :436
     если spillSGPRToVGPR() и есть SGPR-спиллы:                              :453
       для каждого TII->isSGPRSpill(MI):                                     :470
         CSR-спиллы  → ФИЗИЧЕСКИЕ полосы VGPR (CFI статичен)                 :495-498
              allocateSGPRSpillToVGPRLane(..., SpillToPhysVGPRLane=true)
              eliminateSGPRToVGPRSpillFrameIndex(..., true)
         обычные     → ВИРТУАЛЬНЫЕ полосы VGPR                               :505-513
     IMPLICIT_DEF для каждого lane-VGPR, флаг WWM_REG                        :519-538
     determineRegsForWWMAllocation() маски WWM/не-WWM                        :546-553
     removeDeadFrameIndices() убрать мёртвые SGPR-спилл-слоты                :564
```

Бухгалтерия полос — в `SIMachineFunctionInfo::allocateSGPRSpillToVGPRLane`
(`SIMachineFunctionInfo.cpp:448`): раскладывает FI на `NumLanes = Size/4` полос, по одной на
двойное слово, а зарезервированные lane-VGPR держит в `SpillVGPRs`
(`getSGPRSpillVGPRs()`, `SIMachineFunctionInfo.h:688`).

### 5.3. AMDGPU: VGPR спиллятся в scratch-память

VGPR уже векторные, их спилл — это реальная память. Превращение спилл-псевдо в инструкции делает
`SIRegisterInfo::eliminateFrameIndex` (`SIRegisterInfo.cpp:2520`) — большой switch, работающий
**после RA** (по frame index'ам):

```
   eliminateFrameIndex(MI)                                   SIRegisterInfo.cpp:2520
     выбрать FrameReg (base/frame), проверить scratch RSRC зарезервирован    :2532-2538
     case SI_SPILL_V*_SAVE:                                                  :2676-2725
        Opc = flat-scratch? SCRATCH_STORE_DWORD_SADDR : BUFFER_STORE_DWORD_OFFSET :2705
        buildSpillLoadStore(..., Index, VGPR, FrameReg, offset, MMO)         :2715
        MFI->addToSpilledVGPRs(getNumSubRegsForSpillOp(...))   учёт спиллов  :2719
     case SI_SPILL_V*_RESTORE:  симметрично, SCRATCH_LOAD/BUFFER_LOAD        :2733-2813
```

Рабочая лошадка — `buildSpillLoadStore` (`:1570`): определяет класс/ширину, разбивает
слишком большие/невыровненные кортежи на под-спиллы, вычисляет scratch-смещение из
`ScratchOffsetReg`/`InstOffset` и эмитит выбранный `SCRATCH_*`/`BUFFER_*` на каждый элемент.
Scratch адресуется через дескриптор `MFI->getScratchRSrcReg()` плюс per-wave stack-pointer
(`FrameReg`).

SGPR-путь через `eliminateSGPRToVGPRSpillFrameIndex` (`:2467`) → `spillSGPR` (`:2152`) /
`restoreSGPR` (`:2313`) → `V_WRITELANE`/`V_READLANE`. Если полоса не выделена (`!SpillToVGPR`),
SGPR всё же уходит в память через `spillEmergencySGPR` (`:2393`).

### 5.4. Два уровня спилла и влияние на occupancy

```
   SGPR ──writelane──► полоса VGPR ──(если и её вытеснили)──► scratch-память
        дёшево (reg↔reg)                                        дорого (память)

   VGPR ─────────────────────────────────────────────────────► scratch-память
                                                                дорого (память)
```

- **VGPR-спилл дорог**: это реальный `SCRATCH_STORE`/`BUFFER_STORE` на каждую полосу, высокая
  латентность → крайняя мера.
- **SGPR-спилл дёшев**: `writelane`/`readlane` — это reg↔reg. Второй уровень (полоса VGPR → в
  память) включается лишь если сами spill-VGPR вытеснили.
- **Влияние на occupancy.** Любой лишний VGPR — и spill-полосы SGPR, и временные под VGPR-спилл —
  поднимают VGPR-давление, а оно снижает число волн на EU. `Occupancy` инициализируется из
  `ST.computeOccupancy(...)` (`SIMachineFunctionInfo.cpp:72`) и зажимается `limitOccupancy`
  (`:204-207`). Счётчик спилл-VGPR — `getNumSpilledVGPRs()` (`SIMachineFunctionInfo.h:1120`,
  прибавляется в `eliminateFrameIndex:2719`). **Поэтому аллокатор предпочитает упаковку SGPR в
  полосы добавлению новых VGPR** — и поэтому в документах про авто-тюнинг `getVGPRSpills`
  фигурирует как guardrail: рост спиллов = регресс occupancy/латентности.
- На MFMA-железе есть альтернатива scratch: `allocateVGPRSpillToAGPR`
  (`SIMachineFunctionInfo.cpp:495`) — спилл VGPR в AGPR вместо памяти.

### 5.5. WWM вокруг спилла

Полосы SGPR-спилла и прочие whole-wave-mode значения должны быть корректны **для всех полос,
включая неактивные**, иначе спилл затрёт данные выключенных lane. Этим заведуют два прохода
(см. фазу WWM в §2):

- **`SIPreAllocateWWMRegs`** (`SIPreAllocateWWMRegs.cpp:204`) — до обычной VGPR-раскраски
  преназначает физ-VGPR всем WWM-vreg через `LiveRegMatrix` и переписывает их (`rewriteRegs`,
  `:123`), чтобы обычный per-lane аллокатор их не трогал.
- **`SILowerWWMCopies`** (`SILowerWWMCopies.cpp:135`) — после RA превращает `WWM_COPY` обратно в
  `COPY`, обёрнутый в save/restore EXEC (`:160-162`), чтобы копия выполнилась для всех полос.

---

## 6. Связь с pre-RA планировщиком и авто-тюнингом

Почему этот документ важен для соседних заметок про coexec и авто-тюнинг:

```
   pre-RA MachineScheduler (coexec/GCNSchedStrategy)     — двигает ВИРТУАЛЬНЫЕ регистры,
        │                                                  оценивает давление через GCNRegPressure
        ▼
   RAGreedy (SGPR/WWM/VGPR)                              — превращает в ФИЗИЧЕСКИЕ,
        │  tryAssign→tryEvict→trySplit→spill               здесь и рождаются спиллы
        ▼
   RewriteMFMAForm / AGPR-rewrite (addPreRewrite)        — VGPR↔AGPR форма MFMA
        │
        ▼
   реальный occupancy = f(пик физ-VGPR + spill-VGPR)
```

- Планировщик **не видит** финальную раскраску: он лишь оценивает давление. Значит любая
  coexec-эвристика, поднимающая пик давления ради перекрытия окна MFMA, рискует **позже**
  обернуться спиллом или падением occupancy — а это уже решает RA, ниже по конвейеру.
- Именно поэтому guardrail'ы авто-тюнинга (`getVGPRSpills`, occupancy) в
  `amdgpu-backend-autotuning.ru.md` измеряют **пост-RA** результат, а не то, что думал планировщик.
- При жёстком пине occupancy из Triton/Gluon (`amdgpu-waves-per-eu=N,N`) регистровый бюджет
  фиксирован, и «обрыв occupancy» превращается в «спилл в фиксированном бюджете» — то есть в
  ровно тот путь §5, что описан здесь (`eliminateFrameIndex` → scratch).

---

## 7. Карта кода (якоря)

| Тема | Файл:строка |
|------|-------------|
| AMDGPU: 3-фазный RA | `AMDGPUTargetMachine.cpp:1909` `addRegAssignAndRewriteOptimized` |
| AMDGPU: обёртки pre-RA | `AMDGPUTargetMachine.cpp:1785` `addOptimizedRegAlloc` |
| AMDGPU: SGPR/VGPR/WWM alloc-пассы | `AMDGPUTargetMachine.cpp:1830,1845,1860` |
| AMDGPU: `-regalloc` запрещён | `AMDGPUTargetMachine.cpp:1879-1885` |
| Generic: scaffolding RA | `TargetPassConfig.cpp:1485` `addOptimizedRegAlloc` |
| SlotIndexes: под-слоты / InstrDist | `SlotIndexes.h:69-91`, `:112-116` |
| SlotIndexes: нумерация / вставка | `SlotIndexes.cpp:63-123`; `SlotIndexes.h:564-565` |
| LiveInterval: сегмент / VNInfo | `LiveInterval.h:171-207`, `:54-89` |
| LiveInterval: вес / spillable | `LiveInterval.h:733`, `:844-847` |
| LiveInterval: SubRange (полосы) | `LiveInterval.h:712-727`, `:896` |
| LiveIntervals: драйвер | `LiveIntervals.cpp:159-181`, `:236-248` |
| LiveIntervalCalc: 2 шага + PHI | `LiveIntervalCalc.cpp:41-103`, `:134-194` |
| LiveRangeCalc: обратный поток / PHI-def | `LiveRangeCalc.cpp:85-108`, `:330-433` |
| LiveVariables: VarInfo | `LiveVariables.h:78-98` |
| Spill weight: ядро | `CalcSpillWeights.cpp:232-392` |
| Spill weight: формула | `CalcSpillWeights.h:34-42`; `LiveIntervals.cpp:911-922` |
| Greedy: каскад | `RegAllocGreedy.cpp:2647` `selectOrSplitImpl` |
| Greedy: очередь / приоритет | `RegAllocGreedy.cpp:424`, `:443` |
| Greedy: стадии | `RegAllocEvictionAdvisor.h:51-72` |
| Greedy: tryAssign | `RegAllocGreedy.cpp:536`; `LiveRegMatrix.h:84,108` |
| Greedy: tryEvict / каскады | `RegAllocGreedy.cpp:719`, `:621`; `RegAllocEvictionAdvisor.cpp:256` |
| Greedy: split | `RegAllocGreedy.cpp:1972`, `:1203`; `SplitKit.h:96,263` |
| Spiller: вход / ремат | `InlineSpiller.cpp:1442`, `:656`, `:816` |
| Spiller: reload/spill/fold | `InlineSpiller.cpp:1229`, `:1259`, `:988` |
| AMDGPU: SGPR→полосы VGPR | `SILowerSGPRSpills.cpp:425`; `SIMachineFunctionInfo.cpp:448` |
| AMDGPU: VGPR→scratch | `SIRegisterInfo.cpp:2520`, `:1570` |
| AMDGPU: writelane/readlane | `SIRegisterInfo.cpp:2414`, `:2450` |
| AMDGPU: учёт спиллов / occupancy | `SIMachineFunctionInfo.h:1120`; `.cpp:72`, `:204-207` |
| AMDGPU: WWM | `SIPreAllocateWWMRegs.cpp:204`; `SILowerWWMCopies.cpp:135` |

---

## 8. Мини-глоссарий

- **pre-RA / post-RA** — до / после регистровой аллокации. До RA код оперирует бесконечным
  числом виртуальных регистров в (почти) SSA; после — физическими регистрами и стек-слотами.
- **liveness / live interval** — «время жизни» значения: от определения до последнего чтения.
  У LLVM это `[start,end)`-сегменты над номерами слотов.
- **SlotIndex** — числовая позиция инструкции с 2-битным под-слотом; зазоры между номерами дают
  место для вставки спилл/сплит-инструкций без перенумерации.
- **VNInfo** — «номер значения»: какое именно значение живёт в данном сегменте (важно на слияниях
  потоков управления и для PHI).
- **SubRange / LaneBitmask** — по-полосная liveness внутри кортежа регистров; позволяет
  отслеживать жизнь отдельных подрегистров.
- **spill weight** — оценка стоимости спилла интервала; выше вес → сильнее держать в регистре.
- **greedy allocator (`RAGreedy`)** — основной аллокатор LLVM: по одному vreg, каскад
  assign→evict→split→spill, очередь по приоритету.
- **eviction / cascade** — вытеснение уже назначенного интервала ради более ценного; каскады —
  номера, запрещающие циклические вытеснения (гарантируют завершимость).
- **live range splitting (SplitKit)** — разрезание интервала на части, чтобы удобные раскрасить,
  а неудобную часть спиллить в узком месте.
- **rematerialization (ремат)** — вместо спилл/reload заново вычислить дешёвый def перед
  использованием.
- **writelane / readlane** — запись/чтение одного скалярного значения в отдельную полосу
  векторного регистра; так AMDGPU спиллит SGPR (в VGPR, не в память).
- **scratch** — приватная память потока/волны, куда спиллятся VGPR (`SCRATCH_*`/`BUFFER_*`).
- **WWM (whole-wave-mode)** — режим, в котором инструкция работает на всех полосах, включая
  неактивные; требует особой раскраски spill-полос и копий.
- **occupancy** — число одновременно активных волн на EU; падает при росте VGPR-давления,
  поэтому каждый лишний spill-VGPR потенциально режет occupancy.
