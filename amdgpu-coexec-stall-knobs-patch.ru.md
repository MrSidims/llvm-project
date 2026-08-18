# Текущий патч: настраиваемые «ручки» для `tryEffectiveStall` в CoExec-планировщике

Файл: `llvm/lib/Target/AMDGPU/AMDGPUCoExecSchedStrategy.cpp` (изменения в рабочем дереве,
не закоммичены).

Это Фаза 1 плана авто-тюнинга (см. `amdgpu-backend-autotuning.ru.md`): превратить жёстко
зашитую эвристику «эффективного столла» в набор настраиваемых, но **по умолчанию не
меняющих поведение** флагов, чтобы внешний перебор мог их подбирать на корпусе GEMM/attention.

---

## 1. Что было до патча

Шаг 4 стека эвристик (`tryEffectiveStall`) — центральное решение о перекрытии — считал
«эффективный столл» кандидата жёстко как максимум трёх компонент:

```cpp
// БЫЛО
Costs.Effective = std::max({Costs.Ready, Costs.Structural, Costs.Latency});
...
return tryLess(TryCosts.Effective, CandCosts.Effective, TryCand, Cand, Stall);
```

Где:

```
   Ready      = max(0, SU.TopReadyCycle - CurrCycle)   // операнды ещё не готовы
   Structural = getStructuralStallCycles(Zone, SU)     // нужный аппаратный юнит занят
   Latency    = Zone.getLatencyStallCycles(SU)         // латентность критического пути
```

Проблемы этой формы для тюнинга:

1. **Чистый `max`** — если кандидат-производитель окна (WMMA) даёт хоть один такт столла,
   он проигрывает нулевому «наполнителю». Нет способа взвесить компоненты по-разному.
2. **Нет допуска (slack)** — строгий `tryLess` решает даже при разнице в 1 такт, не оставляя
   поздним, более «умным» coexec-эвристикам (`tryCriticalResource`) шанса высказаться на
   почти-ничьих.
3. **Нет ни одного `cl::opt`** во всём файле — перебирать нечего.

---

## 2. Что добавляет патч

Пять скрытых флагов `-mllvm` + вспомогательная функция `combineEffectiveStall` + guard по
допуску. **Значения по умолчанию точно воспроизводят старое поведение.**

### 2.1. Флаги (`:24-58`)

| Флаг | Тип | По умолч. | Смысл |
|------|-----|-----------|-------|
| `-amdgpu-coexec-stall-weight-ready`   | unsigned | 1 | вес компоненты Ready |
| `-amdgpu-coexec-stall-weight-struct`  | unsigned | 1 | вес компоненты Structural |
| `-amdgpu-coexec-stall-weight-latency` | unsigned | 1 | вес компоненты Latency |
| `-amdgpu-coexec-stall-combine`        | max/sum  | max | как объединять взвешенные компоненты |
| `-amdgpu-coexec-stall-slack`          | unsigned | 0 | разницу ≤ slack тактов считать ничьёй |

### 2.2. Функция объединения (`:60-72`)

```cpp
static unsigned combineEffectiveStall(unsigned Ready, unsigned Structural,
                                      unsigned Latency) {
  unsigned R = Ready * CoExecStallWeightReady;
  unsigned S = Structural * CoExecStallWeightStruct;
  unsigned L = Latency * CoExecStallWeightLatency;
  switch (CoExecStallCombine) {
  case StallCombineMode::Max:
    return std::max({R, S, L});
  case StallCombineMode::Sum:
    return R + S + L;
  }
  llvm_unreachable("Unknown StallCombineMode");
}
```

При весах `(1,1,1)` и режиме `max` это тождественно старому `std::max({Ready,Structural,Latency})`.

### 2.3. Точка вызова + guard по допуску (`:734-765`)

```cpp
// СТАЛО
Costs.Effective =
    combineEffectiveStall(Costs.Ready, Costs.Structural, Costs.Latency);
...
if (CoExecStallSlack) {
  unsigned Diff = TryCosts.Effective > CandCosts.Effective
                      ? TryCosts.Effective - CandCosts.Effective
                      : CandCosts.Effective - TryCosts.Effective;
  if (Diff <= CoExecStallSlack)
    return false;            // почти-ничья: пропускаем к tryCriticalResource
}
return tryLess(TryCosts.Effective, CandCosts.Effective, TryCand, Cand, Stall);
```

`return false` из `tryEffectiveStall` означает «этот шаг не решает» — управление уходит на
шаг 5 (`sortHWUIResources` + `tryCriticalResource`, `:661-662`) и шаг 6
(`tryCriticalResourceDependency`, `:667`).

---

## 3. Блок-схема: поток решения до и после

```
                       tryEffectiveStall(Cand, TryCand, Zone)
                                       │
                    ┌──────────────────┴──────────────────┐
                    │  для TryCand и для Cand посчитать:    │
                    │   Ready, Structural, Latency          │
                    └──────────────────┬──────────────────┘
                                       │
              БЫЛО                     │                    СТАЛО
     ┌───────────────────┐            │           ┌────────────────────────────┐
     │ Eff = max(R,S,L)  │            │           │ R*=wReady  S*=wStruct       │
     └─────────┬─────────┘            │           │ L*=wLat                     │
               │                      │           │ Eff = (mode==max)           │
               │                      │           │        ? max(R,S,L)          │
               │                      │           │        : R+S+L               │
               │                      │           └──────────────┬─────────────┘
               │                      │                          │
               │                      │              ┌───────────┴───────────┐
               │                      │              │ slack > 0 и            │
               │                      │              │ |ΔEff| ≤ slack ?        │
               │                      │              └───────┬───────┬───────┘
               │                      │                   да │       │ нет
               │                      │                      ▼       │
               │                      │            return false ─────┤ (пропустить
               │                      │            (к шагу 5/6)      │  к crit-resource)
               ▼                      ▼                              ▼
        return tryLess(TryEff, CandEff, ..., Stall)  ◄───────────────┘
```

По умолчанию (`w=1,1,1`, `mode=max`, `slack=0`) обе ветви идентичны: guard пропускается
(`slack==0`), `combineEffectiveStall` возвращает тот же `max`, тот же `tryLess`.

---

## 4. Как «ручки» меняют расписание — примеры

Драйвер — существующий тест `coexec-sched-effective-stall.mir` (регион: две
`V_WMMA_SCALE_F32_16X16X128`, одна `GLOBAL_LOAD_DWORDX2`, одно `V_PK_ADD_F32`).

### 4.1. Значения по умолчанию → байт-в-байт как раньше

```
llc -mtriple=amdgpu12.50 -run-pass=machine-scheduler \
    -amdgpu-sched-strategy=coexec in.mir -o out_default.mir

llc ... -amdgpu-sched-strategy=coexec \
    -amdgpu-coexec-stall-weight-ready=1 \
    -amdgpu-coexec-stall-weight-struct=1 \
    -amdgpu-coexec-stall-weight-latency=1 \
    -amdgpu-coexec-stall-combine=max \
    -amdgpu-coexec-stall-slack=0 in.mir -o out_explicit.mir

diff out_default.mir out_explicit.mir      # пусто — идентичны
```

Существующий lit-тест проходит без изменений.

### 4.2. `-amdgpu-coexec-stall-weight-struct=0` → загрузка теряет приоритет окна

Обнуляя вес структурного столла, мы «ослепляем» шаг 4 к занятости матричного движка. При
сравнении второй WMMA (`Structural≈15`) с загрузкой (`Structural=0`) эффективный столл WMMA
падает до 0 — шаг 4 больше не двигает загрузку в окно, и порядок относительно WMMA меняется:

```
Effective(WMMA)  = max(Ready*1, 15*0, Latency*1) = max(0,0,0) = 0
Effective(LOAD)  = max(0, 0, 0)                   = 0
→ ничья на шаге 4, решает шаг 5/NodeOrder → GLOBAL_LOAD едет иначе, чем при default
```

### 4.3. `-amdgpu-coexec-stall-slack=20` → почти-ничьи уходят к crit-resource

Большой допуск заставляет `tryEffectiveStall` возвращать `false` на любой разнице ≤ 20
тактов, отдавая решение `tryCriticalResource`/`tryCriticalResourceDependency`. Это меняет
позицию `GLOBAL_LOAD_DWORDX2` относительно двух WMMA — то есть демонстрирует, что поздние
coexec-эвристики теперь получают голос там, где раньше строгий `tryLess` их перекрывал.

Оба флага **по отдельности** переставляют расписание относительно default — значит «ручки»
реально работают (bite), а не декоративны.

---

## 5. Что проверено

1. **Регистрация**: все 5 флагов видны в `llc --help-hidden`.
2. **Сохранение поведения по умолчанию**: (а) существующий lit-тест
   `coexec-sched-effective-stall.mir` проходит без изменений; (б) явные значения-по-умолчанию
   дают вывод, байт-в-байт совпадающий с прогоном без флагов.
3. **«Ручки» кусают**: `-amdgpu-coexec-stall-weight-struct=0` и
   `-amdgpu-coexec-stall-slack=20` каждый по отдельности переставляют `GLOBAL_LOAD_DWORDX2`
   относительно двух WMMA.
4. **Сборка**: чисто компилируется (объектник
   `AMDGPUCoExecSchedStrategy.cpp.o`), сборка в `/home/mrsidims/LLVM-work/build`.

---

## 6. Зачем это нужно для авто-тюнинга

`tryEffectiveStall` — самая нагруженная одиночная точка coexec-планировщика: именно она
решает, встанет ли независимая работа в окно матричной операции. Раньше её поведение было
единственным (чистый `max`, строгий порядок). Теперь это **пространство поиска**:

```
   ось weight-struct   : насколько «видна» занятость юнита  (0 → игнор, >1 → усилить)
   ось weight-ready    : насколько важна готовность операндов
   ось weight-latency  : насколько важна латентность критич. пути
   ось combine         : max (worst-case такт) vs sum (совокупная стоимость)
   ось slack           : сколько «почти-ничьих» отдать умным crit-resource эвристикам
```

Внешний перебор (OpenTuner / CompilerGym-обёртка вокруг `llc`/`clang`), меряя occupancy и
такты на корпусе GEMM, ищет по этим осям лучшую точку — не трогая дефолтную сборку, так как
все флаги `cl::init` восстанавливают исходную эвристику. Это и есть «дешёвая» Фаза 1:
доказать наличие запаса и собрать размеченный датасет для более тяжёлых фаз (частотно-зависимые
эвристики, затем опциональный MLGO-советник).

Естественное продолжение той же фазы — вынести в `cl::opt` коэффициенты `sortHWUIResources`
(`:304-323`, предварительно разобравшись с багом size-компаратора на `:315-317`), битовую
маску `ProducesCoexecWindow` (`:268-270`) и множитель спилла ×2 в стадии
`RewriteMFMAForm` (`GCNSchedStrategy.cpp:2516`).

---

## 7. Полная сводка изменений (diff по смыслу)

| Место | Изменение |
|-------|-----------|
| `:16` | `#include "llvm/Support/CommandLine.h"` |
| `:24` | `enum class StallCombineMode { Max, Sum };` |
| `:26-58` | 5 `cl::opt` флагов |
| `:60-72` | `combineEffectiveStall(Ready,Structural,Latency)` |
| `:740-741` | `Costs.Effective = combineEffectiveStall(...)` вместо `std::max({...})` |
| `:757-763` | guard по `CoExecStallSlack` перед финальным `tryLess` |

Инвариант всего патча: **при значениях по умолчанию — ни одного изменения в
сгенерированном коде**.
