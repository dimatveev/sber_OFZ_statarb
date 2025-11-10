# Арбитражная стратегия: логика донабора позиции (пирамидинг)

Ниже описана формальная логика динамического донабора позиции, используемая в `2nd_version/arb_backtest.py`.

## Обозначения

- $z_t$ — z‑score спреда в момент $t$.
- $z_{\text{enter}}$, $z_{\text{exit}}$ — базовые пороги входа и выхода.
- $\beta_t$ — скользящая бета (для расчёта спреда).
- $\text{pos\_dir}_t \in \{-1, 0, +1\}$ — направление позиции (short‑/flat‑/long‑spread).
- $\text{pos\_coef}_t \in \mathbb{R}$ — коэффициент размера позиции (масштаб), знак совпадает с направлением.
- $\text{max\_position\_coef} > 0$ — верхняя граница по модулю для $\text{pos\_coef}$.
- $\text{base\_position\_coef} > 0$ — стартовый размер на входе.
- $\text{enter\_k}, \text{exit\_k} \ge 0$ — чувствительность динамических порогов к текущему размеру.
- $\text{vol\_scale}_t \in [\text{floor}, \text{cap}]$ — множитель плеча по волатильности.

## Динамические пороги входа и выхода

Сначала нормируем текущий размер позиции:

$$
\text{pos\_fraction}_t \;=\; \min\!\left(1, \frac{\lvert \text{pos\_coef}_{t-1} \rvert}{\text{max\_position\_coef}}\right).
$$

Далее вычисляем эффективные пороги (нижняя защита для входа, верхняя отсечка отключена для облегчения донабора):

$$
z_{\text{enter}}^{\text{eff}}(t) \;=\; \max\!\bigl(z_{\text{enter}}^{\min},\; z_{\text{enter}} \cdot (1 + \text{enter\_k} \cdot \text{pos\_fraction}_t)\bigr),
$$
$$
z_{\text{exit}}^{\text{eff}}(t) \;=\; z_{\text{exit}} \cdot (1 + \text{exit\_k} \cdot \text{pos\_fraction}_t).
$$

Где $z_{\text{enter}}^{\min}$ — нижняя граница эффективного порога входа (в коде `enter_eff_min`). Верхняя отсечка намеренно убрана, чтобы не препятствовать донабору при больших $\lvert z_t \rvert$.

## Логика направления позиции (state‑machine)

- Вход: если $z_t > z_{\text{enter}}^{\text{eff}}(t) \Rightarrow \text{pos\_dir}_t = -1$ (short‑spread); если $z_t < -z_{\text{enter}}^{\text{eff}}(t) \Rightarrow \text{pos\_dir}_t = +1$ (long‑spread).
- Держим: пока выполняется условие выхода (см. ниже), направление не меняется.
- Выход: если $\lvert z_t \rvert < z_{\text{exit}}^{\text{eff}}(t)$ или $z_t$ сменил знак против направления позиции ⇒ $\text{pos\_dir}_t = 0$.
- Дополнительно: интрадей‑ограничения, вола‑халты, shock‑halt и PnL‑стопы могут принудительно выставлять flat.

## Пирамидинг (только донабор внутри позиции)

Внутри активной позиции ($\text{pos\_dir}_t \neq 0$) целевой модуль размера определяется от текущего $\lvert z_t \rvert$. Для линейного режима:

$$
m_t \;=\; \frac{\lvert z_t \rvert}{z_{\text{enter}}^{\text{eff}}(t)}, 
\qquad
\text{desired\_abs}_t \;=\; \min\!\bigl(\text{max\_position\_coef},\; \max(0, \text{base\_position\_coef} \cdot m_t)\bigr).
$$

Для ступенчатого режима используется лестница \( \{(\text{mult}_k, c_k)\} \) по \( m_t \) (см. `size_steps`).

Далее применяется правило “только донабор” — размер не уменьшается внутри текущего направления:

$$
\text{prev\_abs}_t \;=\; 
\begin{cases}
\lvert \text{pos\_coef}_{t-1} \rvert, & \text{если }\operatorname{sign}(\text{pos\_coef}_{t-1}) = \operatorname{sign}(\text{pos\_dir}_t),\\
0, & \text{иначе},
\end{cases}
\qquad
\text{final\_abs}_t \;=\; \max\!\bigl(\text{prev\_abs}_t,\; \text{desired\_abs}_t\bigr).
$$

Итоговый коэффициент размера:

$$
\text{pos\_coef}_t \;=\; \operatorname{sign}(\text{pos\_dir}_t) \cdot \text{final\_abs}_t.
$$

При выходе/халтах/стопах $\text{pos\_coef}_t = 0$.

## Связь с объёмами и лимитами

Рассчитываем валовую нагрузку на каждую ногу с учётом волатильности:

$$
\text{gross\_per\_side}_t \;=\; 0.5 \cdot \text{gross\_limit\_rub} \cdot \text{vol\_scale}_t,
$$

и переводим в количества:

$$
\text{sber\_qty}_t \;=\; \frac{\text{gross\_per\_side}_t}{\text{sber\_close}_t} \cdot \text{pos\_coef}_t, 
\qquad
\text{hedge\_qty}_t \;=\; \frac{\text{gross\_per\_side}_t}{\text{hedge\_close}_t} \cdot (-\text{pos\_coef}_t).
$$

Для корзины ОФЗ количества делятся по весам. Знак указывает на противоположность ног (рублёвая нейтральность).

## Отладка в `signals.csv`

В файл `2nd_version/output_*/signals.csv` экспортируются поля:
- `z`, `beta`, `z_enter_eff`, `z_exit_eff` — фактические пороги на каждом шаге,
- `pos_dir`, `pos_coef` — направление и масштаб позиции,
- цены, количества и компоненты PnL.

Это позволяет прозрачно видеть моменты входов/донаборов/выходов и причины (по сравнениям $\lvert z_t \rvert$ с $ z_{\text{enter}}^{\text{eff}}(t) $, $ z_{\text{exit}}^{\text{eff}}(t) $).

## Стоп‑лоссы и халты

### Shock‑halt по Сберу

Идея: при резком движении Сбера стратегия уходит в flat и остаётся там ещё некоторое время (cooldown).

Пусть окно шока — $L_{\text{shock}}$ минут, порог — $\theta_{\text{shock}} \in (0,1)$.

$$
s\_{\text{move}}(t) \;=\; \left| \frac{P\_{\text{SBER},t}}{P\_{\text{SBER},\,t-L\_{\text{shock}}}} - 1 \right|.
$$

Срабатывание:
$$
\text{shock}(t) \;=\; \mathbb{I}\bigl\{ s\_{\text{move}}(t) \;>\; \theta\_{\text{shock}} \bigr\},
$$
и далее маска расширяется на $L_{\text{cool}}$ минут (cooldown):
$$
\widetilde{\text{shock}}(t) \;=\; \max\_{0 \le k \le L\_{\text{cool}}} \text{shock}(t-k).
$$

Действие: при $\widetilde{\text{shock}}(t)=1$ выставляем flat:
$$
\text{pos\_dir}\_t = 0,\qquad \text{pos\_coef}\_t = 0.
$$

### Стоп по процентной cumulative drawdown

Идея: ограничить просадку относительно исторического максимума equity.

Сначала считаем предварительный PnL по текущей траектории позиций, затем:
$$
\text{equity}\_t \;=\; 1 \;+\; \frac{\text{PnL\_cum}\_t}{\text{gross\_limit\_rub}},
\qquad
\text{peak}\_t \;=\; \max\_{0 \le \tau \le t} \text{equity}\_\tau,
$$
$$
\text{dd}\_\%(t) \;=\; \frac{\text{equity}\_t}{\text{peak}\_t} \;-\; 1.
$$

Порог срабатывания $\theta\_{\text{dd}} \in (0,1)$ и cooldown $L\_{\text{dd}}$:
$$
\text{halt\_dd}(t) \;=\; \mathbb{I}\bigl\{ \text{dd}\_\%(t) \;<\; -\theta\_{\text{dd}} \bigr\},
\qquad
\widetilde{\text{halt\_dd}}(t) \;=\; \max\_{0 \le k \le L\_{\text{dd}}} \text{halt\_dd}(t-k).
$$

Действие: при $\widetilde{\text{halt\_dd}}(t)=1$ — flat:
$$
\text{pos\_dir}\_t = 0,\qquad \text{pos\_coef}\_t = 0.
$$

### Совместная работа (OR‑логика)

Обе защиты применяются одновременно; итоговая маска флата — объединение:
$$
\text{HALT}(t) \;=\; \max\!\bigl( \widetilde{\text{shock}}(t),\; \widetilde{\text{halt\_dd}}(t) \bigr).
$$
Если $\text{HALT}(t)=1$, позиция принудительно сводится к нулю на момент $t$ и в течение заданных cooldown‑периодов.


