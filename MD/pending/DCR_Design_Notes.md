# Dynamic Convergence Reversal (DCR) — Design Notes

## Concept

DCR is an adaptive training scheduler for NNUE chess engine training that monitors per-phase loss curves (Opening, Middlegame, Endgame) and automatically adjusts training configuration to prevent any single phase from diverging too far from the others.

Three training modes serve as anchors:
- **Opening Mode** — based on the "Opening Training" preset (high self-play ratio, more games, moderate depth)
- **Middlegame Mode** — based on the standard/regular training preset (balanced settings)
- **Endgame Mode** — based on the "Mating Training" preset (deep search, high mate boost, more drawn/endgame positions)

When DCR detects that one phase's loss is diverging from the others, it shifts training toward the mode that corrects the imbalance.

---

## Key Design Challenges

### 1. Oscillation Risk
Hard-switching between modes can cause a ping-pong effect where improving one phase degrades another, leading to no net progress.

**Mitigations:**
- Cooldown period — enforce a minimum number of generations before allowing another switch
- Hysteresis — require a sustained divergence signal before switching, and don't switch back prematurely
- Prefer blended transitions over hard switches

### 2. Defining "Divergence"
The three phases have naturally different loss scales (endgames are simpler, openings are more ambiguous), so raw loss comparison is misleading.

**Better metrics:**
- Relative rate of change — is one phase's loss rising while others fall?
- Normalized divergence — % deviation from each phase's own rolling average
- Configurable divergence threshold to trigger intervention

### 3. Mode Effectiveness
Changing self-play settings (depth, game count, draw ratio) only shifts the *data distribution* for future generations. The training step itself sees a mix of all positions. This means:
- Mode switches take multiple generations to meaningfully affect training
- Directly weighting the loss function per phase would act faster
- Filtering or oversampling training data by phase is another fast-acting lever

---

## Recommended Approach: Hybrid Continuous Blending

Rather than discrete hard switches between three modes, a smoother system:

1. **Keep the three presets as anchor points** defining the configuration extremes
2. **Interpolate between them** based on measured divergence — blend settings proportionally instead of hard-switching
3. **Primary lever: per-phase loss weighting** — the fastest-acting knob, takes effect within the same generation. Formula example:
   ```
   weight_phase = (phase_loss / average_loss) ^ alpha
   ```
   Phases that are falling behind get amplified training signal.
4. **Secondary lever: data composition** — gradually adjust draw ratio, self-play depth, and game count toward the lagging phase's preferred settings
5. **Divergence metric**: relative rate of improvement per phase over a rolling window of N generations

This achieves automatic convergence correction without the instability of hard mode switching.

---

## Open Questions

- **Where should DCR logic live?** The Python training script (for loss weighting), the C++ GUI (for config switching), or both?
- **Discrete switching vs. continuous blending?** Blending is safer but harder to implement; switching is simpler but riskier
- **How aggressive should correction be?** Too gentle and divergence persists; too strong and you get oscillation
- **Should there be a "free exploration" phase early in training** before DCR activates, so the network establishes a baseline across all phases first?
