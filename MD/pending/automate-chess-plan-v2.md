# Automate Chess Variant — Architecture Plan

## Exact Rules (from chess.com)

- Each player has **35 points** to spend on pieces
- **Piece costs**: Queen=7, Rook=4, Knight=3, Bishop=3, Pawn=1, King=free
- **Setup is turn-based**: White places one piece, then Black places one piece, alternating
- **Pawns placed first**: each side must place at least 6 pawns before placing any other piece. Pawns go on 2nd or 3rd rank only, max 2 pawns per file
- **Pieces placed second** (Q/R/N/B), on 1st or 2nd rank only — only allowed after that side has placed their 6 mandatory pawns
- **King placed last**; if placed in check, you lose instantly
- Game starts when **Black places their King** (the final placement)
- After setup, the game is played normally — humans move (or bots using the Automate Play NNUE)
- Standard chess rules apply during the play phase (just with non-standard armies)

### Turn-Based Placement Example

```
White places pawn on e2      (White pawns: 1/6 mandatory)
Black places pawn on e7      (Black pawns: 1/6 mandatory)
White places pawn on d2      (White pawns: 2/6 mandatory)
Black places pawn on d7      (Black pawns: 2/6 mandatory)
...                          (continues alternating)
White places pawn on a3      (White pawns: 6/6 — mandatory met, can now place pieces or more pawns)
Black places pawn on h7      (Black pawns: 6/6 — mandatory met)
White places knight on g1    (White has met pawn minimum, pieces now allowed)
Black places pawn on b7      (Black chooses to place more pawns — still allowed)
White places queen on d1     
Black places bishop on c8    
...                          (continues alternating until budgets spent)
White places king on e1      (White done — king placed last)
Black places king on e8      (Black done — game starts!)
```

Key constraints per turn:
- A side **cannot** place a piece (Q/R/N/B) or king until they have placed at least 6 pawns
- A side **can** place additional pawns (beyond 6) even after meeting the minimum
- King must be the **very last** piece placed by each side
- Both sides can see each other's placements in real-time

---

## Game Modes in ChessEngine.exe

| Mode | White Setup | Black Setup | Play Phase |
|---|---|---|---|
| **Human vs Bot** | Human places pieces manually | Setup NN decides placement (reacts to human's visible pieces) | Human plays White, Automate Play NNUE plays Black |
| **Bot vs Bot** | Setup NN decides placement | Setup NN decides placement (sees White's pieces) | Automate Play NNUE plays both sides |
| **Human vs Human** | Human places manually | Human places manually | Humans play both sides |

In **Human vs Bot**, the human places their own army and then plays as White; the bot uses the Setup NN for its army and then the Automate Play NNUE for its moves. In **Bot vs Bot**, both sides are fully automated. In **Human vs Human**, both players place and play manually. The play phase follows standard chess rules — just with non-standard armies.

---

## Architecture: Three Separate NNs

### 1. Setup NN (new, small) — handles piece placement
- Learns optimal army composition and placement strategy
- Observes opponent's placements in real-time and adapts
- Only active during the setup phase
- Used by the bot in Human vs Bot, and by both sides in Bot vs Bot

### 2. Automate Play NNUE (isolated copy of existing NNUE, retrained) — plays Automate games
- Starts as a copy of your standard NNUE weights
- Retrained on Automate-specific positions (unusual material: 4 queens, 9 knights, extra rooks, etc.)
- Separate weights file: `assets/automate_play_weights.bin`
- The standard NNUE was trained on normal chess positions and may misevaluate positions with exotic material distributions — this copy learns to handle them properly
- Used as the bot's engine during the play phase (Human vs Bot and Bot vs Bot modes)
- Also used during training to play out games and generate training signal for the Setup NN

### 3. Standard Play NNUE (existing, untouched) — plays normal chess & duck chess
- Your existing NNUE, completely unchanged
- Used for standard chess and duck chess as before
- Never affected by Automate training

**Why three NNs:**
- The setup phase is fundamentally different from play (constructing a position vs evaluating one) — needs its own network
- Automate positions have unusual material that the standard NNUE wasn't trained on (e.g., 4 queens, 9 knights, no queen + extra minor pieces) — an isolated copy can be retrained on these positions without degrading standard chess play
- Clean separation: standard chess weights stay pristine, Automate play learns exotic positions, setup NN learns placement strategy
- Starting the Automate Play NNUE from a copy of existing weights gives it a huge head start — it already understands piece activity, king safety, etc., and just needs to adapt to unusual material counts

### Weight Files

| NN | Weights File | Training |
|---|---|---|
| Standard NNUE | `assets/weights.bin` | Standard self-play (unchanged) |
| Automate Play NNUE | `assets/automate_play_weights.bin` | Self-play from Automate positions |
| Duck Chess NNUE | `assets/duck_weights.bin` | Duck chess self-play |
| Setup NN | `assets/automate_setup_weights.bin` | Policy gradient from setup outcomes |

---

## Setup NN Design

### The Setup as a Sequential Decision Problem

Setup is **turn-based** — White and Black alternate placing one piece at a time. Each side follows a strict phase order:
1. **Pawn phase**: must place at least 6 pawns (ranks 2-3, max 2 per file) before anything else
2. **Piece phase**: after 6+ pawns placed, can place Q/R/N/B (ranks 1-2) or additional pawns
3. **King phase**: king is placed last (must not be in check)

Each placement is one action. The NN makes one decision per turn, then waits for the opponent's placement before deciding the next one. This means the NN sees the opponent's latest placement as new input each turn.

### Input Features

Since placement is turn-based and the NN sees the board after each opponent placement:

```
Board state:          768 features (standard NNUE encoding — all pieces placed so far by BOTH sides)
My budget remaining:    1 feature  (my remaining points / 35)
My phase:               3 features (one-hot: placing_pawns, placing_pieces, placing_king)
My pawns placed:        1 feature  (count / 10)
My min pawns remaining: 1 feature  ((6 - my_pawns_placed) / 6, clamped to 0)
Opp budget remaining:   1 feature  (opponent remaining points / 35)
Opp pawns placed:       1 feature  (opponent pawn count / 10)
Opp min pawns remaining:1 feature  ((6 - opp_pawns_placed) / 6, clamped to 0)
---
Total:              ~777 features
```

Note: since the board encoding already includes both sides' pieces, we don't need a separate "opponent board state" — it's all in one board. The meta features track budget and phase info for both sides.

Or a slimmer version using only the relevant ranks:
```
All pieces on ranks 1-3:      12 types (6 per color) x 24 squares = 288 features
Budget + meta:                ~10 features
---
Total:                        ~298 features
```

### Output

**Policy head** — probability over legal placements:
- During pawn phase: 16 squares (ranks 2-3, 8 files) filtered by 2-per-file rule
- During piece phase: (piece_type, square) where piece_type in {Q,R,N,B} and square on ranks 1-2 = 4 x 16 = 64 options, filtered by budget
- During king phase: 16 squares (ranks 1-2), filtered by not-in-check
- Plus "done placing pawns" / "done placing pieces" action to transition phases

**Value head** — expected game outcome [-1, 1]

### Network Size

```
Input (~298 or ~777 features)
  -> Linear(input, 256) + ReLU
  -> Linear(256, 128) + ReLU
  -> Policy head: Linear(128, 97) + softmax
      (16 pawn squares + 64 piece placements + 16 king squares + 1 "done" action)
  -> Value head: Linear(128, 1) + tanh
```

Small network — setup decisions don't need the depth of a chess evaluation network.

---

## Training Pipeline

### Self-Play Loop

```
for each training game:
    1. Turn-based setup (alternating White/Black):
       while setup not complete:
           - Current side's Setup NN observes full board state (own + opponent pieces)
           - NN outputs policy over legal placements for this turn
           - Sample action (with temperature/noise for exploration)
           - Record (state, action, side) tuple
           - Place the piece on the board
           - Switch to other side
       
       Turn order: W places, B places, W places, B places, ...
       Each side must place 6 pawns before pieces, king last.

    2. Both kings placed -> position is set

    3. Automate Play NNUE plays White vs Black from this position
       - Use a reasonable search depth (e.g., depth 12-16)
       - Play until checkmate, stalemate, or draw
       - Important: this is the isolated Automate NNUE, not the standard one

    4. Record game result (1.0 = white win, 0.0 = black win, 0.5 = draw)

    5. For each (state, action, side) tuple from step 1:
       - Store (state, action, game_result_for_that_side) as training data
```

### Training the Setup NN

```
Loss = policy_loss + lambda * value_loss

policy_loss = -log(pi(action | state)) * advantage
  where advantage = game_result - V(state)  (REINFORCE with baseline)

value_loss = (V(state) - game_result)^2

Alternative: use MCTS during self-play to get improved policy targets,
then train via cross-entropy (AlphaZero style). More complex but stronger.
```

### Simpler Alternative: Evolution / Tournament Selection

If policy gradient training feels too complex:

1. Generate N random setup strategies (parameterized by piece selection + placement heuristics)
2. Run a round-robin tournament (engine plays each matchup)
3. Keep the top performers, mutate them, repeat
4. After convergence, train a small NN to imitate the best strategies

This is less elegant but much easier to implement and debug.

---

## Bootstrapping: Baseline Bot (No New NN)

Before training a Setup NN, you can get a working Automate bot using the Automate Play NNUE (or standard NNUE initially):

1. Generate candidate setups using heuristics:
   - Common templates (8 pawns + 9 knights, 6 pawns + bishops + queen, etc.)
   - Random variations within budget constraints
2. Use the Automate Play NNUE to **evaluate** each candidate position statically
3. Pick the setup with the best eval

**Pros**: No Setup NN needed, working bot immediately
**Cons**: Static eval may miss setup-specific strategy (e.g., piece synergy potential). Also doesn't react to opponent's setup.

This serves as a useful baseline to compare against once the Setup NN is trained.

---

## Implementation Plan

### Phase 1: Rules & UI
- Add `ChessVariant::Automate` to enum
- Implement setup phase UI in VisualGame:
  - Budget display (35 | 35)
  - **Turn-based placement**: White places one piece, then Black, alternating
  - Enforce per-side phase order: 6 mandatory pawns (ranks 2-3, max 2/file) before pieces (ranks 1-2), king last
  - Show piece costs in palette
  - Show whose turn it is to place
  - Show per-side pawn count and whether minimum met
  - King-in-check = instant loss
- Support all three game modes:
  - **Human vs Bot**: human places manually, bot uses Setup NN (or heuristic baseline) to decide its own placement
  - **Bot vs Bot**: both sides use Setup NN for placement
  - **Human vs Human**: both place manually
- After both sides place king, game transitions to normal play phase (human moves or bot moves depending on mode)

### Phase 2: Baseline Bot (no new NN)
- Generate candidate setups via heuristics + NNUE evaluation
- Bot picks highest-eval setup from candidates
- This gives you a working Automate bot immediately

### Phase 3: Setup NN
- Create `AutomateSetupNN.h/cpp` — small policy+value network
- Feature extraction for setup state (own board + opponent board + budget)
- Forward pass for policy + value
- Weight save/load (`assets/automate_setup_weights.bin`)

### Phase 4: Training Pipeline
- **Automate Play NNUE training**:
  - Initialize from a copy of standard NNUE weights
  - Generate self-play games from Automate positions (random/heuristic setups)
  - Train on these positions so the NNUE learns to evaluate exotic material correctly
  - Separate weights file: `assets/automate_play_weights.bin`
- **Setup NN training**:
  - Self-play data generation (Setup NN places, Automate Play NNUE plays game out)
  - Training loop (policy gradient + value regression)
- Add "Automate Chess" option to TrainingRunner variant switcher (trains both Setup NN and Automate Play NNUE)

### Phase 5: Refinement
- MCTS over setup decisions for stronger play
- Opponent modeling (adapt setup based on what opponent is placing)
- Opening book of known strong setups
