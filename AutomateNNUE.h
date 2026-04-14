#pragma once
// AutomateNNUE.h — Isolated copy of the standard NNUE for Automate Chess.
//
// Automate Chess positions have exotic material distributions (e.g. 4 queens,
// 9 knights, no queen + extra minors) that the standard NNUE was not trained on.
// This isolated copy starts from standard weights and is retrained on Automate
// self-play positions so it learns to evaluate unusual armies correctly.
//
// Architecture: identical to NNUE (HalfKAv2, 40960 features) — the piece types
// and board geometry are the same; only the material distributions differ.
// Weights file: assets/automate_play_weights.bin
//
// The standard NNUE (assets/nnue_weights.bin) is NEVER modified by Automate training.

#include "NNUE.h"   // reuse all constants, feature encoding, and Network class

namespace AutomateNNUE {
    // Default weights path for the Automate Play NNUE
    inline const char* DEFAULT_WEIGHTS_PATH = "assets/automate_play_weights.bin";

    // AutomateNNUE uses the exact same Network class as standard NNUE.
    // It is a separate instance with separate weights — not a subclass.
    // Use NNUE::Network directly; the distinction is purely in which
    // weights file is loaded and which Board positions it is trained on.
    using Network = NNUE::Network;
}
