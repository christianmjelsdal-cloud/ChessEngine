#pragma once
// TR_Globals.h  --  Global state declarations for NNUE Training GUI
#include "TR_Types.h"
#include "TR_Logger.h"

// ── Core state ────────────────────────────────────────────────────
extern AppState   g_st;
extern Config     g_cfg;
extern HINSTANCE  g_hInst;
extern UIHandles  g_ui;
extern std::thread g_worker;

// ── Preset system ─────────────────────────────────────────────────
extern std::vector<Preset> g_allPresets;
extern int g_currentPresetIdx;

// ── Benchmark ─────────────────────────────────────────────────────
extern TrainBenchResult g_trainBenchResult;
extern std::vector<BenchmarkResult> g_benchResults;
extern std::atomic<bool> g_benchRunning;
extern std::thread g_benchWorker;

// ── Graph ─────────────────────────────────────────────────────────
extern GraphState g_graph;
extern bool g_muteSounds;

// ── Config panel scroll ───────────────────────────────────────────
extern int g_cfgTotalH;
extern int g_cfgScrollY;

// ── Process management ────────────────────────────────────────────
extern ProcessState g_proc;

// ── Tooltip state ─────────────────────────────────────────────────
extern TipState g_tip;

// ── Log state ─────────────────────────────────────────────────────
extern size_t      g_logSent;
extern std::string g_lastLogText;

// ── File logger (structured .log output) ──────────────────────────
extern FileLogger  g_fileLog;
