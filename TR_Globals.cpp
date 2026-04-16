// TR_Globals.cpp  --  Global variable definitions
#include "TR_Types.h"
#include "TR_Globals.h"

AppState   g_st;
Config     g_cfg;
HINSTANCE  g_hInst    = nullptr;
UIHandles  g_ui;
std::thread g_worker;

std::vector<Preset> g_allPresets;
int g_currentPresetIdx = 1;

TrainBenchResult g_trainBenchResult;
std::vector<BenchmarkResult> g_benchResults;
std::atomic<bool> g_benchRunning{false};
std::thread g_benchWorker;

GraphState g_graph;
bool g_muteSounds = false;

int g_cfgTotalH  = 0;
int g_cfgScrollY = 0;

ProcessState g_proc;

TipState g_tip;

size_t      g_logSent = 0;
std::string g_lastLogText;

FileLogger  g_fileLog;
