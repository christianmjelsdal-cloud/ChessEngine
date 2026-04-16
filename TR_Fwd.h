#pragma once
// TR_Fwd.h  --  Forward declarations for cross-file calls
#include "TR_Types.h"

// ── Config (TR_Config.cpp) ────────────────────────────────────────
Config ReadConfig();
std::string ValidateConfig(const Config& cfg);  // FIX 10: returns "" if valid, or error message
void ApplyPreset(int idx);
void SavePresetAs();
void DeleteCurrentPreset();
void PopulatePresetCombo();
void HardcodedDefaults();
void InitBuiltinPresets();
void LoadCustomPresets();
void SaveCustomPresets();
void SaveCalibration();
void LoadCalibration();
void SaveDefaultPresets();
int findLatestGen(const std::string& dataDir);
int findBestGen();
int findBestGenFor(const std::string& dataDir);
void saveGenStat(int gen, double bestValLoss, const std::string& dataDir);
std::vector<TrainPoint> loadGraphData(const std::string& dataDir);
void saveGraphData(const std::string& dataDir, const std::vector<TrainPoint>& pts);
void appendGraphPoint(const std::string& dataDir, const TrainPoint& p);

// ── Pipeline (TR_Pipeline.cpp) ────────────────────────────────────
void PipelineThread(Config cfg);

// ── Graph (TR_Graph.cpp) ──────────────────────────────────────────
LRESULT CALLBACK GraphProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp);
void SaveGraphPng(const std::string& dataDir);

// ── Benchmark (TR_Benchmark.cpp) ──────────────────────────────────
void BenchmarkThread(Config cfg);
void LoadBenchmarkResults();
void SaveBenchmarkResults();
void RecalibrateTimePresets();

// ── Process (TR_Process.cpp) ──────────────────────────────────────
bool RunProc(const std::wstring& cmd, const std::string& dir,
             std::function<void(const std::string&)> cb,
             std::atomic<bool>& stop);
void SuspendOrTerminateActive();
DWORD GetValidatedActivePid();
void SuspendProcessThreads(DWORD pid);
void ResumeProcessThreads(DWORD pid);

// ── UI (TR_UI.cpp) ────────────────────────────────────────────────
void BuildConfigPane(HWND pane, int PW);
void FlushLog();
void UpdateCfgScroll(HWND hw);
void ScrollCfgTo(HWND hw, int newPos);
LRESULT CALLBACK PanelProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp,
                           UINT_PTR uIdSubclass, DWORD_PTR dwRefData);
LogColor ClassifyLogLine(const std::string& line);
LRESULT CALLBACK TipWndProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp);

// ── Path validation (inline, used by pipeline) ───────────────────
inline void ValidatePathsForInjection(const std::string& dataDir, const std::string& exeName) {
    for (const char* path : {dataDir.c_str(), exeName.c_str()}) {
        for (char c : {'"', '&', '|', '>', '<'}) {
            if (strchr(path, c)) {
                throw std::runtime_error("Invalid path character");
            }
        }
    }
}
