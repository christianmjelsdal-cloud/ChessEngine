#pragma once
// TR_Logger.h -- Structured file logging for NNUE Training Runner
// Writes a machine-parseable log file to disk so an AI assistant can
// analyze training runs, find errors, track metrics, etc.
//
// Log format (each line):
//   [ISO8601] [LEVEL] [PHASE:GEN] message
//
// Structured event lines use a KEY=VALUE format after the header:
//   [ISO8601] [EVENT] [pipeline:0] PIPELINE_START gens=10 startGen=5 ...
//   [ISO8601] [METRIC] [training:3] EPOCH epoch=2 train_loss=0.094 val_loss=0.075 lr=0.001 acc=0.62
//   [ISO8601] [EVENT] [elo:3] ELO_RESULT elo=+45 wins=35 draws=42 losses=23
//   [ISO8601] [ERROR] [training:3] Process exited with code 1
//
// Levels: INFO, WARN, ERROR, EVENT, METRIC, CONFIG, CMD

#include <string>
#include <mutex>
#include <fstream>
#include <chrono>

enum class LogLevel {
    INFO,
    WARN,
    ERR,
    EVENT,
    METRIC,
    CONFIG,
    CMD
};

class FileLogger {
public:
    FileLogger() = default;
    ~FileLogger() { close(); }

    // Open a new log file. Called at pipeline start.
    // Filename: training_run_YYYYMMDD_HHMMSS.log in the given directory.
    // Returns the full path of the created log file.
    std::string open(const std::string& logDir);

    // Close the log file. Called at pipeline end.
    void close();

    // Is the logger currently writing to a file?
    bool isOpen() const;

    // ── Core write methods ────────────────────────────────────────

    // Write a general log line (mirrors pushLog output).
    void log(LogLevel level, const std::string& phase, int gen, const std::string& msg);

    // Convenience wrappers
    void info (const std::string& phase, int gen, const std::string& msg);
    void warn (const std::string& phase, int gen, const std::string& msg);
    void error(const std::string& phase, int gen, const std::string& msg);
    void cmd  (const std::string& phase, int gen, const std::string& msg);

    // ── Structured event writers ──────────────────────────────────

    // Log full config at pipeline start (key=value pairs on one line).
    void logConfig(const std::string& configDump);

    // Log a training metric data point.
    void logMetric(int gen, int epoch, double trainLoss, double valLoss,
                   double lr, double acc,
                   double openingLoss, double middlegameLoss, double endgameLoss,
                   bool hasVal, bool hasAcc, bool hasPhase);

    // Log an ELO result.
    void logElo(int gen, int elo, int wins, int draws, int losses,
                const std::string& matchType);  // "elo" or "swa"

    // Log a phase transition.
    void logPhaseStart(const std::string& phase, int gen, const std::string& detail);
    void logPhaseEnd  (const std::string& phase, int gen, bool success, double elapsedSec);

    // Log pipeline start/end.
    void logPipelineStart(int firstGen, int lastGen, int totalGens);
    void logPipelineEnd(bool stopped, double totalElapsedSec, int completedGens);

    // Log a generation summary.
    void logGenSummary(int gen, double trainLoss, double valLoss, int elo, double genElapsedSec);

    // Get the current log file path (empty if not open).
    std::string getPath() const;

private:
    std::mutex      m_mtx;
    std::ofstream   m_file;
    std::string     m_path;

    // Produce an ISO 8601 timestamp string.
    static std::string timestamp();

    // Convert LogLevel to string tag.
    static const char* levelTag(LogLevel level);

    // Write a raw line (already formatted). Must hold m_mtx.
    void writeLine(const std::string& line);
};
