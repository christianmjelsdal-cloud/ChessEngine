// TR_Logger.cpp -- Structured file logging implementation
#include "TR_Logger.h"
#include <sstream>
#include <iomanip>
#include <ctime>
#include <filesystem>

namespace fs = std::filesystem;

// ── Helpers ───────────────────────────────────────────────────────

std::string FileLogger::timestamp() {
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()) % 1000;
    std::tm tm{};
#ifdef _MSC_VER
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S")
       << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

const char* FileLogger::levelTag(LogLevel level) {
    switch (level) {
        case LogLevel::INFO:   return "INFO";
        case LogLevel::WARN:   return "WARN";
        case LogLevel::ERR:    return "ERROR";
        case LogLevel::EVENT:  return "EVENT";
        case LogLevel::METRIC: return "METRIC";
        case LogLevel::CONFIG: return "CONFIG";
        case LogLevel::CMD:    return "CMD";
    }
    return "INFO";
}

void FileLogger::writeLine(const std::string& line) {
    // Caller must hold m_mtx
    if (m_file.is_open()) {
        m_file << line << '\n';
        m_file.flush();
    }
}

// ── Open / Close ──────────────────────────────────────────────────

std::string FileLogger::open(const std::string& logDir) {
    std::lock_guard<std::mutex> lk(m_mtx);
    if (m_file.is_open()) m_file.close();

    // Ensure the logs directory exists
    fs::path dir = fs::path(logDir) / "logs";
    fs::create_directories(dir);

    // Generate filename: training_run_YYYYMMDD_HHMMSS.log
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _MSC_VER
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    std::ostringstream fname;
    fname << "training_run_"
          << std::put_time(&tm, "%Y%m%d_%H%M%S")
          << ".log";

    fs::path fullPath = dir / fname.str();
    m_path = fullPath.string();
    m_file.open(m_path, std::ios::out | std::ios::trunc);

    if (m_file.is_open()) {
        // Write header
        writeLine("# NNUE Training Runner - Structured Log");
        writeLine("# Format: [TIMESTAMP] [LEVEL] [PHASE:GEN] message");
        writeLine("# Structured events use KEY=VALUE pairs after the tag");
        writeLine("# Levels: INFO, WARN, ERROR, EVENT, METRIC, CONFIG, CMD");
        writeLine("# Generated: " + timestamp());
        writeLine("#");
        writeLine("# ── HOW TO READ THIS LOG ──");
        writeLine("# Lines starting with [EVENT] mark pipeline milestones (start/stop/phase transitions)");
        writeLine("# Lines starting with [METRIC] contain numeric training data points");
        writeLine("# Lines starting with [ERROR] or [WARN] flag issues to investigate");
        writeLine("# Lines starting with [CONFIG] dump the full configuration used");
        writeLine("# Lines starting with [CMD] show exact command lines executed");
        writeLine("# Lines starting with [INFO] are general informational messages");
        writeLine("#");
    }
    return m_path;
}

void FileLogger::close() {
    std::lock_guard<std::mutex> lk(m_mtx);
    if (m_file.is_open()) {
        writeLine("# ── END OF LOG ── " + timestamp());
        m_file.close();
    }
    m_path.clear();
}

bool FileLogger::isOpen() const {
    return m_file.is_open();
}

std::string FileLogger::getPath() const {
    return m_path;
}

// ── Core log method ───────────────────────────────────────────────

void FileLogger::log(LogLevel level, const std::string& phase, int gen,
                     const std::string& msg)
{
    if (msg.empty()) return;

    // Skip \r-prefixed overwrite lines (progress updates) — they create noise.
    // Only log the final version of progress lines.
    if (!msg.empty() && msg[0] == '\r') return;

    std::ostringstream line;
    line << '[' << timestamp() << "] "
         << '[' << levelTag(level) << "] "
         << '[' << (phase.empty() ? "general" : phase)
         << ':' << gen << "] "
         << msg;

    std::lock_guard<std::mutex> lk(m_mtx);
    writeLine(line.str());
}

// ── Convenience wrappers ──────────────────────────────────────────

void FileLogger::info (const std::string& phase, int gen, const std::string& msg) { log(LogLevel::INFO,  phase, gen, msg); }
void FileLogger::warn (const std::string& phase, int gen, const std::string& msg) { log(LogLevel::WARN,  phase, gen, msg); }
void FileLogger::error(const std::string& phase, int gen, const std::string& msg) { log(LogLevel::ERR,   phase, gen, msg); }
void FileLogger::cmd  (const std::string& phase, int gen, const std::string& msg) { log(LogLevel::CMD,   phase, gen, msg); }

// ── Structured event writers ──────────────────────────────────────

void FileLogger::logConfig(const std::string& configDump) {
    std::lock_guard<std::mutex> lk(m_mtx);
    writeLine("[" + timestamp() + "] [CONFIG] [pipeline:0] " + configDump);
}

void FileLogger::logMetric(int gen, int epoch, double trainLoss, double valLoss,
                           double lr, double acc,
                           double openingLoss, double middlegameLoss, double endgameLoss,
                           bool hasVal, bool hasAcc, bool hasPhase)
{
    std::ostringstream line;
    line << '[' << timestamp() << "] [METRIC] [training:" << gen << "] "
         << "EPOCH"
         << " epoch=" << epoch
         << " train_loss=" << std::fixed << std::setprecision(8) << trainLoss;
    if (hasVal)
        line << " val_loss=" << std::fixed << std::setprecision(8) << valLoss;
    line << " lr=" << std::fixed << std::setprecision(8) << lr;
    if (hasAcc)
        line << " acc=" << std::fixed << std::setprecision(4) << acc;
    if (hasPhase)
        line << " opening_loss=" << std::fixed << std::setprecision(8) << openingLoss
             << " middlegame_loss=" << middlegameLoss
             << " endgame_loss=" << endgameLoss;

    std::lock_guard<std::mutex> lk(m_mtx);
    writeLine(line.str());
}

void FileLogger::logElo(int gen, int elo, int wins, int draws, int losses,
                        const std::string& matchType)
{
    std::ostringstream line;
    line << '[' << timestamp() << "] [METRIC] [" << matchType << ':' << gen << "] "
         << "ELO_RESULT"
         << " elo=" << (elo >= 0 ? "+" : "") << elo
         << " wins=" << wins
         << " draws=" << draws
         << " losses=" << losses;

    std::lock_guard<std::mutex> lk(m_mtx);
    writeLine(line.str());
}

void FileLogger::logPhaseStart(const std::string& phase, int gen, const std::string& detail) {
    std::ostringstream line;
    line << '[' << timestamp() << "] [EVENT] [" << phase << ':' << gen << "] "
         << "PHASE_START";
    if (!detail.empty()) line << ' ' << detail;

    std::lock_guard<std::mutex> lk(m_mtx);
    writeLine(line.str());
}

void FileLogger::logPhaseEnd(const std::string& phase, int gen, bool success, double elapsedSec) {
    std::ostringstream line;
    line << '[' << timestamp() << "] [EVENT] [" << phase << ':' << gen << "] "
         << "PHASE_END"
         << " success=" << (success ? "true" : "false")
         << " elapsed_sec=" << std::fixed << std::setprecision(1) << elapsedSec;

    std::lock_guard<std::mutex> lk(m_mtx);
    writeLine(line.str());
}

void FileLogger::logPipelineStart(int firstGen, int lastGen, int totalGens) {
    std::ostringstream line;
    line << '[' << timestamp() << "] [EVENT] [pipeline:0] "
         << "PIPELINE_START"
         << " first_gen=" << firstGen
         << " last_gen=" << lastGen
         << " total_gens=" << totalGens;

    std::lock_guard<std::mutex> lk(m_mtx);
    writeLine(line.str());
}

void FileLogger::logPipelineEnd(bool stopped, double totalElapsedSec, int completedGens) {
    std::ostringstream line;
    line << '[' << timestamp() << "] [EVENT] [pipeline:0] "
         << "PIPELINE_END"
         << " stopped=" << (stopped ? "true" : "false")
         << " total_elapsed_sec=" << std::fixed << std::setprecision(1) << totalElapsedSec
         << " completed_gens=" << completedGens;

    std::lock_guard<std::mutex> lk(m_mtx);
    writeLine(line.str());
}

void FileLogger::logGenSummary(int gen, double trainLoss, double valLoss, int elo, double genElapsedSec) {
    std::ostringstream line;
    line << '[' << timestamp() << "] [EVENT] [pipeline:" << gen << "] "
         << "GEN_SUMMARY"
         << " gen=" << gen
         << " train_loss=" << std::fixed << std::setprecision(8) << trainLoss
         << " val_loss=" << std::fixed << std::setprecision(8) << valLoss
         << " elo=" << (elo >= 0 ? "+" : "") << elo
         << " elapsed_sec=" << std::fixed << std::setprecision(1) << genElapsedSec;

    std::lock_guard<std::mutex> lk(m_mtx);
    writeLine(line.str());
}
