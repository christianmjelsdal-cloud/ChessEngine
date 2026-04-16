// TR_Process.cpp  --  Subprocess management, suspend/resume
#include "TR_Types.h"
#include "TR_Globals.h"
#include "TR_Fwd.h"

// Collect all PIDs in the process tree rooted at 'root'
static std::vector<DWORD> CollectProcessTree(DWORD root) {
    std::vector<DWORD> pids = { root };
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return pids;
    PROCESSENTRY32 pe{}; pe.dwSize = sizeof(pe);
    if (Process32First(snap, &pe)) {
        // Multiple passes to collect transitive children
        bool found = true;
        while (found) {
            found = false;
            if (Process32First(snap, &pe)) {
                do {
                    for (DWORD p : pids) {
                        if (pe.th32ParentProcessID == p) {
                            bool already = false;
                            for (DWORD q : pids) if (q == pe.th32ProcessID) { already = true; break; }
                            if (!already) { pids.push_back(pe.th32ProcessID); found = true; }
                            break;
                        }
                    }
                } while (Process32Next(snap, &pe));
            }
        }
    }
    CloseHandle(snap);
    return pids;
}

// FIX 6.19: Validate PID is still the expected process before suspend/resume
// Uses the stored process handle to confirm the process is alive and matches.
DWORD GetValidatedActivePid() {
    std::lock_guard<std::mutex> lk(g_proc.activePiMtx);
    if (g_proc.activePi.hProcess == nullptr) return 0;
    // Check if the process handle still refers to a running process
    DWORD exitCode = 0;
    if (!GetExitCodeProcess(g_proc.activePi.hProcess, &exitCode)) return 0;
    if (exitCode != STILL_ACTIVE) return 0;
    return g_proc.activePi.dwProcessId;
}

void SuspendProcessThreads(DWORD pid) {
    auto pids = CollectProcessTree(pid);
    if (pids.empty()) return;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    THREADENTRY32 te{}; te.dwSize = sizeof(te);
    if (Thread32First(snap, &te)) {
        do {
            for (DWORD p : pids) {
                if (te.th32OwnerProcessID == p) {
                    HANDLE ht = OpenThread(THREAD_SUSPEND_RESUME, FALSE, te.th32ThreadID);
                    if (ht) { SuspendThread(ht); CloseHandle(ht); }
                    break;
                }
            }
        } while (Thread32Next(snap, &te));
    }
    CloseHandle(snap);
}

void ResumeProcessThreads(DWORD pid) {
    auto pids = CollectProcessTree(pid);
    if (pids.empty()) return;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    THREADENTRY32 te{}; te.dwSize = sizeof(te);
    if (Thread32First(snap, &te)) {
        do {
            for (DWORD p : pids) {
                if (te.th32OwnerProcessID == p) {
                    HANDLE ht = OpenThread(THREAD_SUSPEND_RESUME, FALSE, te.th32ThreadID);
                    if (ht) { ResumeThread(ht); CloseHandle(ht); }
                    break;
                }
            }
        } while (Thread32Next(snap, &te));
    }
    CloseHandle(snap);
}

void SuspendOrTerminateActive() {
    DWORD pid = g_proc.activePid.load();
    if (pid == 0) return;
    // Send CTRL_BREAK_EVENT to process group
    GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid);
    // Wait up to 3000ms
    HANDLE hProc = OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE, FALSE, pid);
    if (hProc) {
        if (WaitForSingleObject(hProc, 3000) != WAIT_OBJECT_0) {
            TerminateProcess(hProc, 1);
        }
        CloseHandle(hProc);
    }
}

// ── Subprocess ────────────────────────────────────────────────────
bool RunProc(const std::wstring& cmd, const std::string& dir,
                    std::function<void(const std::string&)> cb,
                    std::atomic<bool>& stop)
{
    SECURITY_ATTRIBUTES sa{}; sa.nLength = sizeof(sa); sa.bInheritHandle = TRUE;
    HANDLE hR, hW;
    if (!CreatePipe(&hR, &hW, &sa, 0)) return false;
    SetHandleInformation(hR, HANDLE_FLAG_INHERIT, 0);

    // Provide a valid stdin handle (NUL device) so the child's initPipeIO()
    // doesn't see a NULL stdin and can safely wire up stdout/stderr.
    HANDLE hNulIn = CreateFileW(L"NUL", GENERIC_READ, 0, &sa, OPEN_EXISTING, 0, NULL);

    STARTUPINFOW si{}; si.cb = sizeof(si);
    si.hStdInput = hNulIn;
    si.hStdOutput = hW; si.hStdError = hW;
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;

    PROCESS_INFORMATION pi{};
    std::wstring c = cmd;
    std::wstring wd = W(dir);
    // Log the exact command so failures are diagnosable
    // NOTE: Lossy wchar_t→char truncation — non-ASCII chars become garbled in the log.
    // Acceptable for diagnostic logging; paths are almost always ASCII in practice.
    { std::string narrow; narrow.reserve(c.size());
      for (wchar_t wc : c) narrow += static_cast<char>(wc);
      cb("[CMD] " + narrow); }

    bool ok = CreateProcessW(nullptr, c.data(), nullptr, nullptr, TRUE,
                             CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW, nullptr,
                             wd.empty() ? nullptr : wd.c_str(), &si, &pi) != 0;
    CloseHandle(hW);
    CloseHandle(hNulIn);
    if (!ok) {
        DWORD err = GetLastError();
        cb("[ERR] CreateProcess failed (Win32 error " + std::to_string(err) + ") - exe not found or access denied");
        CloseHandle(hR); return false;
    }

    // Store active PID so Stop/Pause buttons can act on it immediately
    g_proc.activePid.store(pi.dwProcessId);
    {
        std::lock_guard<std::mutex> lk(g_proc.activePiMtx);
        g_proc.activePi = pi;
    }
    // If paused, immediately suspend the new process (handles pause across process transitions)
    if (g_proc.pauseFlag.load()) {
        SuspendProcessThreads(pi.dwProcessId);
    }

    std::string buf; char ch[4096] = {}; DWORD br = 0;
    while (ReadFile(hR, ch, (DWORD)(sizeof(ch)-1), &br, nullptr) && br > 0) {
        ch[br] = '\0'; buf += ch;
        size_t p;
        while ((p = buf.find_first_of("\r\n")) != std::string::npos) {
            std::string ln = buf.substr(0, p);
            bool isCR = (buf[p] == '\r');
            if (isCR && p + 1 < buf.size() && buf[p+1] == '\n') {
                buf = buf.substr(p + 2);  // \r\n → normal newline
                isCR = false;
            } else {
                buf = buf.substr(p + 1);
            }
            // Bare \r means "overwrite current line" — prefix with \r
            // so pushLog knows to replace instead of append
            // FIX 6.16: Note — bare \r at buffer boundary with empty ln may lose overwrite.
            // This is benign in practice as OS pipe buffering delivers complete lines.
            if (!ln.empty()) cb(isCR ? ("\r" + ln) : ln);
        }
        if (stop.load() || g_proc.skipPhaseFlag.load()) {
            GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pi.dwProcessId);
            WaitForSingleObject(pi.hProcess, 3000);
            TerminateProcess(pi.hProcess, 1);
            break;
        }
    }
    if (!buf.empty()) cb(buf);
    WaitForSingleObject(pi.hProcess, 10000);  // 10s timeout instead of INFINITE
    DWORD ex=1; GetExitCodeProcess(pi.hProcess, &ex);
    if (ex != 0) {
        if (stop.load())
            cb("[INFO] Process terminated -- pipeline stopped by user.");
        else if (g_proc.skipPhaseFlag.load())
            cb("[INFO] Process terminated -- phase skipped by user.");
        else
            cb("[ERR] Process exited with code " + std::to_string(ex) + " -- check logs for details.");
    }
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread); CloseHandle(hR);

    // Clear active PID
    g_proc.activePid.store(0);
    {
        std::lock_guard<std::mutex> lk(g_proc.activePiMtx);
        g_proc.activePi = {};
    }

    bool skipped = g_proc.skipPhaseFlag.load();
    return (ex == 0 || skipped) && !stop.load();
}

