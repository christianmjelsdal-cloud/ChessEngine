#pragma once
// Resolve asset paths relative to the executable's directory,
// so it works regardless of the working directory (e.g. running from VS).

#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>   // _NSGetExecutablePath
#include <climits>
#include <cstdlib>         // realpath
#else
#include <unistd.h>
#include <climits>
#endif

inline std::string getExeDir() {
    static std::string dir = []() -> std::string {
        std::string result;
#ifdef _WIN32
        char buf[MAX_PATH];
        DWORD len = GetModuleFileNameA(nullptr, buf, MAX_PATH);
        if (len > 0 && len < MAX_PATH) {
            std::string p(buf, len);
            auto pos = p.find_last_of("\\/");
            result = (pos != std::string::npos) ? p.substr(0, pos + 1) : "";
        }
#elif defined(__APPLE__)
        // AUDIT FIX BUG-5: macOS doesn't have /proc/self/exe.
        // Use _NSGetExecutablePath + realpath to resolve symlinks.
        char rawBuf[PATH_MAX];
        uint32_t size = sizeof(rawBuf);
        if (_NSGetExecutablePath(rawBuf, &size) == 0) {
            char realBuf[PATH_MAX];
            if (realpath(rawBuf, realBuf)) {
                std::string p(realBuf);
                auto pos = p.find_last_of('/');
                result = (pos != std::string::npos) ? p.substr(0, pos + 1) : "";
            }
        }
#else
        // Linux: /proc/self/exe
        char buf[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (len > 0) {
            buf[len] = '\0';
            std::string p(buf);
            auto pos = p.find_last_of('/');
            result = (pos != std::string::npos) ? p.substr(0, pos + 1) : "";
        }
#endif
        return result;
    }();
    return dir;
}

// Returns e.g. "C:\Users\chris\...\Release\assets\nnue_weights"
inline std::string assetPath(const std::string& relative) {
    return getExeDir() + relative;
}
