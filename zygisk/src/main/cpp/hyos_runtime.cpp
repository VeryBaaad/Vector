#include <dirent.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "common/logging.h"
#include "core/native_api.h"

#include "zygisk_next_api.h"

/**
 * @file hyos_runtime.cpp
 * @brief Native hooking support for applications running on the HyperOS Rust Runtime.
 *
 * A process spawned by /system_ext/bin/hyos_spawner is not an ART process. It has no JVM, no
 * JNIEnv, and no binder of its own, so none of the machinery the rest of Vector is built on is
 * available inside it: the framework DEX cannot be loaded, and the daemon cannot be asked what is
 * in scope. What such a process does have is the dynamic loader, and therefore native libraries.
 *
 * This translation unit is the whole of Vector's presence there. It is compiled into the same
 * libzygisk.so the Zygisk loader injects everywhere else, and it is reached through Zygisk Next's
 * Runtime API, which the module declares itself for in zn_modules.txt:
 *
 *   path=/system_ext/bin/hyos_spawner companion zygisk/arm64-v8a.so
 *
 * The flow, in order:
 *
 *   1. Zygisk Next injects this library into hyos_spawner and calls `zn_module`'s onModuleLoaded.
 *      getRuntime() reports ZN_RUNTIME_HYOS, and we register `zn_companion_module`'s callbacks.
 *   2. The spawner forks an application process, applies its uid, gid, groups and SELinux context,
 *      and calls onAppSpecialized in the child.
 *   3. The child asks the companion -- a root process Zygisk Next forked for us, reached over a
 *      socket the spawner connected and every child inherited -- for the path the Vector daemon
 *      publishes its per-package state under.
 *   4. The child reads the index for its own package, which names the native libraries of every
 *      Xposed module in scope for it, and loads each one. Each library's `native_init` is then
 *      called with the same NativeAPIEntries an ART process gets, so a module's native part needs
 *      no change to work here: it hooks the HyperOS process with the primitives it was already
 *      written against.
 *
 * The hook primitives come from Zygisk Next rather than from Vector's own Dobby, for the reason
 * SetHookBackend documents: two engines patching the same address destroy each other's trampolines,
 * and the engine that already owns this process has the better claim.
 */

namespace vector::native::hyos {

namespace {

// --- The published state, and where to find it ------------------------------------------------

/**
 * A fixed path the daemon writes the real location into, because there is nothing to list.
 *
 * /data/misc belongs to the system uid with mode 0771, so only root and the daemon can create
 * entries in it, and this file is mode 0600. That combination is deliberate: a process running as
 * an application cannot read it, so an application cannot learn the directory the index lives in
 * and therefore cannot ask whether it is being hooked. The companion is the only reader, and it
 * runs as root.
 */
constexpr auto kPointerPath = "/data/misc/vector.hyos";

/// The prefix of the single line the pointer file carries.
constexpr std::string_view kPointerKey = "misc=";

/// Where the random Vector directory lives, for the fallback that has to go looking.
constexpr auto kMiscRootPath = "/data/misc";

/// The index directory the daemon creates inside its random directory, and its marker file.
constexpr std::string_view kIndexDirName = "hyos";
constexpr std::string_view kIndexMarkerName = ".version";
constexpr std::string_view kIndexMagic = "vector-hyos 1";

/// The socket message that asks the companion for the index location.
constexpr char kRequestMiscRoot = 'M';

/// Longest reply accepted from the companion, so a confused peer cannot exhaust memory.
constexpr size_t kMaxReplyLength = 4096;

/**
 * Where the companion reports whether this process's hyperos support is actually working.
 *
 * A client connects and reads one byte: `1` means the Zygisk Next Runtime API was registered and
 * applications will be specialized, anything else means the runtime did not want us. It is the
 * whole interface, and it exists because the daemon has no other way to ask: the daemon is a Java
 * process, and the connection the spawner holds is not one it can reach.
 *
 * The path is the same one LSPosed 2.2.0 uses, recovered from its released daemon -- its
 * `ILSPManagerService` transaction 67 connects here and reads exactly that byte, and its manager
 * turns the answer into the "HyperOS Runtime injection failed" notice the user sees. Being under
 * /data/adb/lspd, which is mode 0700 owned by root, is deliberate: the companion and the daemon are
 * both root, and nothing running as an application may ask.
 */
constexpr auto kMonitorPath = "/data/adb/lspd/hyos_monitor";

/// How long the companion waits for the spawner's status byte before assuming the worst.
constexpr time_t kStatusByteTimeoutSeconds = 2;

// --- The Zygisk Next view of this process -----------------------------------------------------

const ZygiskNextAPI *g_api = nullptr;

// How long a child waits for the companion before giving up on it.
//
// This runs inside application startup, ahead of everything the application does, so a companion
// that has wedged must cost a bounded delay rather than the launch itself. The reply is a path
// string written by a process that is already running, so anything above milliseconds means the
// companion is not coming.
constexpr time_t kCompanionReplyTimeoutSeconds = 1;

// The companion connection, established by the spawner before any fork and inherited by every
// child. -1 when the companion could not be started, which costs the callback delivery.
int g_companion_fd = -1;

// Whether the HyperOS Runtime API was actually registered here. Zygisk Next's Runtime API is an
// optional feature of that loader: an older build does not offer it, and one that does can still
// refuse the registration. None of that is this process's problem, so every such path turns the
// feature off rather than failing -- an ordinary Zygisk process and a hyos_spawner running without
// us are both perfectly good outcomes.
bool g_registered = false;

// What the monitor hands out: 1 once the Runtime API is registered, 0 until then and forever if it
// never is. Written by the companion when the spawner reports, read by every monitor client -- two
// threads of one process, which is why it is atomic rather than plain.
std::atomic<char> g_injection_status{0};

// Whether this process has already done its work. A child is forked once and specializes once, but
// a runtime is free to call the callback again, and loading a module's libraries twice would run
// its native_init twice.
bool g_specialized = false;

// --- Reading files without assuming anything --------------------------------------------------

/// Reads a whole file, up to a sane bound. False when it cannot be read at all.
bool ReadFile(const std::string &path, std::string &out) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;

    out.clear();
    char buf[512];
    for (;;) {
        const ssize_t n = read(fd, buf, sizeof(buf));
        if (n < 0) {
            close(fd);
            out.clear();
            return false;
        }
        if (n == 0) break;
        out.append(buf, static_cast<size_t>(n));
        // Every file read here is a few hundred bytes of text; a larger one is not one of ours.
        if (out.size() > (1u << 20)) {
            close(fd);
            out.clear();
            return false;
        }
    }
    close(fd);
    return true;
}

/// Writes every byte, or reports that it could not.
bool WriteAll(int fd, const std::string &data) {
    size_t written = 0;
    while (written < data.size()) {
        const ssize_t n = write(fd, data.data() + written, data.size() - written);
        if (n <= 0) return false;
        written += static_cast<size_t>(n);
    }
    return true;
}

/// The first line of `text`, without its terminator.
std::string FirstLine(const std::string &text) {
    const auto end = text.find('\n');
    std::string line = text.substr(0, end == std::string::npos ? text.size() : end);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return line;
}

/**
 * @brief Whether a directory really is the one the Vector daemon publishes into.
 *
 * The random name is the whole point of the directory, so finding one by looking means finding
 * somebody else's directory just as easily. The marker is what tells them apart.
 */
bool IsVectorMiscRoot(const std::string &root) {
    std::string content;
    const std::string marker =
        root + "/" + std::string(kIndexDirName) + "/" + std::string(kIndexMarkerName);
    if (!ReadFile(marker, content)) return false;
    return FirstLine(content) == kIndexMagic;
}

/**
 * @brief Finds the index location.
 *
 * The pointer file first, because it needs no directory listing at all. Failing that, look inside
 * /data/misc, where the daemon's directory is the only one carrying the marker. Both run in the
 * companion, which is root; an application process could do neither.
 */
std::string ResolveMiscRoot() {
    std::string pointer;
    if (ReadFile(kPointerPath, pointer)) {
        const std::string line = FirstLine(pointer);
        const std::string_view view{line};
        if (view.rfind(kPointerKey, 0) == 0) {
            std::string root{view.substr(kPointerKey.size())};
            if (!root.empty() && root.back() == '/') root.pop_back();
            if (IsVectorMiscRoot(root)) return root;
            LOGW("The published Vector directory '{}' carries no index marker.", root.c_str());
        } else {
            LOGW("{} does not name a Vector directory.", kPointerPath);
        }
    } else {
        LOGD("No published Vector directory at {}.", kPointerPath);
    }

    // Fallback: it is a random name under /data/misc, so the marker is the only way to recognise
    // it. Reached when the pointer file is missing, or names something that is not there.
    DIR *dir = opendir(kMiscRootPath);
    if (dir == nullptr) {
        LOGW("Cannot look for the Vector directory in {}: {}.", kMiscRootPath, strerror(errno));
        return {};
    }
    std::string found;
    while (struct dirent *entry = readdir(dir)) {
        if (entry->d_name[0] == '.') continue;
        std::string candidate = std::string(kMiscRootPath) + "/" + entry->d_name;
        if (IsVectorMiscRoot(candidate)) {
            found = std::move(candidate);
            break;
        }
    }
    closedir(dir);
    if (!found.empty()) LOGI("Found the Vector directory by marker: {}", found.c_str());
    return found;
}

// --- The application process ------------------------------------------------------------------

/**
 * @brief Asks the companion where the daemon published its state.
 *
 * Read one byte at a time up to the first newline rather than a whole buffer, because the socket
 * is shared with every sibling process and a longer read could swallow a reply that was never
 * meant for this one. Every reply is the same string, so the first line is always the answer.
 */
std::string AskCompanionForMiscRoot() {
    if (g_companion_fd < 0) {
        LOGW("VectorHyperRuntime: no companion connection; cannot locate the Vector directory.");
        return {};
    }
    const char request = kRequestMiscRoot;
    if (write(g_companion_fd, &request, 1) != 1) {
        LOGW("VectorHyperRuntime: cannot ask the companion: {}.", strerror(errno));
        return {};
    }

    std::string line;
    bool complete = false;
    while (line.size() < kMaxReplyLength) {
        char c = 0;
        const ssize_t n = read(g_companion_fd, &c, 1);
        if (n <= 0) break;
        if (c == '\n') {
            complete = true;
            break;
        }
        line.push_back(c);
    }
    if (!complete) {
        // A half-read path would be worse than none: it would name a directory that does not exist
        // and be reported as a scope miss. The read is also where the timeout lands, so this is the
        // branch a wedged companion produces.
        LOGW("VectorHyperRuntime: the companion did not answer in time; skipping injection.");
        return {};
    }
    if (line.empty()) LOGW("VectorHyperRuntime: the companion knows no Vector directory yet.");
    return line;
}

/// One module native library to load into this process.
struct LibraryEntry {
    std::string module_package;
    std::string library_name;
    std::string library_path;
};

/// Splits one index line into its three tab-separated fields.
bool ParseIndexLine(const std::string &line, LibraryEntry &out) {
    const auto first = line.find('\t');
    if (first == std::string::npos) return false;
    const auto second = line.find('\t', first + 1);
    if (second == std::string::npos) return false;

    out.module_package = line.substr(0, first);
    out.library_name = line.substr(first + 1, second - first - 1);
    out.library_path = line.substr(second + 1);
    return !out.library_name.empty() && !out.library_path.empty();
}

/**
 * @brief Reads the libraries the daemon listed for `key`.
 *
 * A missing file is the ordinary answer for a process that is in nobody's scope, so it is not
 * reported as a failure.
 */
std::vector<LibraryEntry> ReadIndex(const std::string &misc_root, const std::string &key) {
    std::vector<LibraryEntry> entries;
    if (misc_root.empty() || key.empty()) return entries;

    const std::string path =
        misc_root + "/" + std::string(kIndexDirName) + "/" + key;
    std::string content;
    if (!ReadFile(path, content)) {
        LOGD("VectorHyperRuntime: no index for '{}'.", key.c_str());
        return entries;
    }

    bool first = true;
    for (size_t start = 0; start < content.size();) {
        const auto end = content.find('\n', start);
        const std::string line =
            content.substr(start, end == std::string::npos ? std::string::npos : end - start);
        start = end == std::string::npos ? content.size() : end + 1;

        if (first) {
            first = false;
            if (line == kIndexMagic) continue;
            LOGE("VectorHyperRuntime: {} is not a Vector index; ignoring it.", path.c_str());
            return {};
        }
        if (line.empty() || line[0] == '#') continue;

        LibraryEntry entry;
        if (!ParseIndexLine(line, entry)) {
            LOGW("VectorHyperRuntime: skipping a malformed line in {}.", path.c_str());
            continue;
        }
        entries.push_back(std::move(entry));
    }
    return entries;
}

/**
 * @brief Brings the native API, and every module library the daemon listed, into this process.
 */
void LoadModuleLibraries(const std::vector<LibraryEntry> &entries) {
    // The interception of the loader was installed by the spawner before it forked, so it is
    // already in place here and inherited. InstallNativeAPI is idempotent and reports that state.
    const bool intercepts_dlopen = InstallNativeAPI();
    if (!intercepts_dlopen) {
        LOGW("VectorHyperRuntime: the loader is not intercepted. Module libraries will still be "
             "loaded, but nothing will observe the loads that follow.");
    }

    const NativeAPIEntries *api_entries = GetNativeAPIEntries();
    if (api_entries == nullptr) {
        LOGE("VectorHyperRuntime: the native API entries could not be built; not loading modules.");
        return;
    }

    for (const auto &entry : entries) {
        // Registering first is what lets the loader hook recognize this very dlopen and call
        // native_init itself. When the hook is not there it is called by hand below, so a module is
        // initialized either way.
        if (intercepts_dlopen) RegisterNativeLib(entry.library_name);

        void *handle = dlopen(entry.library_path.c_str(), RTLD_NOW);
        if (handle == nullptr) {
            LOGE("VectorHyperRuntime: cannot load {} for '{}': {}.", entry.library_path.c_str(),
                 entry.module_package.c_str(), dlerror());
            continue;
        }
        LOGI("VectorHyperRuntime: loaded {} for '{}'.", entry.library_path.c_str(),
             entry.module_package.c_str());

        if (intercepts_dlopen) continue;
        void *init_sym = dlsym(handle, "native_init");
        if (init_sym == nullptr) {
            LOGW("VectorHyperRuntime: {} does not export native_init.", entry.library_path.c_str());
            continue;
        }
        reinterpret_cast<NativeInit>(init_sym)(api_entries);
    }
}

}  // namespace

/**
 * @brief Called once per specialized application process, after its uid, gid, groups and SELinux
 *        context have been applied.
 *
 * This runs in a process forked from a possibly multithreaded parent, so the work is kept to
 * reading two small files and loading libraries: no threads are started, and no lock the parent
 * could have caught at fork time is taken beyond the ones the loader itself takes.
 */
void OnAppSpecialized(const ZnHyosAppSpecializeArgs *args) {
    if (!g_registered) return;
    if (g_specialized) return;
    g_specialized = true;
    if (args == nullptr || args->package_name == nullptr) {
        LOGE("VectorHyperRuntime: specialization arrived without a package name.");
        return;
    }

    LOGI("VectorHyperRuntime: specializing process '{}' (package '{}').", args->process_name,
         args->package_name);

    const std::string misc_root = AskCompanionForMiscRoot();
    if (misc_root.empty()) return;

    // The daemon publishes per process name, because that is what a scope is keyed by, and the
    // process name is what this process actually is. It is the first choice for that reason, not
    // the second: a package with more than one process has a different scope in each, and looking
    // the package up first would hand the main process's modules to the secondary one.
    //
    // The package name is the fallback, for a runtime that gave us a process name we cannot use --
    // an inherited one, or one truncated to the kernel's fifteen characters. It always resolves to
    // the package's main process, so a secondary process that falls back here is served the main
    // process's list rather than nothing; erring towards loading is the lesser mistake, because a
    // module that meant to hook the main process and hooks a secondary one of the same app has at
    // least been given what it asked for, while a module that is silently absent looks broken.
    std::vector<LibraryEntry> entries;
    if (args->process_name != nullptr) entries = ReadIndex(misc_root, args->process_name);
    if (entries.empty()) entries = ReadIndex(misc_root, args->package_name);
    if (entries.empty()) {
        LOGD("VectorHyperRuntime: '{}' is in no module's scope.", args->package_name);
        return;
    }

    LOGI("VectorHyperRuntime: '{}' has {} module librar{} to load.", args->package_name,
         entries.size(), entries.size() == 1 ? "y" : "ies");
    LoadModuleLibraries(entries);
}

/**
 * @brief Serves the index location to children over the inherited socket.
 *
 * Every reply is the same string, which is what makes one shared connection safe: children are
 * separate processes holding the same socket, so two of them asking at once can each read the
 * other's reply -- and it is still the right answer. The loop ends when the peer does.
 */
void *ServeCompanion(void *arg) {
    const int fd = static_cast<int>(reinterpret_cast<intptr_t>(arg));
    for (;;) {
        char request = 0;
        if (read(fd, &request, 1) != 1) break;
        if (request != kRequestMiscRoot) continue;

        std::string reply = ResolveMiscRoot();
        reply.push_back('\n');
        if (!WriteAll(fd, reply)) break;
    }
    close(fd);
    return nullptr;
}

/**
 * @brief Answers the daemon's question about this runtime.
 *
 * Started with the companion rather than with the first connection, because "started and nothing
 * registered yet" has to be answerable: it is what a runtime the loader refused looks like, and the
 * difference between that and no companion at all is the whole reason the daemon asks.
 */
void *ServeMonitor(void *) {
    // A socket left behind by a companion that was killed would make every later bind fail, and the
    // daemon would keep reading the old inode instead of ours.
    unlink(kMonitorPath);

    const int listener = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (listener < 0) {
        LOGE("VectorHyperRuntime: cannot create {}: {}.", kMonitorPath, strerror(errno));
        return nullptr;
    }

    struct sockaddr_un address {};
    address.sun_family = AF_UNIX;
    strlcpy(address.sun_path, kMonitorPath, sizeof(address.sun_path));
    if (bind(listener, reinterpret_cast<struct sockaddr *>(&address), sizeof(address)) != 0 ||
        listen(listener, 8) != 0) {
        LOGE("VectorHyperRuntime: cannot listen on {}: {}.", kMonitorPath, strerror(errno));
        close(listener);
        return nullptr;
    }
    // The daemon runs as root and the directory is root-only, so the mode is not what admits it;
    // it is set anyway so that a stale socket is never what refuses a legitimate reader.
    if (chmod(kMonitorPath, 0666) != 0) {
        LOGW("VectorHyperRuntime: cannot relax {}: {}.", kMonitorPath, strerror(errno));
    }
    LOGI("VectorHyperRuntime: reporting injection status on {}.", kMonitorPath);

    for (;;) {
        const int client = accept(listener, nullptr, nullptr);
        if (client < 0) {
            if (errno == EINTR) continue;
            break;
        }
        const char status = g_injection_status.load();
        if (write(client, &status, 1) != 1) {
            LOGD("VectorHyperRuntime: a monitor client went away before its answer.");
        }
        close(client);
    }
    close(listener);
    return nullptr;
}

void OnCompanionLoaded() {
    LOGI("VectorHyperRuntime: companion loaded in pid {}.", static_cast<int>(getpid()));
    pthread_t thread;
    if (pthread_create(&thread, nullptr, ServeMonitor, nullptr) != 0) {
        LOGE("VectorHyperRuntime: cannot serve {} on a thread; the daemon will see no runtime.",
             kMonitorPath);
        return;
    }
    pthread_detach(thread);
}

void OnModuleConnected(int fd) {
    LOGI("VectorHyperRuntime: companion connected on fd {}.", fd);

    // The spawner's first write is its status byte, and it is written before any fork, so it is
    // already on its way when this runs. Bounded anyway: a spawner that died between connecting and
    // writing must not leave the companion -- and the loader's own loop that called us -- stuck.
    struct timeval timeout {};
    timeout.tv_sec = kStatusByteTimeoutSeconds;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    char status = 0;
    if (read(fd, &status, 1) == 1 && status == 1) {
        g_injection_status.store(1);
        LOGI("VectorHyperRuntime: the runtime API is registered; the daemon will be told so.");
    } else {
        g_injection_status.store(0);
        LOGW("VectorHyperRuntime: no registration status arrived; the daemon will be told that "
             "injection is not working.");
    }

    pthread_t thread;
    auto *argument = reinterpret_cast<void *>(static_cast<intptr_t>(fd));
    if (pthread_create(&thread, nullptr, ServeCompanion, argument) == 0) {
        pthread_detach(thread);
        return;
    }
    LOGE("VectorHyperRuntime: cannot serve the companion on a thread; serving inline.");
    ServeCompanion(argument);
}

void OnModuleLoaded(void *self_handle, const ZygiskNextAPI *api) {
    if (api == nullptr) {
        LOGW("VectorHyperRuntime: the loader handed us no API; disabled in this process.");
        return;
    }
    g_api = api;

    // Every refusal below is a warning, not an error. The Runtime API is an optional part of the
    // loader: an older build does not have it, one that does can still say no, and in both cases
    // the right outcome is the same -- nothing of ours runs here, and the process carries on
    // exactly as it would without this feature. Saying so at warning level is the whole report.
    const ZygiskNextRuntime *runtime = api->getRuntime ? api->getRuntime() : nullptr;
    if (runtime == nullptr) {
        LOGW("VectorHyperRuntime: this loader offers no Runtime API; disabled in this process.");
        return;
    }
    if (runtime->type != ZN_RUNTIME_HYOS) {
        LOGW("VectorHyperRuntime: runtime type {} is not the HyperOS one; disabled in this process.",
             static_cast<int>(runtime->type));
        return;
    }
    if (runtime->api_version < ZYGISK_NEXT_HYOS_API_VERSION) {
        LOGW("VectorHyperRuntime: the runtime speaks API {}, we need {}; disabled in this process.",
             runtime->api_version, ZYGISK_NEXT_HYOS_API_VERSION);
        return;
    }
    if (runtime->registerModule == nullptr) {
        LOGW("VectorHyperRuntime: the runtime cannot register modules; disabled in this process.");
        return;
    }

    // Take the runtime's own hook engine before the native API is handed to any module: this is the
    // process the hooks will be installed in, and the runtime is the one that knows what it has
    // already patched. Failing here is not fatal -- the API keeps Dobby -- but it is worth saying,
    // because two engines on one address is exactly what the handover avoids.
    if (api->inlineHook == nullptr || api->inlineUnhook == nullptr) {
        LOGW("VectorHyperRuntime: the runtime offers no inline hooks; falling back to Dobby.");
    } else {
        SetHookBackend(api->inlineHook, api->inlineUnhook);
    }

    static const ZygiskNextHyosModule hyos_module = {
        .target_api_version = ZYGISK_NEXT_HYOS_API_VERSION,
        .onAppSpecialized = OnAppSpecialized,
    };
    if (runtime->registerModule(&hyos_module) != ZN_SUCCESS) {
        LOGW("VectorHyperRuntime: the runtime refused our specialization callback; disabled in this "
             "process.");
        return;
    }
    g_registered = true;
    LOGI("VectorHyperRuntime: registered; applications spawned here will be specialized.");

    // Installed here, in the spawner, and inherited by every process it forks. Here rather than in
    // the child is the point: this runs before main, on one thread, with nothing else in the
    // process to race against and no other thread's lock for the fork to have caught mid-hook.
    // Only worth doing now that a callback is coming -- without one, nothing would ever load a
    // module library, and patching a system process's loader for that is a change with no purpose.
    if (!InstallNativeAPI()) {
        LOGW("VectorHyperRuntime: the loader cannot be intercepted; module libraries will still be "
             "loaded on specialization, but later loads will be invisible to them.");
    }

    // One connection, made now and inherited by every child. Children must not ask again: the
    // companion serves a single connection, and a second one would arrive at a companion already
    // busy with this.
    if (api->connectCompanion == nullptr) {
        LOGW("VectorHyperRuntime: the loader offers no companion connection; applications spawned "
             "here will run unhooked.");
        return;
    }
    g_companion_fd = api->connectCompanion(self_handle);
    if (g_companion_fd < 0) {
        LOGW("VectorHyperRuntime: no companion, so the target list cannot be obtained; applications "
             "spawned here will run unhooked.");
    } else {
        // Set here rather than in the child, because it is a property of the connection every child
        // shares and setting it once is enough. The child's read is on the application's startup
        // path; without a bound, a companion that stopped answering would hold the launch open.
        struct timeval timeout{};
        timeout.tv_sec = kCompanionReplyTimeoutSeconds;
        if (setsockopt(g_companion_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0) {
            LOGW("VectorHyperRuntime: cannot bound the companion reply: {}.", strerror(errno));
        }
        LOGI("VectorHyperRuntime: companion connection established on fd {}.", g_companion_fd);

        // The companion's first read is this byte, before it serves anything else, so it has to
        // arrive before the first fork -- which it does, because it is written here and children
        // only exist once the spawner's main runs. It is the answer the daemon will be handed when
        // it asks whether this runtime is being injected at all.
        const char status = g_registered ? 1 : 0;
        if (write(g_companion_fd, &status, 1) != 1) {
            LOGW("VectorHyperRuntime: cannot report the injection status to the companion: {}.",
                 strerror(errno));
        }
    }
}

}  // namespace vector::native::hyos

// =========================================================================================
// Zygisk Next module registration
// =========================================================================================
//
// Both structures are looked up by name in this library when Zygisk Next injects it into
// /system_ext/bin/hyos_spawner, which is what zn_modules.txt asks for. The ordinary Zygisk entry
// point above (REGISTER_ZYGISK_MODULE in module.cpp) is untouched: a library exports as many entry
// points as the loaders loading it need, and the two never run in the same process.

extern "C" __attribute__((visibility("default"))) ZygiskNextModule zn_module = {
    .target_api_version = ZYGISK_NEXT_API_VERSION,
    .onModuleLoaded = vector::native::hyos::OnModuleLoaded,
};

extern "C" __attribute__((visibility("default"))) ZygiskNextCompanionModule zn_companion_module = {
    .target_api_version = ZYGISK_NEXT_API_VERSION,
    .onCompanionLoaded = vector::native::hyos::OnCompanionLoaded,
    .onModuleConnected = vector::native::hyos::OnModuleConnected,
};
