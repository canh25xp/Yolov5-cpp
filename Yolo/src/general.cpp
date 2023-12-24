#include "yolo/general.hpp"

#include <vector>
#include <string>
#include <filesystem>

#include <imutils/paths.hpp>
#include <imutils/convenience.hpp>

namespace Yolo {

std::vector<std::string> IMG_FORMATS {"bmp", "dng", "jpg", "jpeg", "mpo", "png", "tif", "tiff", "webp", "pfm"};
std::vector<std::string> VID_FORMATS {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"};

std::vector<std::string> IMG_EXTS {".bmp", ".dng", ".jpg", ".jpeg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"};
std::vector<std::string> VID_EXTS {".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".wmv"};

std::vector<std::string> URL_PREFIXES {"rtsp://", "rtmp://", "http://", "https://"};

bool isImage(const std::string& path) {
    std::string ext = path.substr(path.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [] (unsigned char c) { return std::tolower(c); });
    return std::find(IMG_FORMATS.begin(), IMG_FORMATS.end(), ext) != IMG_FORMATS.end();
}

bool isImage(const std::filesystem::path& path) {
    std::string ext = path.extension().string().substr(1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [] (unsigned char c) { return std::tolower(c); });
    return std::find(IMG_FORMATS.begin(), IMG_FORMATS.end(), ext) != IMG_FORMATS.end();
}

bool isVideo(const std::string& path) {
    std::string ext = path.substr(path.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [] (unsigned char c) { return std::tolower(c); });
    return std::find(VID_FORMATS.begin(), VID_FORMATS.end(), ext) != VID_FORMATS.end();
}

bool isVideo(const std::filesystem::path& path) {
    std::string ext = path.extension().string().substr(1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [] (unsigned char c) { return std::tolower(c); });
    return std::find(VID_FORMATS.begin(), VID_FORMATS.end(), ext) != VID_FORMATS.end();
}

bool isFolder(const std::filesystem::path& path) {
    return !path.has_extension();
}

bool isURL(const std::string& path) {
    for (const auto& prefix : URL_PREFIXES) {
        if (path.compare(0, prefix.length(), prefix) == 0) {
            return true;
        }
    }
    return false;
}

std::filesystem::path get_from_url(const std::string& url, const std::filesystem::path& save_dir) {
    return imutils::download_image(url);
}

std::filesystem::path increment_path(const std::filesystem::path& pathStr, bool exist_ok, const std::string& sep, bool mkdir) {
    namespace fs = std::filesystem;

    fs::path path(pathStr);

    if (fs::exists(path) && !exist_ok) {
        fs::path base_path, suffix;
        if (fs::is_regular_file(path)) {
            base_path = path.parent_path() / path.stem();
            suffix = path.extension();
        }
        else
            base_path = path;

        for (int n = 2; n < 9999; ++n) {
            fs::path p = base_path;
            if (!sep.empty())
                p += sep;
            
            p += std::to_string(n);
            if (!suffix.empty())
                p += suffix;

            if (!fs::exists(p)) {
                path = p;
                break;
            }
        }
    }

    if (mkdir)
        fs::create_directories(path);

    return path;
}

std::vector<std::filesystem::path> getListFileDirs(const std::filesystem::path& basePath) {
    return imutils::listFiles(basePath, IMG_EXTS);
}

} // namespace Yolo
