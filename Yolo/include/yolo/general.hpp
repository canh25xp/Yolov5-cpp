#pragma once
#include <vector>
#include <string>
#include <filesystem>


namespace Yolo {

extern std::vector<std::string> IMG_FORMATS;
extern std::vector<std::string> VID_FORMATS;

bool isImage(const std::string& path);

bool isImage(const std::filesystem::path& path);

bool isVideo(const std::string& path);

bool isVideo(const std::filesystem::path& path);

bool isFolder(const std::filesystem::path& path);

bool isURL(const std::string& path);

std::filesystem::path increment_path(const std::filesystem::path& pathStr, bool exist_ok = false, const std::string& sep = "", bool mkdir = false);

std::vector<std::string> getListFileDirs(const std::filesystem::path& basePath);

} // namespace Yolo
