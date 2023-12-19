#pragma once
#include <vector>
#include <string>
#include <filesystem>

namespace Yolo {

void get_class_names(std::vector<std::string>& class_names, const std::string& data);

void get_class_names_yaml(std::vector<std::string>& class_names, const std::string& data_yaml);

void get_class_names(std::vector<std::string>& class_names, const std::filesystem::path& data);


} // namespace Yolo
