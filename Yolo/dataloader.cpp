#include "dataloader.hpp"

#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

#include <yaml-cpp/yaml.h>

namespace Yolo {

void get_class_names(std::vector<std::string>& class_names, const std::string& dataFile) {
    std::ifstream file(dataFile);
    std::string name = "";
    while (std::getline(file, name)) {
        class_names.push_back(name);
    }
}

void get_class_names_yaml(std::vector<std::string>& class_names, const std::string& data_yaml) {
    YAML::Node data = YAML::LoadFile(data_yaml);

    YAML::Node namesNode = data["names"];

    if (namesNode && namesNode.IsMap()) {
        for (const auto& name : namesNode) {
            class_names.push_back(name.second.as<std::string>());
        }
    }
}

void get_class_names(std::vector<std::string>& class_names, const std::filesystem::path& data) {
    std::string ext = data.extension().string().substr(1);
    if (ext == "yaml")
        get_class_names_yaml(class_names, data.string());
    else if (ext == "txt")
        get_class_names(class_names, data.string());
    // else
    //     LOG("invalid data file");
}

} // namespace Yolo
