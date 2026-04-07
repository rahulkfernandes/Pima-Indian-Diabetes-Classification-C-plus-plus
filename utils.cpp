#include "utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

using namespace std;

Eigen::MatrixXd load_csv(const string& path, bool has_header = true) {
    ifstream file(path);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + path);
    }

    vector<double> matrixEntries;
    string line;
    int rowCount = 0;

    // Skip header if present
    if (has_header) {
        getline(file, line);  // read and discard
    }

    while (getline(file, line)) {
        stringstream lineStream(line);
        string cell;
        while (getline(lineStream, cell, ',')) {
            // Trim whitespace if any
            size_t start = cell.find_first_not_of(" \t\r\n");
            size_t end = cell.find_last_not_of(" \t\r\n");
            if (start != string::npos && end != string::npos) {
                cell = cell.substr(start, end - start + 1);
            }
            matrixEntries.push_back(stod(cell));
        }
        rowCount++;
    }

    int colCount = matrixEntries.size() / rowCount;
    Eigen::MatrixXd result(rowCount, colCount);
    for (int i = 0; i < rowCount; ++i) {
        for (int j = 0; j < colCount; ++j) {
            result(i, j) = matrixEntries[i * colCount + j];
        }
    }
    return result;
}
