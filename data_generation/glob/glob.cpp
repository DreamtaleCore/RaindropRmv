#include <iostream>
#include <vector>
#include <glob.h>
#include <fstream>

using namespace std;

void getFiles(const string &pattern, vector<string> &filePath);

int main() {
    vector<string> imgPath;
    getFiles("~/repo/dataset/cityscapes/leftImage/test/*/*.png", imgPath);

    ofstream outFile;
    outFile.open("./path", ios::out);
    for(auto &i : imgPath) {
        outFile << i << endl;
    }
    outFile.close();
    return 0;
}

void getFiles(const string &pattern, vector<string> &filePath) {
    glob_t globBuf;

    glob(pattern.c_str(), GLOB_TILDE, NULL, &globBuf);
    
    for(unsigned i{0}; i < globBuf.gl_pathc; i++) {
        filePath.push_back(globBuf.gl_pathv[i]);
    }

    if(globBuf.gl_pathc > 0) {
        globfree(&globBuf);
    }
}

