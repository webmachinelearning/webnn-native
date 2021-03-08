// Copyright (C) 2018-2020 Intel Corporation
// Copyright 2021 The WebNN-native Authors
//
// SPDX-License-Identifier: Apache-2.0
//

#include "MnistUbyte.h"
#include <fstream>
#include <iostream>
#include <string>

int MnistUbyte::ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = (unsigned char)(i & 255);
    ch2 = (unsigned char)((i >> 8) & 255);
    ch3 = (unsigned char)((i >> 16) & 255);
    ch4 = (unsigned char)((i >> 24) & 255);
    return (static_cast<int>(ch1) << 24) + (static_cast<int>(ch2) << 16) +
           (static_cast<int>(ch3) << 8) + ch4;
}

MnistUbyte::MnistUbyte(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "The image file path is invalid, can't open it " << std::endl;
        return;
    }
    int magicNumber = 0;
    int numberOfImages = 0;
    int nRows = 0;
    int nCols = 0;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    magicNumber = ReverseInt(magicNumber);
    if (magicNumber != 2051) {
        return;
    }
    file.read(reinterpret_cast<char*>(&numberOfImages), sizeof(numberOfImages));
    numberOfImages = ReverseInt(numberOfImages);
    file.read(reinterpret_cast<char*>(&nRows), sizeof(nRows));
    nRows = ReverseInt(nRows);
    mHeight = (size_t)nRows;
    file.read(reinterpret_cast<char*>(&nCols), sizeof(nCols));
    nCols = ReverseInt(nCols);
    mWidth = (size_t)nCols;
    if (numberOfImages > 1) {
        std::cout << "[MNIST] Warning: numberOfImages  in mnist file equals " << numberOfImages
                  << ". Only a first image will be read." << std::endl;
    }

    size_t size = mWidth * mHeight * 1;

    mData.reset(new unsigned char[size], std::default_delete<unsigned char[]>());
    size_t count = 0;
    if (0 < numberOfImages) {
        for (int r = 0; r < nRows; ++r) {
            for (int c = 0; c < nCols; ++c) {
                unsigned char temp = 0;
                file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
                mData.get()[count++] = temp;
            }
        }
    }

    file.close();
    mDataInitialized = true;
}
