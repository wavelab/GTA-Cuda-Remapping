/* Copyright (c) 2018, Waterloo Autonomous Vehicles Laboratory (WAVELab),
 * Waterloo Intelligent Systems Engineering (WISE) Lab,
 * University of Waterloo. All Rights Reserved.
 *
 * ############################################################################
 ******************************************************************************
 |                                                                            |
 |                         /\/\__/\_/\      /\_/\__/\/\                       |
 |                         \          \____/          /                       |
 |                          '----________________----'                        |
 |                              /                \                            |
 |                            O/_____/_______/____\O                          |
 |                            /____________________\                          |
 |                           /    (#UNIVERSITY#)    \                         |
 |                           |[**](#OFWATERLOO#)[**]|                         |
 |                           \______________________/                         |
 |                            |_""__|_,----,_|__""_|                          |
 |                            ! !                ! !                          |
 |                            '-'                '-'                          |
 |       __    _   _  _____  ___  __  _  ___  _    _  ___  ___   ____  ____   |
 |      /  \  | | | ||_   _|/ _ \|  \| |/ _ \| \  / |/ _ \/ _ \ /     |       |
 |     / /\ \ | |_| |  | |  ||_||| |\  |||_|||  \/  |||_||||_|| \===\ |====   |
 |    /_/  \_\|_____|  |_|  \___/|_| \_|\___/|_|\/|_|\___/\___/ ____/ |____   |
 |                                                                            |
 ******************************************************************************
 * ############################################################################
 *
 * File: helpers.h
 * Desc: Some simple helper functions
 * Auth: Matt Angus
 *
 * ############################################################################
*/
#pragma once

#include <string>

class Helpers
{
public:
    static bool hasEnding (std::string const &fullString, std::string const &ending) {
        if (fullString.length() >= ending.length()) {
            return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
        } else {
            return false;
        }
    }
    
    static std::vector<std::string> splitpath(const std::string& str, const std::set<char> delimiters)
    {
        std::vector<std::string> result;

        int start = 0;
        for(int i = 0; i < str.length(); i++)
        {
            char cur = str[i];
            if (delimiters.find(cur) != delimiters.end())
            {
                if (start != i)
                {
                    result.push_back(str.substr(start, i - start));
                }
                else
                {
                    result.push_back("");
                }
                start = i + 1;
            }
        }
        result.push_back(str.substr(start));

        return result;
    }

    static std::string remove_extension(const std::string& filename) {
        size_t lastdot = filename.find_last_of(".");
        if (lastdot == std::string::npos) return filename;
        return filename.substr(0, lastdot); 
    }
};
