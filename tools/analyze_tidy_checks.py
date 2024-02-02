#!/usr/bin/env python3
"""
================================================================================
 This script analyzes the clang-tidy output and produces an ordered list of all
 the checks run, how many failures a check generated and the percentage of
 failures a check represents.

 When running, make sure that you have already run clang-tidy with all the
 checks you want enabled since this script looks for the 2 tidy_results_*.log
 files in the root directory of Cholla
================================================================================
"""

import numpy as np
import pandas as pd
import pathlib
import subprocess


def main():
    # Determine path to Cholla directory
    chollaPath = pathlib.Path(__file__).resolve().parent.parent

    # Load required data
    tidyResults = loadTidyResults(chollaPath)
    enabledChecks = getEnabledChecks(chollaPath)

    # Count and sort the errors
    sortedChecks, totalWarnings, numPassing, numFailing = countAndSort(
        tidyResults, enabledChecks
    )

    # Print Results in markdown format
    printResults(sortedChecks, totalWarnings, numPassing, numFailing)


def loadTidyResults(chollaPath):
    with open(chollaPath / "tidy_results_cpp.log", "r") as file:
        cppData = file.read()
    with open(chollaPath / "tidy_results_gpu.log", "r") as file:
        gpuData = file.read()

    return cppData + gpuData


def getEnabledChecks(chollaPath):
    stdout = subprocess.run(
        ["clang-tidy", "--list-checks"], cwd=chollaPath, stdout=subprocess.PIPE
    ).stdout.decode("utf-8")

    # find index where checks start
    stdout = stdout.split()
    for i in range(len(stdout)):
        if "bugprone" in stdout[i]:
            index = i
            break

    return stdout[index:]


def countAndSort(tidyResults, enabledChecks):
    passingChecks = 0
    failingChecks = 0
    numWarnings = np.zeros(len(enabledChecks))

    for i, check in enumerate(enabledChecks):
        numWarnings[i] = tidyResults.count(check)
        if check in tidyResults:
            failingChecks += 1
        else:
            passingChecks += 1

    # Convert to dataframe and sort
    sortedChecks = sorted(list(zip(numWarnings, enabledChecks)))
    sortedChecks.reverse()
    totalWarnings = numWarnings.sum()

    return sortedChecks, totalWarnings, passingChecks, failingChecks


def printResults(sortedChecks, totalWarnings, numPassing, numFailing):
    # Determine percentages
    totalChecks = numPassing + numFailing

    print(f"Total number of warnings: {int(totalWarnings)}")
    print(f"{round(numPassing/totalChecks*100, 2)}% of checks passing")
    print(f"{round(numFailing/totalChecks*100, 2)}% of checks failing")

    col1Title = "Number of Warnings"
    col2Title = "Percentage of Warnings"
    col3Title = "Check"
    col3Length = np.max([len(entry[1]) for entry in sortedChecks])

    print()
    print("Failing Checks:")
    print(f"| {col1Title} | {col2Title} | {col3Title:{col3Length}} |")
    print(f'| {"-"*len(col1Title)} | {"-"*len(col2Title)} | {"-"*col3Length} |')
    for entry in sortedChecks:
        if int(entry[0]) != 0:
            print(
                f"| {int(entry[0]):18} | {(entry[0] / totalWarnings)*100:22.2f} | {entry[1]:{col3Length}} |"
            )


if __name__ == "__main__":
    main()
