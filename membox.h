// membox.h
#ifndef MEMBOX_H
#define MEMBOX_H

#include <time.h>

void PrintMemBox();
void addMem(int mem_rss);
void printMem(int mem_rss, int mem_heap);

void PrintCheckpoints();
void MemCheck(char* label);

// Print the times
void PrintTimes();
void setTimerValue(const char* name, clock_t start, clock_t end);


#endif // MEMBOX_H