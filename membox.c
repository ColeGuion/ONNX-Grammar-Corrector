#include "membox.h"
#include "MoreBin/utils.h"

// Timer names
#define NUM_TIMERS 15
const char* time_names[NUM_TIMERS] = {
    "Load Config",          "Create Geco",          "GEC Preproc",
    "Encoder",              "Main Decoder",         "Past Decoder", 
    "All Decoders",         "Get Max Tokens",       "GEC Postproc",
    "GEC Total",            "Gibb Preproc",         "Gibb Model Run",
    "Calculate Softmax",    "Gibberish Total",      "Destroy Geco", 
};
double allTimes[NUM_TIMERS] = {0};

// Memory holder
int memUsage[20][2] = {0};
char* checkpoints[30] = {"", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""};
int memVals[30] = {0};

void PrintMemBox() {
    Log(DEBUG, "||  \x1b[4mBefore\x1b[0m   ||  \x1b[4mAfter\x1b[0m");
    for (int i=0; i<20; i++) {
        if (memUsage[i][0] != 0) {
            float mem_rss1 = (memUsage[i][0] / 1024.0);
            float mem_rss2 = (memUsage[i][1] / 1024.0);
            if (mem_rss1 == mem_rss2) {
                Log(DEBUG, "||  %-8.2f ||  same", mem_rss1);
            } else if (mem_rss1 > mem_rss2) {
                Log(DEBUG, "||  %-8.2f ||  %-8.2f (\x1b[1;31m-%.2f\x1b[0m)", mem_rss1, mem_rss2, (mem_rss1-mem_rss2));
            } else {
                Log(DEBUG, "||  %-8.2f ||  %-8.2f (\x1b[1;32m+%.2f\x1b[0m)", mem_rss1, mem_rss2, (mem_rss1-mem_rss2));
            }
        }
    }
}
void addMem(int mem_rss) {
    int mem_rss2 = get_memstat("VmRSS");
    for (int i=0; i<20; i++) {
        if (memUsage[i][0] == 0) {
            memUsage[i][0] = mem_rss;
            memUsage[i][1] = mem_rss2;
            return;
        }
    }
}

void PrintCheckpoints() {
    printf("Memory Checkpoints:\n");
    int prev = 0;
    for (int i=0; i<30; i++) {
        if (strcmp(checkpoints[i], "") != 0) {
            int val = memVals[i];
            //float val = (memVals[i] / 1024.0);
            printf("%25s ||  %-10d", checkpoints[i], val);
            if (val > prev) {
                printf(" (\x1b[1;32m+%d\x1b[0m)", (val-prev));
                //printf(" (\x1b[1;32m+%.2f\x1b[0m)", (val-prev));
            } else if (val < prev) {
                printf(" (\x1b[1;31m%d\x1b[0m)", (val-prev));
                //printf(" (\x1b[1;31m%.2f\x1b[0m)", (val-prev));
            }
            printf("\n");
                
            prev = val;//memVals[i];
            checkpoints[i] = ""; 
            memVals[i] = 0;
        }
    }
    printf("\n");
}
void MemCheck(char* label) {
    for (int i=0; i<30; i++) {
        if (strcmp(checkpoints[i], "") == 0) {
            checkpoints[i] = label;
            memVals[i] = get_memstat("VmRSS");
            return;
        }
    }
}

void printMem(int mem_rss, int mem_heap) {
    addMem(mem_rss);
    return;

    int mem_rss2 = get_memstat("VmRSS");
    int mem_heap2 = get_memstat("VmData");

    Log(DEBUG, "       ||  \x1b[4mBefore\x1b[0m  ||  \x1b[4mAfter\x1b[0m");
    
    float diff = (mem_rss / 1024.0) - (mem_rss2 / 1024.0);
    if (mem_rss == mem_rss2) {
        Log(DEBUG, " VmRSS || %-8.2f || same", (mem_rss / 1024.0));
    } else if (mem_rss > mem_rss2) {
        Log(DEBUG, " VmRSS || %-8.2f || %-8.2f (\x1b[1;31m-%.2f\x1b[0m)", (mem_rss / 1024.0), (mem_rss2 / 1024.0), diff);    
    } else {
        Log(DEBUG, " VmRSS || %-8.2f || %-8.2f (\x1b[1;32m+%.2f\x1b[0m)", (mem_rss / 1024.0), (mem_rss2 / 1024.0), (diff*-1));   
    }


    /* diff = (mem_heap / 1024.0) - (mem_heap2 / 1024.0);
    if (mem_heap == mem_heap2) {
        Log(DEBUG, "VmData || %-8.2f || same", (mem_heap / 1024.0));
    } else if (mem_heap > mem_heap2) {
        Log(DEBUG, "VmData || %-8.2f || %-8.2f (\x1b[1;31m-%.2f\x1b[0m)", (mem_heap / 1024.0), (mem_heap2 / 1024.0), diff);    
    } else {
        Log(DEBUG, "VmData || %-8.2f || %-8.2f (\x1b[1;32m+%.2f\x1b[0m)", (mem_heap / 1024.0), (mem_heap2 / 1024.0), (diff*-1));    
    } */
    
    printf("\n");
}


// Print the times
void PrintTimes() {
    Log(DEBUG, "\nTimed Results:");
    for (int i=0; i<NUM_TIMERS; i++) {
        if (allTimes[i] != 0) {
            Log(DEBUG, "   %.5f - %s", allTimes[i], time_names[i]);
            allTimes[i] = 0;
        }
    }
}
void setTimerValue(const char* name, clock_t start, clock_t end) {
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    for (int i = 0; i < NUM_TIMERS; i++) {
        if (strcmp(time_names[i], name) == 0) {
            allTimes[i] += total_time;
            return;
        }
    }
    Log(DEBUG, "Invalid timer name: %s", name);
}

