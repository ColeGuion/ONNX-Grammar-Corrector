#include "my_tester.h"
#include "utils.h"


// Test Cases
TestGibb gibbTests[] = {
    {
        .shape = {7, 4},
        .texts = (char*[]) {
            "Word.",
            "\n",
            "Word.",
            "\n",
            "Word.",
            "\n",
            "Word.",
        },
        .ids = (int*[]) {
            (int[]){101, 2773, 1012, 102},
            (int[]){101, 102, 0, 0},
            (int[]){101, 2773, 1012, 102},
            (int[]){101, 102, 0, 0},
            (int[]){101, 2773, 1012, 102},
            (int[]){101, 102, 0, 0},
            (int[]){101, 2773, 1012, 102},
        },
        .attn_mask = (int*[]) {
            (int[]){1, 1, 1, 1},
            (int[]){1, 1, 0, 0},
            (int[]){1, 1, 1, 1},
            (int[]){1, 1, 0, 0},
            (int[]){1, 1, 1, 1},
            (int[]){1, 1, 0, 0},
            (int[]){1, 1, 1, 1},
        },
        .exp_scores = (double*[]) {
            (double[]){94.5, 4.3, 0.2, 0.9},
            (double[]){94.2, 4.7, 0.2, 0.9},
            (double[]){90.9, 6.9, 0.4, 1.8},
            (double[]){0.0, 0.0, 0.0, 0.0},
            (double[]){96.7, 2.8, 0.1, 0.4},
            (double[]){0.0, 0.0, 0.0, 0.0},
            (double[]){94.3, 4.2, 0.4, 1.1},
        }
    },
    {
        .shape = {6, 12},
        .texts = (char*[]) {
            "He plays the guitar well.",
            "He did well in the exam.",
            "She speaks English well.",
            "Word.",
            "We don't know our neighbor very well.",
            "He did the job well."
        },
        .ids = (int*[]) {
            (int[]){101, 2002, 3248, 1996, 2858, 2092, 1012, 102, 0, 0, 0, 0},
            (int[]){101, 2002, 2106, 2092, 1999, 1996, 11360, 1012, 102, 0, 0, 0},
            (int[]){101, 2016, 8847, 2394, 2092, 1012, 102, 0, 0, 0, 0, 0},
            (int[]){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){101, 2057, 2123, 1005, 1056, 2113, 2256, 11429, 2200, 2092, 1012, 102},
            (int[]){101, 2002, 2106, 1996, 3105, 2092, 1012, 102, 0, 0, 0, 0},
        },
        .attn_mask = (int*[]) {
            (int[]){1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
            (int[]){1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
            (int[]){1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
            (int[]){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            (int[]){1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        },
        .exp_scores = (double*[]) {
            (double[]){94.5, 4.3, 0.2, 0.9},
            (double[]){94.2, 4.7, 0.2, 0.9},
            (double[]){90.9, 6.9, 0.4, 1.8},
            (double[]){0.0, 0.0, 0.0, 0.0},
            (double[]){96.7, 2.8, 0.1, 0.4},
            (double[]){94.3, 4.2, 0.4, 1.1},
        }
    },
    {
        .shape = {14, 39},
        .texts = (char*[]) {
            "If you loook toward tthe westduring sunset and and see a red sky, it mean there is dry aiir and dust particles becase of high-pressure sytem.",
            "This dry airis moving towards you, so there won't be any rain, but the wind whill come soon.",
            "xqwertyuioasd",
            "Word green slide ground red grape John unique water.",
            "Animal pens carpet turquoise salamander spaghetti.",
            "Website hollow words belly under fire truck car limo quickly.",
            "22 mad old Punjab pickle Chennai.",
            "What are you doing?",
            "17 What are 3034 you doing 99?",
            "Upon observation, it was evident that his candy-eating process was more efficient and quicker compared to Caleb Two.",
            "The interlopers' nature is that they carry trees and use them for wood so they can have food or the conflict to have it.",
            "I love mom because she loves \"me.\" \"She\"\" makes \"\"good\" or \"happy.\"\"",
            "asdqwfbeqbfuilac",
            "CreateNewUser",
        },
        .ids = (int*[]) {
            (int[]){101, 2065, 2017, 8840, 14659, 2646, 23746, 5369, 2225, 24979, 2075, 10434, 1998, 1998, 2156, 1037, 2417, 3712, 1010, 2009, 2812, 2045, 2003, 4318, 9932, 4313, 1998, 6497, 9309, 2022, 18382, 1997, 2152, 1011, 3778, 25353, 18532, 1012, 102},
            (int[]){101, 2023, 4318, 2250, 2483, 3048, 2875, 2017, 1010, 2061, 2045, 2180, 1005, 1056, 2022, 2151, 4542, 1010, 2021, 1996, 3612, 1059, 7100, 2272, 2574, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){101, 1060, 4160, 13777, 3723, 10179, 10441, 16150, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){101, 2773, 2665, 7358, 2598, 2417, 14722, 2198, 4310, 2300, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){101, 4111, 25636, 10135, 28653, 16183, 23093, 4063, 26666, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){101, 4037, 8892, 2616, 7579, 2104, 2543, 4744, 2482, 23338, 2855, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){101, 2570, 5506, 2214, 9213, 4060, 2571, 12249, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){101, 2054, 2024, 2017, 2725, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){101, 2459, 2054, 2024, 19988, 2549, 2017, 2725, 5585, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){101, 2588, 8089, 1010, 2009, 2001, 10358, 2008, 2010, 9485, 1011, 5983, 2832, 2001, 2062, 8114, 1998, 19059, 4102, 2000, 10185, 2048, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){101, 1996, 6970, 4135, 7347, 1005, 3267, 2003, 2008, 2027, 4287, 3628, 1998, 2224, 2068, 2005, 3536, 2061, 2027, 2064, 2031, 2833, 2030, 1996, 4736, 2000, 2031, 2009, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){101, 1045, 2293, 3566, 2138, 2016, 7459, 1000, 2033, 1012, 1000, 1000, 2016, 1000, 1000, 3084, 1000, 1000, 2204, 1000, 2030, 1000, 3407, 1012, 1000, 1000, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){101, 2004, 2094, 4160, 2860, 26337, 2063, 4160, 29292, 19231, 6305, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){101, 3443, 2638, 16050, 8043, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        },
        .attn_mask = (int*[]) {
            (int[]){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            (int[]){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            (int[]){1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        },
        .exp_scores = (double*[]) {
            (double[]){2.4, 97.3, 0.0, 0.2},
            (double[]){36.4, 62.1, 0.2, 1.3},
            (double[]){0.1, 0.1, 99.7, 0.1},
            (double[]){0.4, 98.4, 0.1, 1.1},
            (double[]){2.0, 95.4, 0.1, 2.5},
            (double[]){2.0, 83.5, 0.2, 14.3},
            (double[]){7.1, 17.9, 1.1, 73.9},
            (double[]){80.9, 12.7, 1.6, 4.9},
            (double[]){3.6, 9.7, 1.3, 85.4},
            (double[]){52.5, 47.2, 0.1, 0.2},
            (double[]){3.7, 95.8, 0.1, 0.3},
            (double[]){75.9, 15.9, 0.4, 7.8},
            (double[]){0.1, 0.1, 99.8, 0.1},
            (double[]){21.1, 53.0, 10.3, 15.5},
        }
    },
    {
        .shape = {5, 3},
        .texts = (char*[]) {
            "\n\n\n\t   \n\t", 
            "", 
            "Hello", 
            "         ", 
            "World"
        },
        .ids = (int*[]) {
            (int[]){101, 102, 0},
            (int[]){101, 102, 0},
            (int[]){101, 7592, 102},
            (int[]){101, 102, 0},
            (int[]){101, 2088, 102},
        },
        .attn_mask = (int*[]) {
            (int[]){1, 1, 0},
            (int[]){1, 1, 0},
            (int[]){1, 1, 1},
            (int[]){1, 1, 0},
            (int[]){1, 1, 1},
        },
        .exp_scores = (double*[]) {
            (double[]){29.4, 30.3, 16.1, 24.2},
            (double[]){29.4, 30.3, 16.1, 24.2},
            (double[]){26.3, 28.1, 16.1, 29.6},
            (double[]){29.4, 30.3, 16.1, 24.2},
            (double[]){24.2, 28.6, 13.2, 34.0},
        }
    },
}; 
PreProcessingTests tc[] = {
    {
        // "He plays the guitar good. He did good on the exam.    \n\t\t  \n\nShe speaks English good.\n  \t We don't know our neighbor very good. He did the job good.\t\n\t",
        .batchSize = 8,
        .maxTokens = 80,
        .origTexts = (char*[]) {
            "He plays the guitar good.",
            "He did good on the exam.",
            "\n\t\t  \n\n",
            "She speaks English good.",
            "\n  \t ",
            "We don't know our neighbor very good.",
            "He did the job good.",
            "\t\n\t",
        },
        .exp_texts = (char*[]) {
            "He plays the guitar good. He did good on the exam.",
            "She speaks English good.",
            "We don't know our neighbor very good. He did the job good.",
        },
        .exp_num_texts = 3,
        .exp_newLn_size = 3,
        .exp_newLn_strs = (char*[]) {
            "\n\t\t  \n\n",
            "\n  \t ",
            "\t\n\t",
        },
        .exp_newLn_inds = (int[]) {2, 4, 7}
        /* .newTexts = (char*[]) {
            "He plays the guitar well.",
            "He did well on the exam.",
            "\n\t\t  \n\n",
            "She speaks English well.",
            "\n  \t ",
            "We don't know our neighbor very well.",
            "He did the job good.",
            "\t\n\t",
        },
        expected:     "He plays the guitar good. He did good on the exam.\n\t\t  \n\nShe speaks English good.\n  \t We don't know our neighbor very good. He did the job good.\t\n\t", */
    },
    {
        .batchSize = 16,
        .maxTokens = 80,
        .origTexts = (char*[]) {
            "\n\n\n",
            "There are countless different species thriving in this world, including humans.",
            "But due to the rapid expansion and population growth of humans, species in different ecosystems are threatened.",
            "Habitats and food sources are destroyed to clear a way for human structures.",
            "But, there are efforts by humans to preserve the biodivserity observed within our planet, to make it a home for all life.",
            "Orangutans tend to live in tree, and venture on the ground only for water and some food.",
            "When logging and deforestation occurs, the habitats are destroyed.",
            "Poaching and trophy hunting have a massive negative impact.",
            "\n   ",
            "Orangutans are very slow in reproduction, and like humans, have nine-month gestation periods.",
            "After a female births a baby, it take from six to nine years for her to able to have another baby, and twins are extremely rare.",
            "So even if all habitat and food source destruction, or illegal hunting and killing stops, it will take many years for numbers to rejuvenate.",
            "Measures to ensure that cetain habitat areas are reserved for wildlife take place, and animals are taken to rescue facilities, and zoos, to protect them, and study their behavior.",
            "\n",
            "All three species of orangutans are critically endangered and many conservation measures are constantly being taken to defend these creatures now and in the future.",
            "\n",
        },
        .exp_texts = (char*[]) {
            "There are countless different species thriving in this world, including humans. But due to the rapid expansion and population growth of humans, species in different ecosystems are threatened. Habitats and food sources are destroyed to clear a way for human structures.",
            "But, there are efforts by humans to preserve the biodivserity observed within our planet, to make it a home for all life. Orangutans tend to live in tree, and venture on the ground only for water and some food. When logging and deforestation occurs, the habitats are destroyed. Poaching and trophy hunting have a massive negative impact.",
            "Orangutans are very slow in reproduction, and like humans, have nine-month gestation periods. After a female births a baby, it take from six to nine years for her to able to have another baby, and twins are extremely rare.",
            "So even if all habitat and food source destruction, or illegal hunting and killing stops, it will take many years for numbers to rejuvenate. Measures to ensure that cetain habitat areas are reserved for wildlife take place, and animals are taken to rescue facilities, and zoos, to protect them, and study their behavior.",
            "All three species of orangutans are critically endangered and many conservation measures are constantly being taken to defend these creatures now and in the future.",
        },
        .exp_num_texts = 5,
        .exp_newLn_size = 4,
        .exp_newLn_strs = (char*[]) {
            "\n\n\n",
            "\n   ",
            "\n",
            "\n",
        },
        .exp_newLn_inds = (int[]) {0, 3, 6, 8}
    }
};
TestWP wp_tests[] = {
    {
        .shape = {4, 10},
        .texts = (char*[]) {
            "He playsing the guitar well.",
            "xqwertyuioasd",
            "\t\tWe'll  see   ",
            "I can't go shopping today."
        },
        .ids = (int64_t[]) {101, 2002, 3248, 2075, 1996, 2858, 2092, 1012, 102, 0, 101, 1060, 4160, 13777, 3723, 10179, 10441, 16150, 102, 0, 101, 2057, 1005, 2222, 2156, 102, 0, 0, 0, 0, 101, 1045, 2064, 1005, 1056, 2175, 6023, 2651, 1012, 102},
        .attn_mask = (int64_t[]) {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    },
    {
        .shape = {2, 35},
        .texts = (char*[]) {
            "\nMy Biology Essay 1:\n\nA soil tester is needed before any kind of building or structure is started, and is a well payijng, and necessary field.",
            "Did you mean “trees”?"
        },
        .ids = (int64_t[]) {101, 2026, 7366, 9491, 1015, 1024, 1037, 5800, 3231, 2121, 2003, 2734, 2077, 2151, 2785, 1997, 2311, 2030, 3252, 2003, 2318, 1010, 1998, 2003, 1037, 2092, 3477, 28418, 3070, 1010, 1998, 4072, 2492, 1012, 102, 101, 2106, 2017, 2812, 1523, 3628, 1524, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        .attn_mask = (int64_t[]) {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    },
    {
        .shape = {14, 39},
        .texts = (char*[]) {
            "If you loook toward tthe westduring sunset and and see a red sky, it mean there is dry aiir and dust particles becase of high-pressure sytem.",
            "This dry airis moving towards you, so there won't be any rain, but the wind whill come soon.",
            "xqwertyuioasd",
            "Word green slide ground red grape John unique water.",
            "Animal pens carpet turquoise salamander spaghetti.",
            "Website hollow words belly under fire truck car limo quickly.",
            "22 mad old Punjab pickle Chennai.",
            "What are you doing?",
            "17 What are 3034 you doing 99?",
            "Upon observation, it was evident that his candy-eating process was more efficient and quicker compared to Caleb Two.",
            "The interlopers' nature is that they carry trees and use them for wood so they can have food or the conflict to have it.",
            "I love mom because she loves \"me.\" \"She\"\" makes \"\"good\" or \"happy.\"\"",
            "asdqwfbeqbfuilac",
            "CreateNewUser",
        },
        .ids = (int64_t[]) {101, 2065, 2017, 8840, 14659, 2646, 23746, 5369, 2225, 24979, 2075, 10434, 1998, 1998, 2156, 1037, 2417, 3712, 1010, 2009, 2812, 2045, 2003, 4318, 9932, 4313, 1998, 6497, 9309, 2022, 18382, 1997, 2152, 1011, 3778, 25353, 18532, 1012, 102, 101, 2023, 4318, 2250, 2483, 3048, 2875, 2017, 1010, 2061, 2045, 2180, 1005, 1056, 2022, 2151, 4542, 1010, 2021, 1996, 3612, 1059, 7100, 2272, 2574, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 1060, 4160, 13777, 3723, 10179, 10441, 16150, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 2773, 2665, 7358, 2598, 2417, 14722, 2198, 4310, 2300, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 4111, 25636, 10135, 28653, 16183, 23093, 4063, 26666, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 4037, 8892, 2616, 7579, 2104, 2543, 4744, 2482, 23338, 2855, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 2570, 5506, 2214, 9213, 4060, 2571, 12249, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 2054, 2024, 2017, 2725, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 2459, 2054, 2024, 19988, 2549, 2017, 2725, 5585, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 2588, 8089, 1010, 2009, 2001, 10358, 2008, 2010, 9485, 1011, 5983, 2832, 2001, 2062, 8114, 1998, 19059, 4102, 2000, 10185, 2048, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 1996, 6970, 4135, 7347, 1005, 3267, 2003, 2008, 2027, 4287, 3628, 1998, 2224, 2068, 2005, 3536, 2061, 2027, 2064, 2031, 2833, 2030, 1996, 4736, 2000, 2031, 2009, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 1045, 2293, 3566, 2138, 2016, 7459, 1000, 2033, 1012, 1000, 1000, 2016, 1000, 1000, 3084, 1000, 1000, 2204, 1000, 2030, 1000, 3407, 1012, 1000, 1000, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 2004, 2094, 4160, 2860, 26337, 2063, 4160, 29292, 19231, 6305, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 3443, 2638, 16050, 8043, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        .attn_mask = (int64_t[]) {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    },
    {
        .shape = {5,12},
        .texts = (char*[]) {
            "He plays the guitar well.",
            "He did well in the exam.",
            "She speaks English well.",
            "We don't know our neighbor very well.",
            "He did the job well."
        },
        .ids = (int64_t[]) {101, 2002, 3248, 1996, 2858, 2092, 1012, 102, 0, 0, 0, 0, 101, 2002, 2106, 2092, 1999, 1996, 11360, 1012, 102, 0, 0, 0, 101, 2016, 8847, 2394, 2092, 1012, 102, 0, 0, 0, 0, 0, 101, 2057, 2123, 1005, 1056, 2113, 2256, 11429, 2200, 2092, 1012, 102, 101, 2002, 2106, 1996, 3105, 2092, 1012, 102, 0, 0, 0, 0},
        .attn_mask = (int64_t[]) {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}
    },
    /* {
        .shape = {},
        .texts = (char*[]) {},
        .ids = (int64_t[]) {},
        .attn_mask = (int64_t[]) {}
    }, */
};

// GEC Test Cases
TestCase tests[] = {
    {
        .batchSize = 5,
        .avgTime = 0.55,
        .avgTimeGpu = 0.115,
        .texts = (char*[]) {
            "He plays the guitar well.",
            "He did well in the exam.",
            "She speaks English well.",
            "We don't know our neighbor very well.",
            "He did the job well."
        },
        .expect = (char*[]) {
            "He plays the guitar well.",
            "He did well in the exam.",
            "She speaks English well.",
            "We don't know our neighbor very well.",
            "He did the job well."
        },
        .exp_text = "He plays the guitar well. He did well in the exam. She speaks English well. We don't know our neighbor very well. He did the job well."
    },
    {
        .batchSize = 2,
        .avgTime = 1.55,
        .avgTimeGpu = 0.194,
        .texts = (char*[]) {
            "If you loook toward tthe westduring sunset and and see a red sky, it mean there is dry aiir and dust particles becase of high-pressure sytem.",
            "This dry airis moving towards you, so there won't be any rain, but the wind whill come soon."
        },
        .expect = (char*[]) {
            "If you look toward the west during sunset and see a red sky, it means there is dry air and dust particles in the atmosphere, as in high-pressure systems.",
            "This dry air is moving towards you, so there won't be any rain, but the wind will come soon."
        },
        .exp_text = "If you look toward the west during sunset and see a red sky, it means there is dry air and dust particles in the atmosphere, as in high-pressure systems. This dry air is moving towards you, so there won't be any rain, but the wind will come soon."
    },
    {
        .batchSize = 3,
        .avgTime = 1.24,
        .avgTimeGpu = 0.144,
        .texts = (char*[]) {
            "Hello world?",
            "Sharon is the biggest pool lover I know?",
            "Sharon is the nicest person ever, Before she had time to think about it Sharon jumped in an icy pool"
        },
        .expect = (char*[]) {
            "Hello world,",
            "Sharon is the biggest pool lover I know.",
            "Sharon is the nicest person ever. Before she had time to think about it, Sharon jumped in an icy pool."
        },
        .exp_text = "Hello world, Sharon is the biggest pool lover I know. Sharon is the nicest person ever. Before she had time to think about it, Sharon jumped in an icy pool."
    },
    {
        // Test #4 (Essay 1)
        .batchSize = 10,
        .avgTime = 3.1,
        .avgTimeGpu = 0.789,
        .texts = (char*[]) {
            "Directional selection is one of the the three types of natural selection that selects one extreme of a trait.",
            "When a flower produces more nectar or seeds so that it's population grows, therefore it is more successful.",
            "When a population is larger, it's harder to sway the genetic traits of the population.",
            "When the population is smaller, it's easier to directionalize genes to be more successful.",
            "Instead of balancing an average of all traits, organisms choose an extreme of one trait, and that causes animals to evovle in such direction.",
            "In the example, the flowers evovled to attract more pollinators to carry around the specific gene so that all members in a population recieve that gene, making the population more successful.",
            "However, since the population is evovled to one gene, this makes it more susceptible to diseases or unhealthy factors that said organism didn't evolve to.",
            "I don't really know what I'm talking about, and I'm kind of frustrated.",
            "I don't really want to write about this anymore, and I already have a good grade in this class.",
            "All I want to do is get over it and say that organisms evovle really fast with directional selection."
        },
        .expect = (char*[]) {
            "Directional selection is one of the three types of natural selection that selects one extreme of a trait.",
            "When a flower produces more nectar or seeds so that its population grows, it is more successful.",
            "When a population is larger, it's harder to influence the genetic traits of the population.",
            "When the population is smaller, it's easier to directionalize genes to be more successful.",
            "Instead of balancing an average of all traits, organisms choose an extreme of one trait, and that causes animals to behave in such a direction.",
            "In this example, the flowers are evovled to attract more pollinators to carry around the specific gene so that all members of a population receive that gene, making the population more successful.",
            "However, since the population is evovled to one gene, this makes it more susceptible to diseases or unhealthy factors that said organism didn't evolve to.",
            "I don't really know what I'm talking about, and I'm kind of frustrated.",
            "I don't really want to write about this anymore, and I already have a good grade in this class.",
            "All I want to do is get over it and say that organisms evovle really fast with directional selection."
        },
        .exp_text = "Directional selection is one of the three types of natural selection that selects one extreme of a trait. When a flower produces more nectar or seeds so that its population grows, it is more successful. When a population is larger, it's harder to influence the genetic traits of the population. When the population is smaller, it's easier to directionalize genes to be more successful. Instead of balancing an average of all traits, organisms choose an extreme of one trait, and that causes animals to behave in such a direction. In this example, the flowers are evovled to attract more pollinators to carry around the specific gene so that all members of a population receive that gene, making the population more successful. However, since the population is evovled to one gene, this makes it more susceptible to diseases or unhealthy factors that said organism didn't evolve to. I don't really know what I'm talking about, and I'm kind of frustrated. I don't really want to write about this anymore, and I already have a good grade in this class. All I want to do is get over it and say that organisms evovle really fast with directional selection."
    },
    {
        // Test #5 (Essay 4)
        .batchSize = 16,
        .avgTime = 2.92,
        .avgTimeGpu = 0.553,
        .texts = (char*[]) {
            "\n\n\n",
            "There are countless different species thriving in this world, including humans.",
            "But due to the rapid expansion and population growth of humans, species in different ecosystems are threatened.",
            "Habitats and food sources are destroyed to clear a way for human structures.",
            "But, there are efforts by humans to preserve the biodivserity observed within our planet, to make it a home for all life.",
            "Orangutans tend to live in tree, and venture on the ground only for water and some food.",
            "When logging and deforestation occurs, the habitats are destroyed.",
            "Poaching and trophy hunting have a massive negative impact.",
            "\n   ",
            "Orangutans are very slow in reproduction, and like humans, have nine-month gestation periods.",
            "After a female births a baby, it take from six to nine years for her to able to have another baby, and twins are extremely rare.",
            "So even if all habitat and food source destruction, or illegal hunting and killing stops, it will take many years for numbers to rejuvenate.",
            "Measures to ensure that cetain habitat areas are reserved for wildlife take place, and animals are taken to rescue facilities, and zoos, to protect them, and study their behavior.",
            "\n",
            "All three species of orangutans are critically endangered and many conservation measures are constantly being taken to defend these creatures now and in the future.",
            "\n",
        },
        .expect = (char*[]) {
            "\n\n\n",
            "There are countless different species thriving in this world, including humans.",
            "But due to the rapid expansion and population growth of humans, species in different ecosystems are threatened.",
            "Habitats and food sources are destroyed to clear a way for human structures.",
            "But, there are efforts by humans to preserve the biodiversity observed within our planet and to make it a home for all life.",
            "Orangutans tend to live in trees and venture out on the ground only for water and some food.",
            "When logging and deforestation occurs, the habitat is destroyed.",
            "Poaching and trophy hunting have a massive negative impact.",
            "\n   ",
            "Orangutans are very slow in reproduction and, like humans, have nine-month gestation periods.",
            "After a woman births a baby, it takes from six to nine years for her to be able to have another baby, and twins are extremely rare.",
            "So even if all habitat and food source destruction, or illegal hunting and killing, stops, it will take many years for numbers to rejuvenate.",
            "Measures to ensure that cetain habitat areas are reserved for wildlife are taking place, and animals are taken to rescue facilities and zoos, to protect them, and study their behavior.",
            "\n",
            "All three species of orangutans are critically endangered, and many conservation measures are constantly being taken to defend these creatures now and in the future.",
            "\n",
        },
        .exp_text = "\n\n\nThere are countless different species thriving in this world, including humans. But due to the rapid expansion and population growth of humans, species in different ecosystems are threatened. Habitats and food sources are destroyed to clear a way for human structures. But, there are efforts by humans to preserve the biodiversity observed within our planet, to make it a home for all life. Orangutans tend to live in trees and venture on the ground only for water and some food. When logging and deforestation occurs, the habitats are destroyed. Poaching and trophy hunting have a massive negative impact.\n   Orangutans are very slow in reproduction, and like humans, have nine-month gestation periods. After a woman births a baby, it takes from six to nine years for her to be able to have another baby, and twins are extremely rare. So even if all habitat and food source destruction, or illegal hunting and killing, it will take many years for numbers to rejuvenate. Measures to ensure that cetain habitat areas are reserved for wildlife have been taken place, and animals are taken to rescue facilities and zoos, to protect them, and study their behavior.\nAll three species of orangutans are critically endangered, and many conservation measures are constantly being taken to defend these creatures now and in the future.\n"
    },
    {
        // Test #6 (More 2)
        .batchSize = 20,
        .avgTime = 1.4,
        .avgTimeGpu = 0.163,
        .texts = (char*[]) {
            "This is good lasagna!",
            "Today, at last, life is good.",
            "She is a good singer.",
            "We are good students.",
            "He is a good listener.",
            "They are good neighbors.",
            "He did a good job.",
            "The food tastes good.",
            "The house smells good.",
            "She looks good in that dress.",
            "The car appears good on the exterior.",
            "The idea seems good to me.",
            "He didn't feel good when he lied to his mom.",
            "I'm not feeling good about the test results.",
            "He feels good about the decision.",
            "We feel good about our choice for candidate.",
            "My eyesight is good.",
            "This is a good movie.",
            "What a good idea!",
            "You speak good English."
        },
        .expect = (char*[]) {
            "This is a good lasagna!",
            "Today, at last, life is good.",
            "She is a good singer.",
            "We are good students.",
            "He is a good listener.",
            "They are good neighbors.",
            "He did a good job.",
            "The food tastes good.",
            "The house smells good.",
            "She looks good in that dress.",
            "The car appears good on the exterior.",
            "The idea seems good to me.",
            "He didn't feel good when he lied to his mom.",
            "I'm not feeling good about the test results.",
            "He feels good about the decision.",
            "We feel good about our choice of candidate.",
            "My eyesight is good good.",
            "This is a good movie.",
            "What a good idea!",
            "You speak good English."
        }
    },
    {
        // Test #7 (More 6)
        .batchSize = 17,
        .avgTime = 2.69,
        .avgTimeGpu = 0.423,
        .texts = (char*[]) {
            "Golly I'm trying to decide among these shirts.",
            "The bees were buzzing among the flowers.",
            "He smiled, revealing fangs among the neat row of white teeth.",
            "Iliana has been a favorite among them.",
            "The sailors divided his money among themselves, and the ship sailed on.",
            "Yes, among other things.",
            "But the five phenomena I chose to tackle in this book are among the great blights on humanity that I believe the Internet and technology will help solve.",
            "Among Mrs. Marsh's attributes was mind reading.",
            "We'd discussed among ourselves numerous times over the past months.",
            "I'm happy you're among us.",
            "The people, with Petya among them, rushed toward the balcony.",
            "Jule is still not in favor among my kind.",
            "He strolled among the self-conscious pets and people, smiling and looking until he was sure she wasn't there.",
            "We agreed among ourseleves.",
            "Divide this among yourselves.",
            "Tom is ill at ease among strangers.",
            "Shew was chosen from among many students."
        },
        .expect = (char*[]) {
            "Golly, I'm trying to decide among these shirts.",
            "The bees were buzzing among the flowers.",
            "He smiled, revealing fangs among the neat row of white teeth.",
            "Iliana has been a favorite among them.",
            "The sailors divided his money among themselves, and the ship sailed on.",
            "Yes, among other things.",
            "But the five phenomena I chose to tackle in this book are among the great blights on humanity that I believe the Internet and technology will help solve.",
            "Among Mrs. Marsh's attributes was mind reading.",
            "We'd discussed among ourselves numerous times over the past months.",
            "I'm happy you're among us.",
            "The people, with Petya among them, rushed toward the balcony.",
            "Jule is still not in favor among my kind.",
            "He strolled among the self-conscious pets and people, smiling and looking until he was sure she wasn't there.",
            "We agreed among our selected candidates.",
            "Divide this among yourselves.",
            "Tom is ill at ease among strangers.",
            "Shew was chosen from among many students."
        }
    },
    {
        // Test #8 (Glitch Test #2)      (128 'j' characters in commented .expect)
        .batchSize = 3,
        .avgTime = 4.086,
        .avgTimeGpu = 0.612,
        .texts = (char*[]) {
            "A b c d e f g h i j k l m n o p q r s t u v w x y z I'm it'd how'd couldn't we'd.",
            "j j j j j j j j.",
            "Thank u."
        },
        .expect = (char*[]) {
            "A b c d e f g h i j k l m n o p q r s t u v w x y z we couldn't we'd.",
            "j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j ",
            //"j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j",
            "Thank you."
        }
    },
    {
        // Test #9 (Glitch Test #2 without long first sequence)
        .batchSize = 2,
        .avgTime = 1.529,
        .avgTimeGpu = 0.167,
        .texts = (char*[]) {
            "j j j j j j j j.",
            "Thank u."
        },
        .expect = (char*[]) {
            "j j j j j j j j j j j j j j j j j j ",
            "Thank you."
        }
    },
    {
        // Test #10 (JFK Speech) (Length of texts: [222, 496, 45, 84, 300, 61, 22, 232])
        .batchSize = 8,
        .avgTime = 8.366,
        .avgTimeGpu = 2.623,
        .texts = (char*[]) {
            "The people of America, through Congress, felt that in the passing of this Act, that the veterans of World War II would not be the recipients of the neglect and the indifference experienced by the veterans of World War I.",
            "The single dominant thought of the public was that the men and women who offered their lives in the most terrible of all wars should be assured a full share in the traditional American Life which they fought to defend; that there should be a generous measure of that free opportunity which is the basis of the American Way of Life; that this Servicemen's Readjustment Act should be a master rehabilitation plan and that it would provide a scientific approach to the veteran problem of this war.",
            "Some of the public even felt that this G.I.",
            "Bill of Rights would repay a man for the fighting and sacrifices that he had made.",
            "A large part of the public felt that, with the passage of this Servicemen's Readjustment Act, the veteran would be given all those things for which he fought; that it would provide him with an opportunity to get ahead by his own efforts and abilities unhampered by private or government compulsion.",
            "What are the facts in regards to veterans' hospitalization?",
            "In theory, this G.I.",
            "Bill of Rights has authorized the appropriation of $500,000,000 for hospitals and proper medical and psychiatric treatment for our veterans; that no veterans would be without competent medical attention and proper hospitalization.",
        },
        .expect = (char*[]) {
            "The people of America, through Congress, felt that in the passing of this Act, the veterans of World War II would not be the recipients of the neglect and the indifference experienced by the veterans of World War I.",
            "The single dominant thought of the public was that the men and women who offered their lives in the most terrible of all wars should be assured a full share in the traditional American Life which they fought to defend; that there should be a generous measure of that free opportunity which is the basis of the American Way of Life; that this Servicemen's Readjustment Act should be a master rehabilitation plan and that it would provide a scientific approach to the veteran problem of this war.",
            "Some of the public even felt that this G.I.",
            "The Bill of Rights would repay a man for the fighting and sacrifices that he had made.",
            "A large part of the public felt that, with the passage of this Servicemen's Readjustment Act, the veteran would be given all those things for which he fought; that it would provide him with an opportunity to get ahead with his own efforts and abilities, unhampered by private or government compulsion.",
            "What are the facts regarding veterans' hospitalization?",
            "In theory, this G.I.",
            "The Bill of Rights authorized the appropriation of $500,000,000 for hospitals and proper medical and psychiatric treatment for our veterans; that no veteran would be without competent medical attention and proper hospitalization."
        }
    },
    {
        // Test #11 (Two large sequences) (Lengths: [222, 496])
        .batchSize = 2,
        .avgTime = 14,
        .avgTimeGpu = 0.631,
        .texts = (char*[]) {
            "The people of America, through Congress, felt that in the passing of this Act, that the veterans of World War II would not be the recipients of the neglect and the indifference experienced by the veterans of World War I.",
            "The single dominant thought of the public was that the men and women who offered their lives in the most terrible of all wars should be assured a full share in the traditional American Life which they fought to defend; that there should be a generous measure of that free opportunity which is the basis of the American Way of Life; that this Servicemen's Readjustment Act should be a master rehabilitation plan and that it would provide a scientific approach to the veteran problem of this war.",
        },
        .expect = (char*[]) {
            "The people of America, through Congress, felt that in the passing of this Act, the veterans of World War II would not be the recipients of the neglect and the indifference experienced by the veterans of World War I.",
            "The single dominant thought of the public was that the men and women who offered their lives in the most terrible of all wars should be assured a full share in the traditional American Life which they fought to defend; that there should be a generous measure of that free opportunity which is the basis of the American Way of Life; that this Servicemen's Readjustment Act should be a master rehabilitation plan and that it would provide a scientific approach to the veteran problem of this war.",
        }
    },
    {
        // Test #12 (40 small sequences)
        .batchSize = 40,
        .avgTime = 14,
        .avgTimeGpu = 0.291,
        .texts = (char*[]) {
            "Time heals almost everything; give it time.",
            "Small steps lead to big changes.",
            "Patience is bitter, but its fruit is sweet.",
            "Focus on what you can control.",
            "Growth starts outside your comfort zone.",
            "Silence can be more powerful than words.",
            "Kindness costs nothing but means everything.",
            "Courage is the birthplace of confidence.",
            "Your vibe attracts your tribe.",
            "A goal without a plan is just a wish.",
            "Stay curious, and keep learning.",
            "Progress is better than perfection.",
            "Actions always speak louder than words.",
            "The journey matters more than the destination.",
            "Dream big, but start small.",
            "Stay humble and stay grounded.",
            "Time flies; make each moment count.",
            "Success is built on small daily wins.",
            "Worry less, smile more.",
            "Your mind is your most powerful tool.",
            "Change is hard at first, messy in the middle.",
            "Mistakes are proof you're trying.",
            "Celebrate even small victories.",
            "Gratitude turns what we have into enough.",
            "Your attitude defines your altitude.",
            "Don't count the days; make the days count.",
            "Today's pain is tomorrow's strength.",
            "Success is a journey, not a destination.",
            "Embrace the beauty of imperfection.",
            "Your only limit is your mind.",
            "You are stronger than you think.",
            "Never stop being curious.",
            "Hustle until your haters ask if you're hiring.",
            "A single act of kindness is never wasted.",
            "Focus on solutions, not problems.",
            "True happiness starts from within.",
            "Don't let fear decide your future.",
            "Positivity is a powerful choice.",
            "Be the energy you want to attract.",
            "Good things take time, be patient.",    
        },
        .expect = (char*[]) {
            "Time heals almost everything; give it time.",
            "Small steps lead to big changes.",
            "Patience is bitter, but its fruit is sweet.",
            "Focus on what you can control.",
            "Growth starts outside your comfort zone.",
            "Silence can be more powerful than words.",
            "Kindness costs nothing but means everything.",
            "Courage is the birthplace of confidence.",
            "Your vibe attracts your tribe.",
            "A goal without a plan is just a wish.",
            "Stay curious and keep learning.",
            "Progress is better than perfection.",
            "Actions always speak louder than words.",
            "The journey matters more than the destination.",
            "Dream big, but start small.",
            "Stay humble and stay grounded.",
            "Time flies; make each moment count.",
            "Success is built on small daily wins.",
            "Worry less, smile more.",
            "Your mind is your most powerful tool.",
            "Change is hard at first, messy in the middle.",
            "Mistakes are proof you're trying.",
            "Celebrate even small victories.",
            "Gratitude turns what we have into enough.",
            "Your attitude defines your altitude.",
            "Don't count the days; make the days count.",
            "Today's pain is tomorrow's strength.",
            "Success is a journey, not a destination.",
            "Embrace the beauty of imperfection.",
            "Your only limit is your mind.",
            "You are stronger than you think.",
            "Never stop being curious.",
            "Hustle until your haters ask if you're hiring.",
            "A single act of kindness is never wasted.",
            "Focus on solutions, not problems.",
            "True happiness starts from within.",
            "Don't let fear decide your future.",
            "Positivity is a powerful choice.",
            "Be the energy you want to attract.",
            "Good things take time; be patient.", 
        }
    },
    {
        // Test #13
        .batchSize = 63,
        .avgTime = 5,
        .avgTimeGpu = 10,
        .texts = (char*[]) {
            "The cat scratched me when I tried to pet her.",
            "Will you please give me the rest of your ice cream cone?",
            "Give the recipe to Jim and me, and we'll cook you a delicious breakfast.",
            "The gift is for me.",
            "    \n\n  \t\n\t", //* 4
            "She told me to go away.",
            "He told Tom and me to get ready.",
            "Just between you and me, this is a bad idea.",
            "The comedic section of the play kills me every time.",
            "I heard he had a bone to pick between you and me.",
            "He'll come to the store with you and me.",
            "The voice inside me says this is a bad idea.",
            "The cat that belongs to me is orange.",
            "Julie accidentally hit me with her bag as she walked by.",
            "Henry told Tran and me to wait for him.",
            "He was bullying me and my friend.",
            "Kevin smiled at me.",
            "Cheryl and her kids gave the card to me in person.",
            "The bird flew over Ben and me before landing in the tree.",
            "The new student decided to sit with me and Kim and lunch.",
            "Just give me a minute.",
            "\n",   //* 21
            "Keep the third piece of wisdom for your own use, and let me have the gold.",
            "\n",   //* 23
            "What did you want me to say?",
            "If they give me plenty of it, I'll not complain about its color.",
            "Will you give me a cup of tea?",
            "And---pardon me for the foolish question---but, are you invisible?",
            "Follow me, please, to meet your doom.",
            "I have no money with me.",
            "John throws the ball to me.",
            "John throws me the ball.",
            "He gave me a cold!",
            "You're under no obligation to listen to me.",
            "Are you going to the party with Ricardo and me?",
            "Rose spent the day with Jake and me.",
            "Please remind him or me.",
            "Between you and me, I think Sandy cheated.",
            "Arlene asked him and me to complete the job.",
            "I hit the ball.",
            "He and I will meet at the gym.",
            "He and I completed the job for Arlene.",
            "I went to the movies.",
            "\n\n\n",   //* 43
            "Sally spoke to Jane and me.",
            "I will lead you to it.",
            "I'm so glad I have you.",
            "\"I don't know,\" said Zeb, who was still confused.",
            "I love you so much.",
            "I have faith only in god and the lofty destiny of our adored monarch.",
            "I suppose they're both a little artificial.",
            "\n",   //* 51
            "I don't like him.",
            "They tell me I walked the day I was a year old.",
            "I don't know why he is so secretive.",
            "The second time I saw my father's naked brutality he came at my mother-I mean the second time I physically witnessed my father looking more animal than man, his embodied rage-he threw a coffee mug at her head.",
            "From then on the gap between my parents and me widened as I realized that, as well as an intangible intellectual world different to the one I had grown up in, there rwas an actual social world too.",
            "I remember how relaxing it felt when I first went to New York, to be in a place where everyone was in a hurry all the time.",
            "She knew I wouldn't repeat the information at school.",
            "The boy started to respond but I didn't hear it, because I launched into a coughing fit.",
            "She looked at me to corroborate but I couldn't speak.",
            "I saw black spots and my head spun and I couldn't stay upright.",
            "I'll save your life--so be as I can--from them.",
        },
        .expect = (char*[]) {
            "The cat scratched me when I tried to pet her.",
            "Will you please give me the rest of your ice cream cone?",
            "Give the recipe to Jim and me, and we'll cook you a delicious breakfast.",
            "The gift is for me.",
            "    \n\n  \t\n\t",
            "She told me to go away.",
            "He told Tom and me to get ready.",
            "Just between you and me, this is a bad idea.",
            "The comedic section of the play kills me every time.",
            "I heard he had a bone to pick between you and me.",
            "He'll come to the store with you and me.",
            "The voice inside me says this is a bad idea.",
            "The cat that belongs to me is orange.",
            "Julie accidentally hit me with her bag as she walked by.",
            "Henry told Tran and me to wait for him.",
            "He was bullying me and my friend.",
            "Kevin smiled at me.",
            "Cheryl and her kids gave the card to me in person.",
            "The bird flew over Ben and me before landing in the tree.",
            "The new student decided to sit with me and Kim and lunch.",
            "Just give me a minute.",
            "\n",
            "Keep the third piece of wisdom for your own use, and let me have the gold.",
            "\n",
            "What did you want me to say?",
            "If they give me plenty of it, I'll not complain about its color.",
            "Will you give me a cup of tea?",
            "And---pardon me for the foolish question---but, are you invisible?",
            "Follow me, please, to meet your doom.",
            "I have no money with me.",
            "John throws the ball to me.",
            "John throws me the ball.",
            "He gave me a cold!",
            "You're under no obligation to listen to me.",
            "Are you going to the party with Ricardo and me?",
            "Rose spent the day with Jake and me.",
            "Please remind him or me.",
            "Between you and me, I think Sandy cheated.",
            "Arlene asked him and me to complete the job.",
            "I hit the ball.",
            "He and I will meet at the gym.",
            "He and I completed the job for Arlene.",
            "I went to the movies.",
            "\n\n\n",
            "Sally spoke to Jane and me.",
            "I will lead you to it.",
            "I'm so glad I have you.",
            "\"I don't know,\" said Zeb, who was still confused.",
            "I love you so much.",
            "I have faith only in god and the lofty destiny of our adored monarch.",
            "I suppose they're both a little artificial.",
            "\n",
            "I don't like him.",
            "They tell me I walked the day I was a year old.",
            "I don't know why he is so secretive.",
            "The second time I saw my father's naked brutality he came at my mother-I mean the second time I physically witnessed my father looking more animal than man, his embodied rage-he threw a coffee mug at her head.",
            "From then on the gap between my parents and me widened as I realized that, as well as an intangible intellectual world different to the one I had grown up in, there rwas an actual social world too.",
            "I remember how relaxing it felt when I first went to New York, to be in a place where everyone was in a hurry all the time.",
            "She knew I wouldn't repeat the information at school.",
            "The boy started to respond but I didn't hear it, because I launched into a coughing fit.",
            "She looked at me to corroborate but I couldn't speak.",
            "I saw black spots and my head spun and I couldn't stay upright.",
            "I'll save your life--so be as I can--from them.",
        }
    },
    {
        // Test #14
        .batchSize = 15,
        .avgTime = 5,
        .avgTimeGpu = 10,
        .texts = (char*[]) {
            "The cat scratched me when I tried to pet her. Will you please give me the rest of your ice cream cone? Give the recipe to Jim and me, and we'll cook you a delicious breakfast. The gift is for me.",
            "    \n\n  \t\n\t",
            "She told me to go away. He told Tom and me to get ready. Just between you and me, this is a bad idea. The comedic section of the play kills me every time. I heard he had a bone to pick between you and me. He'll come to the store with you and me. The voice inside me says this is a bad idea. The cat that belongs to me is orange.",
            "Julie accidentally hit me with her bag as she walked by. Henry told Tran and me to wait for him. He was bullying me and my friend. Kevin smiled at me. Cheryl and her kids gave the card to me in person. The bird flew over Ben and me before landing in the tree. The new student decided to sit with me and Kim and lunch. Just give me a minute.",
            "\n",
            "Keep the third piece of wisdom for your own use, and let me have the gold.",
            "\n",
            "What did you want me to say? If they give me plenty of it, I'll not complain about its color. Will you give me a cup of tea? And---pardon me for the foolish question---but, are you invisible? Follow me, please, to meet your doom. I have no money with me. John throws the ball to me. John throws me the ball. He gave me a cold!",
            "You're under no obligation to listen to me. Are you going to the party with Ricardo and me? Rose spent the day with Jake and me. Please remind him or me. Between you and me, I think Sandy cheated. Arlene asked him and me to complete the job. I hit the ball. He and I will meet at the gym. He and I completed the job for Arlene. I went to the movies.",
            "\n\n\n",
            "Sally spoke to Jane and me. I will lead you to it. I'm so glad I have you. \"I don't know,\" said Zeb, who was still confused. I love you so much. I have faith only in god and the lofty destiny of our adored monarch. I suppose they're both a little artificial.",
            "\n",
            "I don't like him. They tell me I walked the day I was a year old. I don't know why he is so secretive. The second time I saw my father's naked brutality he came at my mother-I mean the second time I physically witnessed my father looking more animal than man, his embodied rage-he threw a coffee mug at her head.",
            "From then on the gap between my parents and me widened as I realized that, as well as an intangible intellectual world different to the one I had grown up in, there rwas an actual social world too. I remember how relaxing it felt when I first went to New York, to be in a place where everyone was in a hurry all the time. She knew I wouldn't repeat the information at school.",
            "The boy started to respond but I didn't hear it, because I launched into a coughing fit. She looked at me to corroborate but I couldn't speak. I saw black spots and my head spun and I couldn't stay upright. I'll save your life--so be as I can--from them.",
        },
        .expect = (char*[]) {
            "The cat scratched me when I tried to pet her. Will you please give me the rest of your ice cream cone? Give the recipe to Jim and me, and we'll cook you a delicious breakfast. The gift is for me.",
            "    \n\n  \t\n\t",
            "She told me to go away. He told Tom and me to get ready. Just between you and me, this is a bad idea. The comedic section of the play kills me every time. I heard he had a bone to pick between you and me. He'll come to the store with you and me. The voice inside me says this is a bad idea. The cat that belongs to me is orange.",
            "Julie accidentally hit me with her bag as she walked by. Henry told Tran and me to wait for him. He was bullying me and my friend. Kevin smiled at me. Cheryl and her kids gave the card to me in person. The bird flew over Ben and me before landing in the tree. The new student decided to sit with me and Kim and lunch. Just give me a minute.",
            "\n",
            "Keep the third piece of wisdom for your own use, and let me have the gold.",
            "\n",
            "What did you want me to say? If they give me plenty of it, I'll not complain about its color. Will you give me a cup of tea? And---pardon me for the foolish question---but, are you invisible? Follow me, please, to meet your doom. I have no money with me. John throws the ball to me. John throws me the ball. He gave me a cold!",
            "You're under no obligation to listen to me. Are you going to the party with Ricardo and me? Rose spent the day with Jake and me. Please remind him or me. Between you and me, I think Sandy cheated. Arlene asked him and me to complete the job. I hit the ball. He and I will meet at the gym. He and I completed the job for Arlene. I went to the movies.",
            "\n\n\n",
            "Sally spoke to Jane and me. I will lead you to it. I'm so glad I have you. \"I don't know,\" said Zeb, who was still confused. I love you so much. I have faith only in god and the lofty destiny of our adored monarch. I suppose they're both a little artificial.",
            "\n",
            "I don't like him. They tell me I walked the day I was a year old. I don't know why he is so secretive. The second time I saw my father's naked brutality he came at my mother-I mean the second time I physically witnessed my father looking more animal than man, his embodied rage-he threw a coffee mug at her head.",
            "From then on the gap between my parents and me widened as I realized that, as well as an intangible intellectual world different to the one I had grown up in, there rwas an actual social world too. I remember how relaxing it felt when I first went to New York, to be in a place where everyone was in a hurry all the time. She knew I wouldn't repeat the information at school.",
            "The boy started to respond but I didn't hear it, because I launched into a coughing fit. She looked at me to corroborate but I couldn't speak. I saw black spots and my head spun and I couldn't stay upright. I'll save your life--so be as I can--from them.",
        }
    },
    {
        // Test #15
        .batchSize = 1,
        .avgTime = 5,
        .avgTimeGpu = 10,
        .texts = (char*[]) { "we shood buy an car", },
        .expect = (char*[]) {""}
    },
    {
        // Test #16
        .batchSize = 3,
        .avgTime = 5,
        .avgTimeGpu = 10,
        .texts = (char*[]) {
            "Hello, world.",
            "\\x1b[32mAdditional Sentence.",
            "\\x1b[0mMy name is John."
        },
        .expect = (char*[]) {""}
    },
    {
        // Test #17
        .batchSize = 11,
        .avgTime = 5,
        .avgTimeGpu = 10,
        .texts = (char*[]) {
            "I am inspired by the way Luna sees the world in a different way.",
            "\n",
            "For example,",
            "\n",
            "s",
            "\n",
            "he sees beauty in everything, even the most scary creatures. (She thinks/behaves like this)",
            "\n",
            "I am not a.",
            "\n",
            "Hi my name is Amerigo"
        },
        .expect = (char*[]) {""}
    }
    /* {
        .batchSize = ,
        .avgTime = 50.5,
        .avgTimeGpu = 10,
        .texts = (char*[]) {
            
        },
        .expect = (char*[]) {
        
        }
    } */
}; 

TestGec badTests[] = {
    {
        // Test #1 - "we shood buy an car"
        .batchSize = 1,
        .texts = (char*[]) { "we shood buy an car", },
        .exp = "Wehood buy a car.",
    },
    {
        // Test #2 - Escape chars
        .batchSize = 3,
        .texts = (char*[]) {
            "Hello, world.",
            "\\x1b[32mAdditional Sentence.",
            "\\x1b[0mMy name is John."
        },
        .exp = "Hello, world. x1b[32mAdditional Sentence. x1b[0mMy name is John.",
    },
    {
        .batchSize = 11,
        .texts = (char*[]) {
            "I am inspired by the way Luna sees the world in a different way.",
            "\n",
            "For example,",
            "\n",
            "s",
            "\n",
            "he sees beauty in everything, even the most scary creatures. (She thinks/behaves like this)",
            "\n",
            "I am not a.",
            "\n",
            "Hi my name is Amerigo"
        },
        .exp = "I am inspired by the way Luna sees the world in a different way.\nFor example,\ns s s\nHe sees beauty in everything, even the most scary creatures. (She thinks/behaves like this).\nI am not a.\nHi, my name is Amerigo.",
    },
    {
        // Essay #1: "Directional selection is one of the the three types of natural selection that selects one extreme of a trait. When a flower produces more nectar or seeds so that it's population grows, therefore it is more successful. When a population is larger, it's harder to sway the genetic traits of the population. When the population is smaller, it's easier to directionalize genes to be more successful. Instead of balancing an average of all traits, organisms choose an extreme of one trait, and that causes animals to evovle in such direction. In the example, the flowers evovled to attract more pollinators to carry around the specific gene so that all members in a population recieve that gene, making the population more successful. However, since the population is evovled to one gene, this makes it more susceptible to diseases or unhealthy factors that said organism didn't evolve to. I don't really know what I'm talking about, and I'm kind of frustrated. I don't really want to write about this anymore, and I already have a good grade in this class. All I want to do is get over it and say that organisms evovle really fast with directional selection."
        .batchSize = 10,
        .texts = (char*[]) {
            "Directional selection is one of the the three types of natural selection that selects one extreme of a trait.",
            "When a flower produces more nectar or seeds so that it's population grows, therefore it is more successful.",
            "When a population is larger, it's harder to sway the genetic traits of the population.",
            "When the population is smaller, it's easier to directionalize genes to be more successful.",
            "Instead of balancing an average of all traits, organisms choose an extreme of one trait, and that causes animals to evovle in such direction.",
            "In the example, the flowers evovled to attract more pollinators to carry around the specific gene so that all members in a population recieve that gene, making the population more successful.",
            "However, since the population is evovled to one gene, this makes it more susceptible to diseases or unhealthy factors that said organism didn't evolve to.",
            "I don't really know what I'm talking about, and I'm kind of frustrated.",
            "I don't really want to write about this anymore, and I already have a good grade in this class.",
            "All I want to do is get over it and say that organisms evovle really fast with directional selection.",
        },
        .exp = "Directional selection is one of the three types of natural selection that selects one extreme of a trait. When a flower produces more nectar or seeds so that its population grows, it is more successful. When a population is larger, it's harder to change the genetic traits of the population. When the population is smaller, it's easier to directionalize genes to be more successful. Instead of balancing an average of all traits, organisms choose an extreme of one trait, and that causes animals to behave in such a direction. In the example, the flowers are evovled to attract more pollinators to carry around the specific gene so that all members of a population receive that gene, making the population more successful. However, since the population is evovled to one gene, this makes it more susceptible to diseases or unhealthy factors that said organism didn't evolve to. I don't really know what I'm talking about, and I'm kind of frustrated. I don't really want to write about this anymore, and I already have a good grade in this class. All I want to do is get over it and say that organisms evolve really fast with directional selection.",
    },
    {
        // Essay #2: "A marine biome is typically the most common biome, simply because most of the earth is covered in water, and marine biomes require water to be called marine biomes. Hundreds of millions of species from all classes of animals, live in freshwater biomes, saltwater biomes, or even a misture of both. However, it's the way that nutrients flow through each of the biomes. In a food chain, producers like phytoplankton, algae, and seaweed are considered the lowest in a nutrient food chain (unless sunlight is counted). Animals commonly known as primary consumers, such as shrimp or krill, eat the producers and gain most of the nutrient energy from them. Then, the secondary consumers, like cod, or tuna, or trout, eat the primaries, and if in a big enough ecosystem, the tertiary consumers, top predators like seals or sharks, gain nutrients and energy from the secondary consumers. Finally, the decomposers break down dead material, and return nutrients to the ecosystem. Intriguingly, the amount of nutrients and energy decreases as it move up the food chain, of food web. That is how nutrients are distributed throughout all ecosystems"
        .batchSize = 9,
        .texts = (char*[]) {
            "A marine biome is typically the most common biome, simply because most of the earth is covered in water, and marine biomes require water to be called marine biomes.",
            "Hundreds of millions of species from all classes of animals, live in freshwater biomes, saltwater biomes, or even a misture of both.",
            "However, it's the way that nutrients flow through each of the biomes.",
            "In a food chain, producers like phytoplankton, algae, and seaweed are considered the lowest in a nutrient food chain (unless sunlight is counted).",
            "Animals commonly known as primary consumers, such as shrimp or krill, eat the producers and gain most of the nutrient energy from them.",
            "Then, the secondary consumers, like cod, or tuna, or trout, eat the primaries, and if in a big enough ecosystem, the tertiary consumers, top predators like seals or sharks, gain nutrients and energy from the secondary consumers.",
            "Finally, the decomposers break down dead material, and return nutrients to the ecosystem.",
            "Intriguingly, the amount of nutrients and energy decreases as it move up the food chain, of food web.",
            "That is how nutrients are distributed throughout all ecosystems"
        },
        .exp = "A marine biome is typically the most common biome, simply because most of the earth is covered in water, and marine biomes require water to be called marine biomes. Hundreds of millions of species, from all classes of animals, live in freshwater biomes, saltwater biomes, or even a mixture of both. However, it's the way that nutrients flow through each of the biomes. In a food chain, producers like phytoplankton, algae, and seaweed are considered the lowest in a nutrient food chain (unless sunlight is counted). Animals commonly known as primary consumers, such as shrimp or krill, eat the producers and gain most of the nutrient energy from them. Then, the secondary consumers, like cod, or tuna, or trout, eat the primaries, and if there is a big enough ecosystem, the tertiary consumers, top predators like seals or sharks, gain nutrients and energy from the secondary consumers. Finally, the decomposers break down dead material and return nutrients to the ecosystem. Intriguingly, the amount of nutrients and energy decreases as they move up the food chain, or food web, which is how nutrients are distributed throughout all ecosystems.",
    },
    {
        // Test #5
        .batchSize = 10,
        .texts = (char*[]) {
            "It stopped raining so we can finally go to the park.",
            "The engine started back up again so we could drive home.",
            "I'm taking my dog for a walk in the rain so I might get wet.",
            "Tomorrow it will snow so we will wear a heavy coat.",
            "He got a number right on the lottery so he might win some money.",
            "The cats have played all day today so they might fall asleep soon.",
            "The power went out in my neighborhood so I might not be able to cook later.",
            "He forgot to do his readings tonight so he will study in the morning.",
            "Her gas tank is filled up so she can make the drive across town.",
            "She is not feeling well so she might not make it to the wedding later."
        },
        .exp = "It stopped raining so we can finally go to the park. The engine started back up again so we could drive home. I'm taking my dog for a walk in the rain so I might get wet. Tomorrow it will snow, so we will wear a heavy coat. He got a number right in the lottery so he might win some money. The cats have played all day today so they might fall asleep soon. The power went out in my neighborhood so I might not be able to cook later. He forgot to do his readings tonight so he will study in the morning. Her gas tank is filled up so she can make the drive across town. She is not feeling well so she might not make it to the wedding later.",
    },
    {
        // Test #6
        .batchSize = 58,
        .texts = (char*[]) {
            "The cat scratched me when I tried to pet her.",
            "Will you please give me the rest of your ice cream cone?",
            "Give the recipe to Jim and me, and we'll cook you a delicious breakfast.",
            "The gift is for me.",
            "She told me to go away.",
            "He told Tom and me to get ready.",
            "Just between you and me, this is a bad idea.",
            "The comedic section of the play kills me every time.",
            "I heard he had a bone to pick between you and me.",
            "He'll come to the store with you and me.",
            "The voice inside me says this is a bad idea.",
            "The cat that belongs to me is orange.",
            "Julie accidentally hit me with her bag as she walked by.",
            "Henry told Tran and me to wait for him.",
            "He was bullying me and my friend.",
            "Kevin smiled at me.",
            "Cheryl and her kids gave the card to me in person.",
            "The bird flew over Ben and me before landing in the tree.",
            "The new student decided to sit with me and Kim and lunch.",
            "Just give me a minute.",
            "Keep the third piece of wisdom for your own use, and let me have the gold.",
            "What did you want me to say?",
            "If they give me plenty of it, I'll not complain about its color.",
            "Will you give me a cup of tea?",
            "And---pardon me for the foolish question---but, are you invisible?",
            "Follow me, please, to meet your doom.",
            "I have no money with me.",
            "John throws the ball to me.",
            "John throws me the ball.",
            "He gave me a cold!",
            "You're under no obligation to listen to me.",
            "Are you going to the party with Ricardo and me?",
            "Rose spent the day with Jake and me.",
            "Please remind him or me.",
            "Between you and me, I think Sandy cheated.",
            "Arlene asked him and me to complete the job.",
            "I hit the ball.",
            "He and I will meet at the gym.",
            "He and I completed the job for Arlene.",
            "I went to the movies.",
            "Sally spoke to Jane and me.",
            "I will lead you to it.",
            "I'm so glad I have you.",
            "\"I don't know,\" said Zeb, who was still confused.",
            "I love you so much.",
            "I have faith only in god and the lofty destiny of our adored monarch.",
            "I suppose they're both a little artificial.",
            "I don't like him.",
            "They tell me I walked the day I was a year old.",
            "I don't know why he is so secretive.",
            "The second time I saw my father's naked brutality he came at my mother-I mean the second time I physically witnessed my father looking more animal than man, his embodied rage-he threw a coffee mug at her head.",
            "From then on the gap between my parents and me widened as I realized that, as well as an intangible intellectual world different to the one I had grown up in, there rwas an actual social world too.",        
            "I remember how relaxing it felt when I first went to New York, to be in a place where everyone was in a hurry all the time.",
            "She knew I wouldn't repeat the information at school.",
            "The boy started to respond but I didn't hear it, because I launched into a coughing fit.",
            "She looked at me to corroborate but I couldn't speak.",
            "I saw black spots and my head spun and I couldn't stay upright.",
            "I'll save your life--so be as I can--from them.",
        },
        .exp = "The cat scratched at me when I tried to pet her. Will you please give me the rest of your ice cream cone? Give the recipe to Jim and me, and we'll cook you a delicious breakfast. The gift is for me. She told me to go away. He told Tom and me to get ready. Just between you and me, this is a bad idea. The comedic section of the play kills me every time. I heard he had a bone to pick between you and me. He'll come to the store with you and me. The voice inside me says this is a bad idea. The cat that belongs to me is orange. Julie accidentally hit me with her bag as she walked by. Henry told Tran and me to wait for him. He was bullying me and my friend. Kevin smiled at me. Cheryl and her kids gave me the card in person. The bird flew over Ben and me before landing in the tree. The new student decided to sit with me and Kim and lunch. Just give me a minute. Keep the third piece of wisdom for your own use, and let me have the gold. What did you want me to say? If they give me plenty of it, I'll not complain about its color. Will you give me a cup of tea? And---pardon me for the foolish question---are you invisible? Follow me, please, to meet your doom. I have no money with me. John throws the ball to me. John throws me the ball. He gave me a cold! You're under no obligation to listen to me. Are you going to the party with Ricardo and me? Rose spent the day with Jake and me. Please remind him or me. Between you and me, I think Sandy cheated. Arlene asked him and me to complete the job. I hit the ball. He and I will meet at the gym. He and I completed the job for Arlene. I went to the movies. Sally spoke to Jane and me. I will lead you to it. I'm so glad I have you. \"I don't know,\" said Zeb, who was still confused. I love you so much. I have faith only in god and the lofty destiny of our adored monarch. I suppose they're both a little artificial. I don't like him. They tell me I walked the day I was a year old. I don't know why he is so secretive. The second time I saw my father's naked brutality, he came at my mother-I mean the second time I physically witnessed my father looking more animal than man, with his embodied rage-he threw a coffee mug at her head. From then on, the gap between my parents and me widened as I realized that, as well as an intangible intellectual world different to the one I had grown up in, there was an actual social world too. I remember how relaxing it felt when I first went to New York, to be in a place where everyone was in a hurry all the time. She knew I wouldn't repeat the information at school. The boy started to respond but I didn't hear it because I launched into a coughing fit. She looked at me to corroborate, but I couldn't speak. I saw black spots, and my head spun, and I couldn't stay upright. I'll save your life--so be as I can--from them.",
    },
    {
        // Test #7
        .batchSize = 37,
        .texts = (char*[]) {
            "Joan likes eggs; Jennifer does not.",
            "If you bring your sunglasses, sunscreen, and a towel; we can go to the beach.",
            "The groups of siblings who will be coming to camp include John and Anne; Jeff, Lisa, and Tommy; and Mark and Jonas.",
            "I have lived in Atlanta, GA; Charleston, SC; and Tallahassee, FL.",
            "The address for the letter is PO Box 37; Martin, NY 30065.",
            "Marie made a 100 on the quiz; Lois made a 85.",
            "It was raining; the game was cancelled.",
            "I like bacon, eggs, and cheese; but not all together on a sandwich.",
            "I always try to pack light for a vacation; however, I always seem to need an extra bag for all of my shoes and books.",
            "We had too many fumbles; we lost the game.",
            "Call me tomorrow; you can give me an answer then.",
            "We have paid our dues; we expect all privileges listed in the contract.",
            "Bring any two items; however, sleeping bags and tents are in short supply.",
            "I bought shiny, ripe apples; small, sweet, juicy grapes; and firm pairs.",
            "I went to the grocery store today; I bought a ton of fruit; apples, grapes, and pears were all on sale.",
            "We can go the museum to do some research; Mondays are pretty quit there.",
            "I ordered a cheeseburger for lunch; life's too short for counting calories.",
            "Money is the root of all evil; I don't believe the reverse is necessarily true.",
            "Martha has gone to the library; Andrew has gone to play soccer.",
            "I saw a magnificent albatross; it was eating a mouse.",
            "This is part of the same rule; the conjunction in question is \"but\" instead of \"and.\"",
            "My plan included taking him to a nice --- though not necessarily expensive --- dinner; going to the park to look at the stars, which, by the way, are amazing this time of ear; and serenading him with my accordion.",
            "I needed to go for a walk and get some fresh air; also, I needed to buy milk.",
            "Reports of the damage caused by the hurricane were greatly exaggerated; indeed, the storm was not a \"hurricane\" at all.",
            "The students had been advised against walking alone at night; however, Cathy decided walking wasn't dangerous if it was early in the evening.",
            "I'm not all that fond of the colors of tiger lilies; moreover, they don't smell very good.",
            "I love ice cream; it is my favorite food.",
            "I like cake; however, ice cream is my favorite dessert.",
            "I know great ice cream shops in Burlington, Vermont; Wickford, Rhode Island; Wakefield, Rhode Island; and Chester, New Jersey.",
            "Dessert is the best meal of the day; it's definitely my favorite!",
            "There is one thing I know; ice cream is the best dessert.",
            "Sometimes I have frozen yogurt; however, it's not as good as ice cream.",
            "They were out of Rocky Road; thus, I was forced to choose another flavor.",
            "John has lived in Atlanta, Georgia; Seattle, Washington, and Miami, Florida.",
            "Rocky Road has chocolate, peanuts, and marshmallows; cookies and cream has chocolate sandwich cookies; Neapolitan has chocolate, vanilla, and strawberry in one.",
            "I have a big test tomorrow; I can't go out tonight.",
            "This week's winners are Joe from Reno, Nevada; Diane from Phoenix, Arizona; and Matt from Irvine, California.",
        },
        .exp = "Joan likes eggs; Jennifer does not. If you bring your sunglasses, sunscreen, and a towel we can go to the beach. The groups of siblings who will be coming to camp include John and Anne; Jeff, Lisa, and Tommy; and Mark and Jonas. I have lived in Atlanta, GA; Charleston, SC; and Tallahassee, FL. The address for the letter is PO Box 37; Martin, NY 30065. Marie made a 100 on the quiz; Lois made an 85. It was raining; the game was cancelled. I like bacon, eggs, and cheese, but not all together on a sandwich. I always try to pack light for a vacation; however, I always seem to need an extra bag for all of my shoes and books. We had too many fumbles; we lost the game. Call me tomorrow; you can give me an answer then. We have paid our dues; we expect all privileges listed in the contract. Bring two items; however, sleeping bags and tents are in short supply. I bought shiny, ripe apples, small, sweet, juicy grapes, and firm pairs. I went to the grocery store today and bought a ton of fruit; apples, grapes, and pears were all on sale. We can go to the museum to do some research; Mondays are pretty quiet there. I ordered a cheeseburger for lunch; life's too short for counting calories. Money is the root of all evil; I don't believe the reverse is necessarily true. Martha has gone to the library; Andrew has gone to play soccer. I saw a magnificent albatross; it was eating a mouse. This is part of the same rule; the conjunction in question is \"but\" instead of \"and.\" My plan included taking him to a nice — though not necessarily expensive — dinner; going to the park to look at the stars, which, by the way, are amazing this time of year; and serenading him with my accordion. I needed to go for a walk and get some fresh air; also, I needed to buy milk. Reports of the damage caused by the hurricane were greatly exaggerated; indeed, the storm was not a \"hurricane\" at all. The students had been advised against walking alone at night; however, Cathy decided walking wasn't dangerous if it was early in the evening. I'm not all that fond of the colors of the tiger lilies; moreover, they don't smell very good. I love ice cream; it is my favorite food. I like cake; however, ice cream is my favorite dessert. I know great ice cream shops in Burlington, Vermont; Wickford, Rhode Island; Wakefield, Rhode Island; and Chester, New Jersey. Dessert is the best meal of the day; it's definitely my favorite! There is one thing I know: ice cream is the best dessert. Sometimes I have frozen yogurt; however, it's not as good as ice cream. They were out of Rocky Road; thus, I was forced to choose another flavor. John has lived in Atlanta, Georgia; Seattle, Washington, and Miami, Florida. Rocky Road has chocolate, peanuts, and marshmallows; Cookies and Cream has chocolate sandwich cookies; Neapolitan has chocolate, vanilla, and strawberry in one. I have a big test tomorrow; I can't go out tonight. This week's winners are Joe from Reno, Nevada; Diane from Phoenix, Arizona; and Matt from Irvine, California.",
    },
    {
        // Test #8
        .batchSize = 25,
        .texts = (char*[]) {
            "Boom, rattle, and pop!",
            "A pine tree grew directly in front of me!",
            "It sparkled in the sunlight, and I could tell it was magical.",
            "\n",
            "I thought I bot normal seeds.",
            "\n",
            "I exclaimed, \"Oh my goodness!\"",
            "I started looking around, and I could see that this was not an ordinary forest.",
            "\n\n",
            "On the path, there were dozens of tiny creatures pouring out bottles of green goo.",
            "As the goo hit the ground, suddenly pine trees popped out of the dirt!",
            "\n\n",
            "I ran over to one of the creatures, and I asked,\"How are you doing this sorcery.",
            "\n\n",
            "The creature answered, \"This is magical snail mucus that comes from snail trees.",
            "their snails that look like giant trees,\".",
            "\n\n",
            "After that, I said \"Mabye I,d grow giant if I ate some\".",
            "\n",
            "I went over to the tree I planted.",
            "There was a hole full of green slime.",
            "\n\n",
            "\"That's the stuff\" The creature said.",
            "\n",
            "I was about to toss the gloop in my mouth.",
        },
        .exp = "Boom, rattle, and pop! A pine tree grew directly in front of me! It sparkled in the sunlight, and I could tell it was magical.\nI thought I had planted normal seeds.\nI exclaimed, \"Oh my goodness!\" I started looking around, and I could see that this was not an ordinary forest.\n\nOn the path, there were dozens of tiny creatures pouring out bottles of green goo. As the goo hit the ground, suddenly, pine trees popped out of the dirt!\n\nI ran over to one of the creatures, and I asked,\"How are you doing this sorcery?\"\n\nThe creature answered, \"This is magical snail mucus that comes from snail trees.\" Their snails that look like giant trees.\n\nAfter that, I said, \"Mabye, I'd grow giant if I ate some.\"\nI went over to the tree I planted. There was a hole full of green slime.\n\n\"That's the stuff,\" the creature said.\nI was about to toss the gloop in my mouth."
    },
    {
        // Test #9
        .batchSize = 37,
        .texts = (char*[]) {
            "Greta: I absolutely agree with you!",
            "\n",
            "Theo: Our pizza is going to be so disgusting!!!",
            "\n",
            "Mr. McFudge: What about we add chocolate fudge.",
            "\n",
            "Elise: Again with the fudge!",
            "\n",
            "Mr. McFudge: But I love Cake.",
            "\n",
            "Elise: That makes no sense.",
            "\n",
            "Greta: Lets get baking!",
            "(Tossing a pan and catching it)",
            "\n",
            "Mr. McFudge: I'll get something.",
            "\n",
            "Elise: That's what I was afraid of.",
            "\n",
            "Greta: Why?",
            "\n",
            "Elise: You'll see.",
            "\n",
            "Mr. McFudge: I'm back!",
            "(Slamming the door and dumping a bag of chocolate fudge in the dough)",
            "\n",
            "Greta: let's finish.",
            "\n",
            "They all topped the pizza.",
            "\n",
            "Mr. McFudge: Sprinkes!",
            "(Shaking a giant gar of sprinkles on-to the pizza)",
            "\n",
            "Greta: Let's taste.",
            "\n",
            "Theo: Mmm candy pizza.",
            "(Eating a slice of the pizza)",
        },
        .exp = "Greta: I absolutely agree with you!\nTheo: Our pizza is going to be so disgusting!!!\nMr. McFudge: What about adding chocolate fudge?\nElise: Again with the fudge!\nMr. McFudge: But I love cake.\nElise: That makes no sense.\nGreta: Lets get baking! (Tossing a pan and catching it).\nMr. McFudge: I'll get something.\nElise: That's what I was afraid of.\nGreta: Why?\nElise: You'll see.\nMr. McFudge: I'm back! (Slamming the door and dumping a bag of chocolate fudge into the dough).\nGreta: let's finish.\nThey all topped the pizza.\nMr. McFudge: Sprinkes! (Shaking a giant gar of sprinkles on the pizza).\nGreta: Let's taste.\nTheo: Mmm. Candy pizza. (Eating a slice of the pizza)."
    },
    {
        // Test #10
        .batchSize = 5,
        .texts = (char*[]) {
            "He plays the guitar well.",
            "He did well in the exam.",
            "She speaks English well.",
            "We don't know our neighbor very well.",
            "He did the job well."
        },
        .exp = "He plays the guitar well. He did well in the exam. She speaks English well. We don't know our neighbor very well. He did the job well."
    },
    {
        // Test #11
        .batchSize = 11,
        .texts = (char*[]) {
            "I am inspired by the way Luna sees the world in a different way.",
            "\n",
            "For example,",
            "\n",
            "s",
            "\n",
            "he sees beauty in everything, even the most scary creatures. (She thinks/behaves like this)",
            "\n",
            "I am not a.",
            "\n",
            "Hi my name is Amerigo"
        },
        .exp = "I am inspired by the way Luna sees the world in a different way.\nFor example,\ns s s\nHe sees beauty in everything, even the most scary creatures. (She thinks/behaves like this).\nI am not a.\nHi, my name is Amerigo.",
    },
    {
        // Test #12: Only includes the first 10 lines (Ignores the last two lines)
        .batchSize = 10,
        .texts = (char*[]) {
            //"As the sun began to set behind the distant hills, casting a warm golden glow across the landscape, the children raced down the hill, their laughter echoing through the air like music, while their parents watched from a nearby picnic blanket, sipping lemonade and reminiscing about their own childhood adventures, which seemed to blend together in a haze of nostalgia, filled with memories of carefree days spent playing outside until dusk, climbing trees and exploring the woods, where they discovered hidden treasures like shiny rocks and unusual insects, all of which made them feel alive and connected to nature in a way that was both exhilarating and comforting.",
            //"As the sun began to set behind",
            //"Hello world",
            "Hello World!",
            "\n",
            "How are you today?",
            "\n",
            "Hi there World!",
            "\n",
            "\n\n\n",
            "\n",
            "\n\n\n",
            "How are you today?",
            "\n",
            "Hi there World!",
        },
        .exp = "Hello World!\nHow are you today?\nHi there World!\nHow are you today?"
    },
    /* {
        .batchSize = 1,
        .texts = (char*[]) {
            ""
        },
        .exp = ""
    } */
};

// Function to run a test and time its execution
void speedTest(void* gecoObj, int testNum, int enableGpu) {
    TestCase test = tests[testNum-1];
    double margin = 0.15;   // CPU

    // Time the InferModel functionH
    clock_t start = clock();
    char* result = GecoRun(gecoObj, test.texts, test.batchSize);
    clock_t end = clock();
    printf("Final Result: \n\"\"\"\n%s\n\"\"\"\n\n", result);

    // Calculate the time difference
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    double timeDiff = test.avgTime - cpu_time_used;

    if (enableGpu) {
        margin = 0.02;   // GPU
        timeDiff = test.avgTimeGpu - cpu_time_used;
    }

    if (timeDiff > margin) {
        printf("Test #%d - Time taken: \x1b[1;32m%.3f seconds\x1b[0m\n", testNum, cpu_time_used);
    } else if (timeDiff < -margin) {
        printf("Test #%d - Time taken: \x1b[1;31m%.3f seconds\x1b[0m\n", testNum, cpu_time_used);
    } else {
        printf("Test #%d - Time taken: %.3f seconds\n", testNum, cpu_time_used);
    }

    printf("\n");
    free(result);
}

void runSpeedTests(void* geco, int numTests, int enableGpu) {
    // Run all test cases and time their execution
    if ((size_t)numTests > (sizeof(tests) / sizeof(tests[0]))) {
        numTests = sizeof(tests) / sizeof(tests[0]);
    }
    for (int i = 0; i < numTests; i++) {
        speedTest(geco, i+1, enableGpu);
    }
}
// Run your own GEC test and time the output
void customRun(void* gecoObj, char** allTexts, int batchSize) {
    clock_t start = clock();
    char* result = GecoRun(gecoObj, allTexts, batchSize);
    clock_t end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("\n\nTime taken: %.3f seconds\n", cpu_time_used);
    printf("Final Result: \x1b[1;32m\"%s\"\x1b[0m\n\n", result);

    free(result);
}


void prtResult(char* result, char* expect, double timeTaken) {
    int col = 2;

    if (strcmp(expect, "") == 0 && strcmp(expect, result) != 0) {
        col = 1;
    }

    if (col == 2) {
        printf("Result: \x1b[1;3%dmPASSED\x1b[0m  (%.3f sec)\n", col, timeTaken);
    } else {
        printf("Result: (%.3f sec)\n\x1b[1;3%dm\"\"\"\n%s\n\"\"\"\x1b[0m\n\n", timeTaken, col, result);
    }
    //printf("\n\n");
    //get_memory_usage();
    printf("\x1b[1;93m---------------------------------------------------------------------------------------------------------------------------------------\x1b[0m\n\n");
}

// Run a test from TestGec
void test_gec(void* geco, bool runAll) {
    // Run the latest test case if numTests is not specified
    size_t tests_length = sizeof(badTests) / sizeof(badTests[0]);
    size_t start_ind = 0;
    if (!runAll) {
        start_ind = tests_length-1;
    }

    for (size_t i = start_ind; i < tests_length; i++) {
        TestGec myTest = badTests[i];
        printf("\x1b[1;93m=======================================================   Running Test #%zu   =======================================================\x1b[0m\n", i);
    
        clock_t start = clock();
        char* result = GecoRun(geco, myTest.texts, myTest.batchSize);
        clock_t end = clock();
        double timeTaken = ((double) (end - start)) / CLOCKS_PER_SEC;

        //writeToFile("results.txt", i, result);    // Write the result to "results.txt"
        prtResult(result, myTest.exp, timeTaken);
        if (result) free(result);
    }
}

void test_gec2(void* geco, int n) {
    // Get test number
    int total_tests = (int)(sizeof(badTests) / sizeof(badTests[0]));

    for (int i=0; i<n; i++) {
        int testNum = total_tests - i - 1;
        TestGec myTest = badTests[testNum];
        printf("\x1b[1;93m=======================================================   Running Test #%d   =======================================================\x1b[0m\n", testNum);

        clock_t start = clock();
        char* result = GecoRun(geco, myTest.texts, myTest.batchSize);
        clock_t end = clock();
        double timeTaken = ((double) (end - start)) / CLOCKS_PER_SEC;

        prtResult(result, myTest.exp, timeTaken);
        if (result) free(result);
    }
}
void test_gec3(void* geco, int testNum) {
    // Get test number
    TestGec myTest = badTests[testNum];
    printf("\x1b[1;93m=======================================================   Running Test #%d   =======================================================\x1b[0m\n", testNum);

    clock_t start = clock();
    char* result = GecoRun(geco, myTest.texts, myTest.batchSize);
    clock_t end = clock();
    double timeTaken = ((double) (end - start)) / CLOCKS_PER_SEC;

    prtResult(result, myTest.exp, timeTaken);
    if (result) free(result);
}

//* Preprocessing Tests
void prtNewlineLits(char* str) {
    for (int k = 0; k < (int)strlen(str); k++) {
        switch (str[k]) {
            case '\n':
                printf("\\n");
                break;
            case '\t':
                printf("\\t");
                break;
            case ' ':
                printf(" ");
                break;
            default:
                printf("%c", str[k]);
                break;
        }
    }
}
/* void preprocTests(void* context) {
    Geco* geco = (Geco*)context;
    int numProcTests = sizeof(tc) / sizeof(tc[0]);

    for (int i=0; i<numProcTests; i++) {
        //if (i == 0) { continue; }

        PreProcessingTests test = tc[i];
        ProcessedTexts res = preprocess_texts(geco->processor, test.origTexts, test.batchSize, test.maxTokens);

        printf("Processed Texts Results:\n");
        printf("  - Number of Texts: %d\n", res.num_texts);
        printf("  - Max Tokens: %d\n", res.max_seq_len);
        printf("  - Texts: [\n");
        for (int j=0; j<res.num_texts; j++) {
            printf("      \"%s\",\n", res.texts[j]);
        }
        printf("    ]\n");
        printf("  - Number of Newlines: %d\n", res.newline_size);
        printf("  - Newline Strings: [\n");
        for (int j=0; j<res.newline_size; j++) {
            printf("      \"");
            prtNewlineLits(res.newline_strs[j]);
            printf("\",\n");
        }
        printf("    ]\n");
        printf("  - Newline Indicies: [ ");
        for (int j=0; j<res.newline_size; j++) {
            if (j!=0) {
                printf(", ");
            }
            printf("%d", res.newline_inds[j]);
        }
        printf(" ]\n\n\n");

    }

} */



//* Gibberish Tests
void compare_gibb_results(double res[MAX_BATCH_SIZE][GIBB_CLASSES], double** exp, int batchSize) {
    printf("Results: \n");
    for (int i = 0; i < batchSize; i++) {
        printf("  Seq #%d: [", i+1);
        for (int j = 0; j < 4; j++) {

            if (fabs(res[i][j] - exp[i][j]) > 0.1) {
                printf("\x1b[1;31m%.1f\x1b[0m, ", res[i][j]);
            } else {
                printf("\x1b[1;32m%.1f\x1b[0m, ", res[i][j]);
            }
        }
        printf("\b\b]\n");
    }
    printf("\n");
}

void test_gibbs(void* geco, int testNum) {
    // Run a gibberish test and time its execution
    TestGibb test = gibbTests[testNum-1];

    double result[MAX_BATCH_SIZE][GIBB_CLASSES];

    // Time the InferModel function
    clock_t start = clock();
    GecoGibb(geco, result, test.texts, test.shape[0]);
    clock_t end = clock();

    // Calculate the time difference
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Gibb Test #%d - Time taken: %.3f seconds\n", testNum, cpu_time_used);

    compare_gibb_results(result, test.exp_scores, test.shape[0]);
    printf("\n");
}
void test_allGibbs(void* geco, int numTests) {
    if ((size_t)numTests > (sizeof(gibbTests) / sizeof(gibbTests[0]))) {
        numTests = sizeof(gibbTests) / sizeof(gibbTests[0]);
    }
    for (int i = 0; i < numTests; i++) {
        test_gibbs(geco, i+1);
    } 
}


//* WordPiece Tokenizer Tests

// Compare WordPiece Tokenizer results
int compTokenOuts(Tokenized_WP_Output res, TestWP exp) {
    int hasError = 0;
    printf("Result:\n  - Length: ");
    if (res.length != (int)(exp.shape[0] * exp.shape[1])) {
        hasError = 1;
        printf("\x1b[1;31m%d\x1b[0m\n", res.length);
    } else {
        printf("\x1b[1;32m%d\x1b[0m\n", res.length);
    }

    printf("  - Shape: ");
    if (res.shape[0] != exp.shape[0] || res.shape[1] != exp.shape[1]) {
        hasError = 1;
        printf("\x1b[1;31m%ld x %ld\x1b[0m\n", res.shape[0], res.shape[1]);
    } else {
        printf("\x1b[1;32m%ld x %ld\x1b[0m\n", res.shape[0], res.shape[1]);
    }

    printf("  - IDs: [");
    for (int i = 0; i < res.length; i++) {
        if (res.ids[i] != exp.ids[i]) {
            hasError = 1;
            printf("\x1b[1;31m%ld\x1b[0m, ", res.ids[i]);
        } else {
            printf("\x1b[1;32m%ld\x1b[0m, ", res.ids[i]);
        }
    }
    printf("]\n");

    printf("  - Attention Mask: [");
    for (int i = 0; i < res.length; i++) {
        if (res.attention_mask[i] != exp.attn_mask[i]) {
            hasError = 1;
            printf("\x1b[1;31m%ld\x1b[0m, ", res.attention_mask[i]);
        } else {
            printf("\x1b[1;32m%ld\x1b[0m, ", res.attention_mask[i]);
        }
    }
    printf("]\n");
    return hasError;
}

/* void tokenizerTests() {
    int numWpTests = sizeof(wp_tests) / sizeof(wp_tests[0]);

    for (int i=0; i<numWpTests; i++) {
        TestWP test = wp_tests[i];
        Tokenized_WP_Output res = batch_gibb_texts(test.texts, test.shape[0]);
        int passFail = compTokenOuts(res, test);
        if (passFail) {
            printf("Test Result: \x1b[1;31mFAILED!\x1b[0m\n\n");
        } else {
            printf("Test Result: \x1b[1;32mPASSED!\x1b[0m\n\n");
        }
        free_tokenized_wp_output(res);
    } 
} */



