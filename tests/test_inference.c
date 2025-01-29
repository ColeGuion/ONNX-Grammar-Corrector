#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <json-c/json.h>
#include "unity.h"
#include "onnxruntime_c_api.h"
#include "../inference.h"

int useGpu = 1; // Set to 1 for GPU, 0 for CPU
void* geco = NULL;

// Initialize common test dependencies
void setUp(void) {
    GecoConfig config = {false, 2, "0", false};
    geco = NewGeco(config);
    
    // Set log level
    LOG_LEVEL = INFO;
}

//TODO: Fix test_encoding.c and test_tokenizer.c
// Free the memory allocated by the tokenizer and processor
void tearDown(void) {
    FreeGeco(geco);
}

void test_gec1(void) {
    int num_texts = 5;
    char* texts[5] = {
        "He plays the guitar well.",
        "He did well in the exam.",
        "She speaks English well.",
        "We don't know our neighbor very well.",
        "He did the job well."
    };
    /* char* expected_results[5] = {
        "He plays the guitar well.",
        "He did well in the exam.",
        "She speaks English well.",
        "We don't know our neighbor very well.",
        "He did the job well."
    }; */
    char* expected = "He plays the guitar well. He did well in the exam. She speaks English well. We don't know our neighbor very well. He did the job well.";

    char* result = GecoRun(geco, texts, num_texts);
    TEST_ASSERT_EQUAL_STRING(expected, result);
    if (result) free(result);
}

void test_gec2(void) {
    int num_texts = 3;
    char* texts[3] = {
        "Hello world?",
        "Sharon is the biggest pool lover I know?",
        "Sharon is the nicest person ever, Before she had time to think about it Sharon jumped in an icy pool"
    };
    char* expected = "Hello world, Sharon is the biggest pool lover I know. Sharon is the nicest person ever, Before she had time to think about it, Sharon jumped into an icy pool.";

    char* result = GecoRun(geco, texts, num_texts);
    TEST_ASSERT_EQUAL_STRING(expected, result);
    if (result) free(result);
}

void test_gec3(void) {
    int num_texts = 2;
    char* texts[2] = {
        "If you loook toward tthe westduring sunset and and see a red sky, it mean there is dry aiir and dust particles becase of high-pressure sytem.",
        "This dry airis moving towards you, so there won't be any rain, but the wind whill come soon.",
    };
    char* expected = "If you look toward the west during sunset and see a red sky, it means there is dry air and dust particles becase of high-pressure systems. This dry air is moving towards you, so there won't be any rain, but the wind will come soon.";

    char* result = GecoRun(geco, texts, num_texts);
    TEST_ASSERT_EQUAL_STRING(expected, result);
    if (result) free(result);
}

// Test #4 (Essay 1)
void test_gec4(void) {
    int num_texts = 10;
    char* texts[10] = {
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
    };
    char* expected = "Directional selection is one of the three types of natural selection that selects one extreme of a trait. When a flower produces more nectar or seeds so that its population grows, it is more successful. When a population is larger, it's harder to change the genetic traits of the population. When the population is smaller, it's easier to directionalize genes to be more successful. Instead of balancing an average of all traits, organisms choose an extreme of one trait, and that causes animals to behave in such a direction. In the example, the flowers are evovled to attract more pollinators to carry around the specific gene so that all members of a population receive that gene, making the population more successful. However, since the population is evovled to one gene, this makes it more susceptible to diseases or unhealthy factors that said organism didn't evolve to. I don't really know what I'm talking about, and I'm kind of frustrated. I don't really want to write about this anymore, and I already have a good grade in this class. All I want to do is get over it and say that organisms evolve really fast with directional selection.";

    char* result = GecoRun(geco, texts, num_texts);
    TEST_ASSERT_EQUAL_STRING(expected, result);
    if (result) free(result);
}

// Test #5 (Essay 4)
void test_gec5(void) {
    int num_texts = 12;
    char* texts[12] = {
        "There are countless different species thriving in this world, including humans.",
        "But due to the rapid expansion and population growth of humans, species in different ecosystems are threatened.",
        "Habitats and food sources are destroyed to clear a way for human structures.",
        "But, there are efforts by humans to preserve the biodivserity observed within our planet, to make it a home for all life.",
        "Orangutans tend to live in tree, and venture on the ground only for water and some food.",
        "When logging and deforestation occurs, the habitats are destroyed.",
        "Poaching and trophy hunting have a massive negative impact.",
        "Orangutans are very slow in reproduction, and like humans, have nine-month gestation periods.",
        "After a female births a baby, it take from six to nine years for her to able to have another baby, and twins are extremely rare.",
        "So even if all habitat and food source destruction, or illegal hunting and killing stops, it will take many years for numbers to rejuvenate.",
        "Measures to ensure that cetain habitat areas are reserved for wildlife take place, and animals are taken to rescue facilities, and zoos, to protect them, and study their behavior.",
        "All three species of orangutans are critically endangered and many conservation measures are constantly being taken to defend these creatures now and in the future."
    };
    char* expected = "There are countless different species thriving in this world, including humans. But due to the rapid expansion and population growth of humans, species in different ecosystems are threatened. Habitats and food sources are destroyed to clear a way for human structures. But, there are efforts by humans to preserve the biodiversity observed within our planet, to make it a home for all life. Orangutans tend to live in trees and venture on the ground only for water and some food. When logging and deforestation occurs, the habitats are destroyed, and poaching and trophy hunting have a massive negative impact. Orangutans are very slow in reproduction, and like humans, have nine-month gestation periods. After a woman births a baby, it takes from six to nine years for her to be able to have another baby, and twins are extremely rare. So even if all habitat and food source destruction, or illegal hunting and killing, it will take many years for numbers to rejuvenate. Measures to ensure that cetain habitat areas are reserved for wildlife have been taken place, and animals are taken to rescue facilities and zoos, to protect them, and study their behavior. All three species of orangutans are critically endangered, and many conservation measures are constantly being taken to defend these creatures now and in the future.";
    
    char* result = GecoRun(geco, texts, num_texts);
    TEST_ASSERT_EQUAL_STRING(expected, result);
    if (result) free(result);
}


int main(int argc, char* argv[]) {
    UNITY_BEGIN();
    RUN_TEST(test_gec1);
    RUN_TEST(test_gec2);
    RUN_TEST(test_gec3);
    RUN_TEST(test_gec4);
    RUN_TEST(test_gec5);
    UNITY_END();
    return 0;
}

