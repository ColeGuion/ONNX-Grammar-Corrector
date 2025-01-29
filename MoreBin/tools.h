#ifndef TOOLS_H
#define TOOLS_H

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include "../sentencepiece_wrapper.h"


// Function declarations
std::string formatInteger(int number, int mxLen);
void prtRepr(std::string text, int inQuotes);
void prtPieces(std::vector<int> pieces);
void prtGroupedIds(std::vector<std::vector<int>> grouped_ids);
void prtGroupedTexts(std::vector<std::string> grouped_texts);
void prtTokens(TokenizedTexts* output, int num_tokens, int mxLen);
void prtNewlineStrings(std::vector<std::string>* nlStrings);
void prtNewlineStrings2(std::vector<std::string> nlStrings);
void prtFinalText(std::string finalText);
void prtGibbIds(Tokenized_WP_Output output);
void prtGibbBatches(std::vector<std::vector<size_t>> batch_input_ids);

#endif // TOOLS_H
