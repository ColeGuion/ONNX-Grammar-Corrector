#include "tools.h"
using namespace std;

std::string formatInteger(int number, int mxLen) {
    // Turn number into a string and add a comma with a total length of mxLen characters
    std::ostringstream oss; // Create a string stream
    oss << number << ",";   // Add the number and a comma to the stream

    // Pad the string with spaces to ensure it's at least 5 characters long
    std::string result = oss.str();
    if ((int)result.length() < mxLen) {
        result.append(mxLen - result.length(), ' ');
    }

    return result;
}

void prtRepr(std::string text, int inQuotes) {
    if (inQuotes) {
        cout << "\"";
    }
    for (char c : text) {
        if (c == '\t') {
            cout << "\\t";
        } else if (c == '\n') {
            cout << "\\n";
        } else {
            cout << c;
        }
    }
    if (inQuotes) {
        cout << "\"";
    }
}

void prtPieces(std::vector<int> pieces) {
    cout << "Pieces: [";
    for (int i = 0; i < (int)pieces.size(); ++i) {
        cout << pieces[i] << ", ";
    }
    cout << "\b\b]\n";
}

void prtGroupedIds(std::vector<std::vector<int>> grouped_ids) {
    cout << "Grouped IDs: [\n";
    for (int i = 0; i < (int)grouped_ids.size(); ++i) {
        cout << "   [";
        for (int j = 0; j < (int)grouped_ids[i].size(); ++j) {
            cout << grouped_ids[i][j] << ", ";
        }
        cout << "\b\b],\n";
    }
    cout << "]\n";
}

void prtGroupedTexts(std::vector<std::string> grouped_texts) {
    cout << "Grouped Texts (" << grouped_texts.size() << ")\n";
    for (int i = 0; i < (int)grouped_texts.size(); ++i) {
        cout << "  - \"" << grouped_texts[i] << "\"\n";
    }
    cout << "\n";
}

void prtTokens(TokenizedTexts* output, int num_tokens, int mxLen) {
    cout << "Tokens: [\n   ";
    int xx = 0;
    for (int i = 0; i < num_tokens; ++i) {    //(int)MAX_BATCH_SIZE*MAX_TOKENS
        if (output->ids[i] != 0) {
            xx++;
        }
        cout << formatInteger(output->ids[i], 7);
        //cout << output->ids[i] << ", ";
        if ((i+1) % mxLen == 0) {
            cout << "\n   ";
        }
    }
    cout << "\b\b\b]\n";
    cout << "Num of non-zero tokens: " << xx << "\n\n";
}

void prtNewlineStrings(std::vector<std::string> *nlStrings) {
    cout << "(" << nlStrings->size() << ") Newline Strings: [\n   ";
    for (int i = 0; i < (int)nlStrings->size(); ++i) {
        prtRepr(nlStrings->at(i), 1);
        cout << ",\n   ";
    }
    cout << "\b\b\b]\n";
}

void prtNewlineStrings2(std::vector<std::string> nlStrings) {
    cout << "(" << nlStrings.size() << ") Newline Strings: [\n   ";
    for (int i = 0; i < (int)nlStrings.size(); ++i) {
        prtRepr(nlStrings[i], 1);
        cout << ",\n   ";
    }
    cout << "\b\b\b]\n";
}

void prtFinalText(std::string finalText) {
    cout << "     > Final: ";
    prtRepr(finalText, 1);
    cout << "\n";
}

void prtGibbIds(Tokenized_WP_Output output) {
    cout << "Gibberish IDs: [";
    for (int i=0; i<output.length; i++) {
        cout << output.ids[i] << ", ";
    }
    cout << "\b\b]\n\n";
}

void prtGibbBatches(std::vector<std::vector<size_t>> batch_input_ids) {
    cout << "Batch Gibberish:\n";
    for (size_t i = 0; i < batch_input_ids.size(); ++i) {
        cout << "  Batch[" << i << "] (" << batch_input_ids[i].size() << "): [";
        if (batch_input_ids[i].size() == 0) {
            cout << "]\n";
            continue;
        }
        
        for (size_t j = 0; j < batch_input_ids[i].size(); ++j) {
            cout << batch_input_ids[i][j] << ", ";
        }
        cout << "\b\b]\n";
    }
    cout << "\n";
}

